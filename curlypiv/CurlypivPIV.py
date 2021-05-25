# test CurlypivImageCollection
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import sys
import os
from os import listdir
from os.path import isfile, basename, join, isdir
from collections import OrderedDict
import glob

# quality control and debugging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Maths/Scientifics
import numpy as np
import numpy.ma as ma
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage as scn
import pandas as pd

# Image Processing
import cv2 as cv
from skimage import io
import matplotlib.pyplot as plt


# Curlypiv
import curlypiv
from curlypiv.CurlypivPIVSetup import CurlypivPIVSetup
from curlypiv.CurlypivFile import CurlypivFile
from curlypiv.CurlypivUtils import find_substring, get_sublevel
from curlypiv.CurlypivPlotting import plot_quiver, plot_per_loc, plot_quiver_and_u_mean, plot_u_mean_columns

# OpenPIV
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv"))
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv/openpiv"))
from openpiv import *
from windef import Settings
import openpiv.piv
from openpiv import windef
from openpiv.windef import Settings, create_deformation_field
from openpiv import tools, scaling, validation, filters, preprocess
from openpiv.pyprocess import extended_search_area_piv, get_field_shape, get_coordinates
from openpiv import smoothn
from openpiv.preprocess import mask_coordinates


# 2.0 define class
class CurlypivPIV(object):

    def __init__(self, testCollection, testSetup, pivSetup=None,
                 bpespecs=None, cropspecs=None, resizespecs=None, filterspecs=None, backsubspecs=None,
                 init_frame=100, backSub_init_frames=50, num_analysis_frames=30, img_piv='filtered', img_piv_plot='filtered',
                 piv_mask=None, exclude=[]):

        if not isinstance(testCollection, curlypiv.CurlypivTestCollection):
            raise ValueError("Specified test collection {} is not a CurlypivTestCollection".format(testCollection))

        if not isinstance(testSetup, curlypiv.CurlypivTestSetup):
            raise ValueError("Specified test collection {} is not a CurlypivTestSetup".format(testSetup))

        self.collection = testCollection
        self.setup = testSetup
        self.pivSetup = pivSetup

        self.bpespecs = bpespecs
        self.cropspecs = cropspecs
        self.resizespecs = resizespecs
        self.filterspecs = filterspecs
        self.backsubspecs = backsubspecs

        self.bg_method = backsubspecs        #backsubspecs.get('bg_method', None)
        self.backSub = None
        self.img_piv = img_piv
        self.init_frame = init_frame
        self.backSub_init_frames = backSub_init_frames
        self.num_analysis_frames = num_analysis_frames
        self.img_piv_plot = img_piv_plot

        # experimental
        self.L_bpe = testSetup.chip.bpe.length
        self.L_channel = testSetup.chip.channel.length

        # mask PIV velocity vectors
        self.piv_mask = piv_mask    # example: piv_mask = 'bpe'

    def piv_analysis(self, level,
                     save_texts=False, save_plots=False, show_plots=False,
                     loc=None, test=None, run=None, seq=None, file=None, calc_zeta=False):

        if level == 'all':
            for locs in self.collection.locs.values():
                if self.bg_method['bg_method'] == 'KNN':
                    self.backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
                for tests in locs.tests.values():
                    for runs in tests.runs.values():
                        seq0_bkg = 0
                        for seqs in runs.seqs.values():

                            # --- per seq sampling: ---

                            # refresh files in each sequence to make sure they are up to date and loaded
                            seqs.refresh_files()

                            # if flatfield/darkfield, correct images prior to background subtraction
                            if self.backsubspecs['darkfield'] is not None:
                                seqs.apply_flatfield_correction(self.backsubspecs['darkfield'], self.backsubspecs['flatfield'])

                            # if min/mean background subtraction used, calculate background image and write to file
                            if self.bg_method['bg_method'] in ['min', 'mean']:
                                seqs.calculate_background_image(bg_method=self.bg_method['bg_method'])

                            # perform PIV on sequence
                            self.piv(seqs, testname=(tests.name[0]*1e-3/self.L_channel, tests.name[1]), seqname=seqs.name, u_mag_bkg=seq0_bkg)
                            print('Loc: {}, Test: {}, Run: {}, Sequence: {}'.format(locs.name, tests.name, runs.name,seqs.name))

                        # per run sampling
                        if calc_zeta:
                            runs.add_piv_data(zeta=True)
                        else:
                            runs.add_piv_data(zeta=False, testname=(tests.name[0]*1e-3/self.L_channel, tests.name[1]))

                    # per test sampling
                    if calc_zeta:
                        tests.add_piv_data(zeta=True)
                    else:
                        tests.add_piv_data(zeta=False, testname=(tests.name[0]*1e-3/self.L_channel, tests.name[1]))
                        plot_u_mean_columns(tests, plot_value='u', testname=(tests.name[0]*1e-3/self.L_channel, tests.name[1]), num_analysis_frames=self.num_analysis_frames, pivSetup=self.pivSetup)
                    print(tests)

                # per loc sampling
                if calc_zeta:
                    plot_per_loc(locs)


    def piv(self, seqs,
            locname=None, testname=None, runname=None, seqname=None,
            u_mag_bkg=0):

        if self.pivSetup.calculate_zeta:
            u_mag_means = []
            u_mag_stds = []
            u_means = []
            v_means = []
            u_bkgs = []
        else:
            u_means = []
            u_stds = []


        img_b_list = list(seqs.files.values())
        counter = 0

        for filename, img in seqs.files.items():

            # break condition
            if counter > self.init_frame + self.backSub_init_frames + self.num_analysis_frames:
                continue

            # ----- Image Pre-Processing -----
            elif counter < self.init_frame:
                pass

            elif counter <= self.init_frame + self.backSub_init_frames:

                # flatfield correction (THIS FUNCTION IS NOW APPLIED PRIOR TO BACKGROUND SUBTRACTION)
                #if self.setup.optics.microscope.ccd.darkfield.img is not None:
                #    img.apply_flatfield_correction(darkfield=self.setup.optics.microscope.ccd.darkfield.img, flatfield=self.setup.optics.microscope.illumination.flatfield)

                # manual background subtraction should be before image cropping and filtering
                if self.backsubspecs['bg_method'] in ['min', 'mean']:
                    img.image_subtract_background(image_input='raw', backgroundSubtractor=self.backSub, bg_method=self.bg_method['bg_method'], bg_filepath=seqs.img_background)

                # bpe region filtering
                if self.bpespecs:
                    img.image_bpe_filter(bpespecs=self.bpespecs)

                # crop
                if self.cropspecs:
                    img.image_crop(cropspecs=self.cropspecs)

                # resize
                if self.resizespecs:
                    img.image_resize(resizespecs=self.resizespecs)

                # filter
                if self.filterspecs:
                    img.image_filter(filterspecs=self.filterspecs, image_input='raw', image_output='filtered', force_rawdtype=True)

                # algorithmic background subtraction should be last
                if self.backsubspecs['bg_method'] in ['KNN', 'MOG2', 'CMG']:
                    img.image_subtract_background(image_input='filtered', backgroundSubtractor=self.backSub, bg_method=self.bg_method['bg_method'])

            if counter >= self.init_frame + self.backSub_init_frames:
                img_b = img_b_list[counter + 1]

                for im in [img_b]:

                    # flatfield correction (THIS FUNCTION IS NOW APPLIED PRIOR TO BACKGROUND SUBTRACTION)
                    #if self.setup.optics.microscope.ccd.darkfield.img is not None:
                    #    im.apply_flatfield_correction(darkfield=self.setup.optics.microscope.ccd.darkfield.img, flatfield=self.setup.optics.microscope.illumination.flatfield)

                    # manual background subtraction should be before image cropping and filtering
                    if self.backsubspecs['bg_method'] in ['min', 'mean']:
                        im.image_subtract_background(image_input='raw', backgroundSubtractor=self.backSub, bg_method=self.bg_method['bg_method'], bg_filepath=seqs.img_background)

                    # bpe region filtering
                    if self.bpespecs:
                        im.image_bpe_filter(bpespecs=self.bpespecs)

                    # crop
                    if self.cropspecs:
                        im.image_crop(cropspecs=self.cropspecs)

                    # resize
                    if self.resizespecs:
                        im.image_resize(resizespecs=self.resizespecs)

                    # filter
                    if self.filterspecs:
                        im.image_filter(filterspecs=self.filterspecs, image_input='raw', image_output='filtered', force_rawdtype=True)

                    # subtract background
                    if self.backsubspecs['bg_method'] in ['KNN', 'MOG2', 'CMG']:
                        im.image_subtract_background(image_input='filtered', backgroundSubtractor=self.backSub, bg_method=self.bg_method['bg_method'])


                # 3.1.4 - Start First Pass PIV
                if self.img_piv == 'raw':
                    x, y, u, v, s2n = windef.first_pass(img.raw, img_b.raw, self.pivSetup.settings)
                if self.img_piv == 'filtered':
                    x, y, u, v, s2n = windef.first_pass(img.filtered, img_b.filtered, self.pivSetup.settings)
                if self.img_piv == 'bg':
                    x, y, u, v, s2n = windef.first_pass(img.bg, img_b.bg, self.pivSetup.settings)
                if self.img_piv == 'bgs':
                    x, y, u, v, s2n = windef.first_pass(img.bgs, img_b.bgs, self.pivSetup.settings)
                if self.img_piv == 'masked':
                    x, y, u, v, s2n = windef.first_pass(img.masked,img_b.masked,self.pivSetup.settings)

                if np.isnan(u[0][0]) is True:
                    print("PIV First-Pass gives no results: (u,v) = Nan")
                    #raise KeyboardInterrupt

                # Masking
                if self.pivSetup.settings.image_mask and self.piv_mask is not None:
                    if self.piv_mask == 'bpe':
                        image_mask = np.logical_and(img.bpe_mask, img_b.bpe_mask)
                        min_length=2
                    else:
                        image_mask = np.logical_and(img.mask, img_b.mask)
                        min_length = 10
                    mask_coords = preprocess.mask_coordinates(image_mask, min_length=min_length)
                    # mark those points on the grid of PIV inside the mask
                    grid_mask = preprocess.prepare_mask_on_grid(x, y, mask_coords)

                    # mask the velocity
                    u = np.ma.masked_array(u, mask=grid_mask)
                    v = np.ma.masked_array(v, mask=grid_mask)
                else:
                    mask_coords = []
                    u = np.ma.masked_array(u, mask=np.ma.nomask)
                    v = np.ma.masked_array(v, mask=np.ma.nomask)

                if self.pivSetup.settings.validation_first_pass:
                    u, v, mask = validation.typical_validation(u, v, s2n, self.pivSetup.settings)

                if self.pivSetup.settings.num_iterations == 1 and self.pivSetup.settings.replace_vectors:
                    if self.pivSetup.replace_Nans_with_zeros is True:
                        u, v, = self.replace_with_zeros(u, v)
                    else:
                        u, v = filters.replace_outliers(u,v, method=self.pivSetup.settings.filter_method,
                                                        max_iter=self.pivSetup.settings.max_filter_iteration,
                                                        kernel_size=self.pivSetup.settings.filter_kernel_size)
                elif self.pivSetup.settings.num_iterations > 1:
                    if self.pivSetup.replace_Nans_with_zeros is True:
                        u, v, = self.replace_with_zeros(u, v)
                    else:
                        u, v = filters.replace_outliers(u,v, method=self.pivSetup.settings.filter_method,
                                                        max_iter=self.pivSetup.settings.max_filter_iteration,
                                                        kernel_size=self.pivSetup.settings.filter_kernel_size)

                if self.pivSetup.settings.smoothn:
                    u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(u, s=self.pivSetup.settings.smoothn_p)
                    v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(v, s=self.pivSetup.settings.smoothn_p)

                if not isinstance(u, np.ma.MaskedArray):
                    raise ValueError("Expected masked array")

                # 3.2.0 - Run multi pass windows deformation loop
                for current_iteration in range(0, self.pivSetup.settings.num_iterations):
                    if self.img_piv == 'raw':
                        x, y, u, v, s2n, mask = windef.multipass_img_deform(img.raw, img_b.raw, current_iteration,
                                                                            x, y, u, v,
                                                                            self.pivSetup.settings,
                                                                            mask_coords=mask_coords)
                    if self.img_piv == 'filtered':
                        x, y, u, v, s2n, mask = self.custom_multipass_img_deform(img.filtered, img_b.filtered, current_iteration,
                                                                            x, y, u, v, mask_coords=mask_coords)
                    if self.img_piv == 'bg':
                        x, y, u, v, s2n, mask = windef.multipass_img_deform(img.bg, img_b.bg, current_iteration,
                                                                            x, y, u, v,
                                                                            self.pivSetup.settings,
                                                                            mask_coords=mask_coords)
                    if self.img_piv == 'bgs':
                        x, y, u, v, s2n, mask = windef.multipass_img_deform(img.bgs, img_b.bgs, current_iteration,
                                                                            x, y, u, v,
                                                                            self.pivSetup.settings,
                                                                            mask_coords=mask_coords)
                    if self.img_piv == 'masked':
                        x, y, u, v, s2n, mask = windef.multipass_img_deform(img.masked, img_b.masked, current_iteration,
                                                                            x, y, u, v,
                                                                            self.pivSetup.settings,
                                                                            mask_coords=mask_coords)

                # If the smoothing is active, we do it at each pass
                # but not the last one
                if self.pivSetup.settings.smoothn is True and current_iteration < self.pivSetup.settings.num_iterations - 1:
                    u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(u, s=self.pivSetup.settings.smoothn_p)
                    v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(v, s=self.pivSetup.settings.smoothn_p)

                if not isinstance(u, np.ma.MaskedArray):
                    raise ValueError('not a masked array anymore')

                # Replace Nan with zeros
                u = u.filled(0.)
                v = v.filled(0.)

                # 3.2.2 - Adjust scaling
                u = u / self.pivSetup.settings.dt
                v = v / self.pivSetup.settings.dt
                x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=self.pivSetup.settings.scaling_factor)

                # calculate the PIV stats
                if self.pivSetup.calculate_zeta:
                    img.calculate_stats_zeta(u, v)

                    if seqname == 0:
                        if img.u_mean < 0:
                            u_bkgs.append(-1*img.M_mean)
                        else:
                            u_bkgs.append(img.M_mean)

                    if seqname > 0:
                        u_mag_means.append(img.M_mean - u_mag_bkg)
                    else:
                        u_mag_means.append(img.M_mean)
                    u_mag_stds.append(img.M_std)
                    u_means.append(img.u_mean)
                    v_means.append(img.v_mean)
                else:
                    # mask non-BPE regions
                    u_zeros_masked_array = ma.masked_where(u == 0, u)
                    u_far_left_masked_array = u_zeros_masked_array[:, :3]
                    u_centerline_masked_array = u_zeros_masked_array[:, 5:6]
                    u_far_right_masked_array = u_zeros_masked_array[:, 8:]

                    # mask BPE regions
                    u_pos_left_masked_array = ma.masked_less_equal(u, 0)
                    u_pos_left_masked_array2 = u_pos_left_masked_array[:,3:5]
                    u_neg_right_masked_array = ma.masked_greater_equal(u, 0)
                    u_neg_right_masked_array2 = u_neg_right_masked_array[:,6:8]

                    # concatenate all the masks together
                    u_dir_masked_array = ma.concatenate([u_far_left_masked_array, u_pos_left_masked_array2,
                                                         u_centerline_masked_array,
                                                         u_neg_right_masked_array2, u_far_right_masked_array], axis=1)

                    u_mask = np.logical_and(u_zeros_masked_array.mask, u_dir_masked_array.mask)
                    #u_masked_array = ma.array(u, mask=u_dir_masked_array.mask)
                    u_masked_array = u_dir_masked_array
                    u_masked_plot = ma.filled(u_masked_array, fill_value=0)

                    # calculate mean and std
                    u_mean_masked = np.mean(u_masked_array, axis=0)
                    u_mean = u_mean_masked.data
                    u_means.append(u_mean)
                    u_std_masked = np.std(u_masked_array, axis=0)
                    u_std = u_std_masked.data
                    u_stds.append(u_std)


                if (self.pivSetup.save_plot or self.pivSetup.show_plot) and (self.pivSetup.calculate_zeta):
                    plot_quiver(x, y, u, v, img, self.pivSetup, img_piv_plot=self.img_piv_plot,
                                u_mag_mean = img.M_mean, u_mag_std = img.M_std,
                                locname=locname, testname=testname, runname=runname, seqname=seqname)
                elif self.pivSetup.save_plot or self.pivSetup.save_u_mean_plot or self.pivSetup.show_plot:
                    plot_quiver_and_u_mean(x, y, u_masked_plot, v, img, self.pivSetup, img_piv_plot=self.img_piv_plot,
                                           u_mean_columns=u_mean, locname=locname, testname=testname, runname=runname, seqname=seqname)

                # empty the file to reduce RAM storage
                img.empty_file(to_empty='all')

            counter += 1

        # empty files from seq to reduce RAM storage
        seqs.remove_files(file='all')

        if self.pivSetup.calculate_zeta:
            seq_u_mag_mean = np.round(np.mean(u_mag_means),1)
            seq_u_mag_std = np.round(np.mean(u_mag_stds),2)
            seq_u_means = np.round(np.mean(u_means),1)
            seq_v_means = np.round(np.mean(v_means),1)
            if len(u_bkgs) > 1:
                seq_u_bkgs = np.round(np.mean(u_bkgs),1)
            else:
                seq_u_bkgs = None
            seqs.add_piv_data(zeta=True, u_mag_mean=seq_u_mag_mean, u_mag_std=seq_u_mag_std, u_mean=seq_u_means,
                              v_mean=seq_v_means, u_mag_bkg=seq_u_bkgs)
        else:
            seq_u_mean = np.round(np.mean(u_means, axis=0), 2)
            seq_u_mean_std = np.round(np.std(u_stds, axis=0), 2)
            seqs.add_piv_data(zeta=False,  u_mean_x=x[0,:], u_mean_columns=seq_u_mean, u_mean_columns_std=seq_u_mean_std, testname=testname)



    def custom_multipass_img_deform(self, frame_a, frame_b, current_iteration,
                                    x_old, y_old, u_old, v_old, mask_coords):

        # inits
        old_frame_a = frame_a.copy()
        old_frame_b = frame_b.copy()

        # get coordinates
        x, y = get_coordinates(frame_a.shape, self.pivSetup.settings.windowsizes[current_iteration],
                               self.pivSetup.settings.overlap[current_iteration])

        # fix coordinate system and prepare for RectBivariateSpline
        y_old = y_old[:, 0]
        x_old = x_old[0, :]
        y_int = y[:, 0]
        x_int = x[0, :]

        # interpolating the displacements from the old grid onto the new grid
        ip = RectBivariateSpline(y_old, x_old, u_old.filled(0.))
        u_pre = ip(y_int, x_int)
        ip2 = RectBivariateSpline(y_old, x_old, v_old.filled(0.))
        v_pre = ip2(y_int, x_int)

        # Image deformation has to occur in image coordinates
        if self.pivSetup.settings.deformation_method == "symmetric":
            x_new, y_new, ut, vt = create_deformation_field(frame_a, x, y, u_pre, v_pre)
            frame_a = scn.map_coordinates(frame_a, ((y_new - vt / 2, x_new - ut / 2)), order=self.pivSetup.settings.interpolation_order, mode='nearest')
            frame_b = scn.map_coordinates(frame_b, ((y_new + vt / 2, x_new + ut / 2)), order=self.pivSetup.settings.interpolation_order, mode='nearest')
        else:
            raise Exception("Deformation method is not valid.")

        # compute cross correlation
        u, v, s2n = extended_search_area_piv(
            frame_a,
            frame_b,
            window_size=self.pivSetup.settings.windowsizes[current_iteration],
            overlap=self.pivSetup.settings.overlap[current_iteration],
            width=self.pivSetup.settings.sig2noise_mask,
            subpixel_method=self.pivSetup.settings.subpixel_method,
            sig2noise_method=self.pivSetup.settings.sig2noise_method,  # if it's None, it's not used
            correlation_method=self.pivSetup.settings.correlation_method,
            normalized_correlation=self.pivSetup.settings.normalized_correlation,
        )

        shapes = np.array(get_field_shape(frame_a.shape, self.pivSetup.settings.windowsizes[current_iteration], self.pivSetup.settings.overlap[current_iteration]))
        u = u.reshape(shapes)
        v = v.reshape(shapes)
        s2n = s2n.reshape(shapes)

        u += u_pre
        v += v_pre

        # reapply the image mask to new grid
        if self.pivSetup.settings.image_mask:
            grid_mask = preprocess.prepare_mask_on_grid(x, y, mask_coords)
            u = np.ma.masked_array(u, mask=grid_mask)
            v = np.ma.masked_array(v, mask=grid_mask)
            # validate
            u, v, mask = validation.typical_validation(u, v, s2n, self.pivSetup.settings)
        else:
            u = np.ma.masked_array(u, mask=np.ma.nomask)
            v = np.ma.masked_array(v, mask=np.ma.nomask)

        # we must replace outliers
        if self.pivSetup.settings.replace_vectors:
            if self.pivSetup.replace_Nans_with_zeros is True:
                u, v, = self.replace_with_zeros(u, v)
            else:
                u, v = filters.replace_outliers(u, v, method=self.pivSetup.settings.filter_method, max_iter=self.pivSetup.settings.max_filter_iteration, kernel_size=self.pivSetup.settings.filter_kernel_size)

        # reapply the image mask to new grid
        if self.pivSetup.settings.image_mask:
            grid_mask = preprocess.prepare_mask_on_grid(x, y, mask_coords)
            u = np.ma.masked_array(u, mask=grid_mask)
            v = np.ma.masked_array(v, mask=grid_mask)
        else:
            u = np.ma.masked_array(u, mask=np.ma.nomask)
            v = np.ma.masked_array(v, mask=np.ma.nomask)

        if self.pivSetup.settings.image_mask is False:
            mask = []

        return x, y, u, v, s2n, mask


    def get_analysis_level(self, level='seq',
                           loc=None, test=None, run=None, seq=None, file=None):

        lev = self.collection

        valid_levels = ['all', 'loc', 'test', 'run', 'seq', 'file']

        if level not in valid_levels:
            raise ValueError(
                "Specified analysis level {} is not one of the valid levels: {}".format(level, valid_levels))

        if level == 'file':

            if None in [loc, test, run, seq, file]:
                raise ValueError("Must specify: loc, test, run, seq, and file for file-level analysis")
            levels = [loc, test, run, seq, file]

        if level == 'seq':

            if None in [loc, test, run, seq]:
                raise ValueError("Must specify: loc, test, run, and seq for seq-level analysis")
            levels = [loc, test, run, seq]

        if level == 'run':

            if None in [loc, test, run]:
                raise ValueError("Must specify: loc, test, and run for run-level analysis")
            levels = [loc, test, run]

        if level == 'test':

            if None in [loc, test]:
                raise ValueError("Must specify: loc, and test for test-level analysis")
            levels = [loc, test]

        if level == 'loc':

            if None in [loc]:
                raise ValueError("Must specify loc for loc-level analysis")
            levels = [loc]

        if level == 'all':
            pass
        else:
            for l in levels:
                #lev = get_sublevel(lev, l)
                lev = lev.get_sublevel(l)
                #sub = fileCollection.get_sublevel(key)

        return lev

    def replace_with_zeros(self, u, v):
        uf = np.nan_to_num(u)
        vf = np.nan_to_num(v)

        return uf, vf


