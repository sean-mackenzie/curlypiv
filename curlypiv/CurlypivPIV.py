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
import curlypiv.CurlypivImageProcessing as CurlypivImageProcessing

# OpenPIV
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv"))
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv/openpiv"))
from openpiv import *
from openpiv import windef
from openpiv.windef import Settings, create_deformation_field
from openpiv import tools, scaling, validation, filters, preprocess
from openpiv.pyprocess import extended_search_area_piv, get_field_shape, get_coordinates
from openpiv import smoothn
# useless comment



# 2.0 define class
class CurlypivPIV(object):

    def __init__(self, testCollection, testSetup, pivSetup=None,
                 bpespecs=None, cropspecs=None, resizespecs=None, filterspecs=None, backsubspecs=None,
                 init_frame=10, backSub_init_frames=10, num_analysis_frames=20, img_piv='filtered', img_piv_plot='filtered',
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
                if self.backsubspecs['bg_method'] == 'KNN':
                    self.backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
                else:
                    self.backSub = None
                for tests in locs.tests.values():
                    for runs in tests.runs.values():
                        seq0_bkg = 0
                        for seqs in runs.seqs.values():

                            # --- per seq sampling: ---

                            # refresh files in each sequence to make sure they are up to date and loaded
                            seqs.refresh_files()

                            # ------------------------------------------------------------------------------------------
                            # TODO: Simplify this test-process section into a sub-function?
                            # if flatfield/darkfield, correct images prior to background subtraction
                            if self.backsubspecs['darkfield'] is not None:
                                seqs.apply_flatfield_correction(self.backsubspecs['darkfield'], self.backsubspecs['flatfield'], plot_flatfield_correction=self.backsubspecs['show_flatfield_correction'])

                            # if min/mean background subtraction used, calculate background image and write to file
                            if self.backsubspecs['bg_method'] in ['min', 'mean']:
                                seqs.calculate_background_image(bg_method=self.backsubspecs['bg_method'], plot_background_subtraction=self.backsubspecs['show_backsub'])

                            if self.filterspecs['show_filtering']:
                                img_, img_bpe_mask_ = CurlypivImageProcessing.img_apply_bpe_filter(img=seqs.first_file.raw, bpespecs=self.bpespecs)
                                # TODO: Ensure that applying the bpe filter is being applied correctly
                                curlypiv.CurlypivPlotting.plot_image_process(img_before=img_, img_after=img_bpe_mask_, plot_type='process')
                            # ------------------------------------------------------------------------------------------

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
                        tests.add_piv_data(zeta=False, testname=(tests.name[0] * 1e-3 / self.L_channel, tests.name[1]))

                        # ----------------------------------------------------------------------------------------------
                        # TODO: The number of analysis frames needs to an addition step for tests of multiple sequences. Currently, the number of frames in the last sequence is printed as the results.
                        plot_u_mean_columns(tests, plot_value='u', leftedge=self.bpe_leftedge, rightedge=self.bpe_rightedge, testname=(tests.name[0]*1e-3/self.L_channel, tests.name[1]), num_analysis_frames=self.num_analysis_frames, pivSetup=self.pivSetup)
                        plot_u_mean_columns(tests, plot_value='mobility', leftedge=self.bpe_leftedge, rightedge=self.bpe_rightedge, testname=(tests.name[0] * 1e-3 / self.L_channel, tests.name[1]), num_analysis_frames=self.num_analysis_frames, pivSetup=self.pivSetup)
                        # ----------------------------------------------------------------------------------------------

                    print(tests)

                # per loc sampling
                if calc_zeta:
                    plot_per_loc(locs)

        print("Successful completion of PIV analysis.")


    def piv(self, seqs,
            locname=None, testname=None, runname=None, seqname=None,
            u_mag_bkg=0):

        # --------------------------------------------------------------------------------------------------------------
        # TODO: There's definitely a better implementation of the caculate_zeta_PIV vs. calculate_u_column_means.
        # maybe they should be different PIV functions entirely? This would greatly simplify code readability at the
        # expense of some code duplication.

        # setup lists for data storage
        piv_data_out = []
        if self.pivSetup.calculate_zeta:
            u_mag_means = []
            u_mag_stds = []
            u_means = []
            v_means = []
            u_bkgs = []
        else:
            u_means = []
            u_stds = []
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # TODO: Fix number of analysis frames so it uses the number of frames in the sequence only if there are less than the input num analysis frames.
        img_a_list = len(list(seqs.files.values()))
        img_b_list = list(seqs.files.values())
        imgs_in_sequence = len(img_b_list)
        if imgs_in_sequence > 15:
            self.num_analysis_frames = np.round(imgs_in_sequence - self.init_frame - self.backSub_init_frames - 6, -1)
        else:
            self.num_analysis_frames = imgs_in_sequence - self.init_frame - self.backSub_init_frames - 2
        # --------------------------------------------------------------------------------------------------------------

        counter = 0

        for filename, img in seqs.files.items():

            if counter % 100 == 0:
                print("PIV processing at counter: {}".format(counter))

            # break condition
            if counter > self.init_frame + self.backSub_init_frames + self.num_analysis_frames:
                print("init frame + backsub init + num analysis frames: {}".format(self.init_frame + self.backSub_init_frames + self.num_analysis_frames))
                print("Counter at continue statement: {}".format(counter))
                self.num_analysis_frames = counter - self.init_frame + self.backSub_init_frames - 1
                break

            # ----- Image Pre-Processing -----
            elif counter < self.init_frame:
                pass

            elif counter <= self.init_frame + self.backSub_init_frames:
                # NOTE: It's important that init_frame + backSub_init_frames >= 1, otherwise, the first image will not
                # get processed and PIV will be performed on conflicting images or it will error.

                # ------------------------------------------------------------------------------------------------------
                # TODO: Ensure the flatfield image correction is be applied correctly and enable.
                # flatfield correction (THIS FUNCTION IS NOW APPLIED PRIOR TO BACKGROUND SUBTRACTION)
                #if self.setup.optics.microscope.ccd.darkfield.img is not None:
                #    img.apply_flatfield_correction(darkfield=self.setup.optics.microscope.ccd.darkfield.img, flatfield=self.setup.optics.microscope.illumination.flatfield)
                # ------------------------------------------------------------------------------------------------------

                # manual background subtraction should be before image cropping and filtering
                if self.backsubspecs['bg_method'] in ['min', 'mean']:
                    img.image_subtract_background(image_input='raw', backgroundSubtractor=self.backSub, bg_method=self.backsubspecs['bg_method'], bg_filepath=seqs.img_background, plot_background_subtraction=False)

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
                    img.image_subtract_background(image_input='filtered', backgroundSubtractor=self.backSub, bg_method=self.backsubspecs['bg_method'],  plot_background_subtraction=False)

            if counter >= self.init_frame + self.backSub_init_frames:

                img_b = img_b_list[counter + 1]

                for im in [img_b]:

                    # manual background subtraction should be before image cropping and filtering
                    if self.backsubspecs['bg_method'] in ['min', 'mean']:
                        im.image_subtract_background(image_input='raw', backgroundSubtractor=self.backSub, bg_method=self.backsubspecs['bg_method'], bg_filepath=seqs.img_background,  plot_background_subtraction=False)

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
                        im.image_subtract_background(image_input='filtered', backgroundSubtractor=self.backSub, bg_method=self.backsubspecs['bg_method'],  plot_background_subtraction=False)

                # 3.1.4 - Start First Pass PIV
                # ------------------------------------------------------------------------------------------------------
                # TODO: reduce this to one simple line using a custom windef.first_pass function.
                if self.img_piv == 'raw':
                    x, y, u, v, s2n = windef.first_pass(img.raw, im.raw, self.pivSetup.settings)
                if self.img_piv == 'filtered':
                    x, y, u, v, s2n = windef.first_pass(img.filtered, im.filtered, self.pivSetup.settings)
                if self.img_piv == 'bg':
                    x, y, u, v, s2n = windef.first_pass(img.bg, im.bg, self.pivSetup.settings)
                if self.img_piv == 'bgs':
                    x, y, u, v, s2n = windef.first_pass(img.bgs, im.bgs, self.pivSetup.settings)
                if self.img_piv == 'masked':
                    x, y, u, v, s2n = windef.first_pass(img.masked,im.masked,self.pivSetup.settings)
                # ------------------------------------------------------------------------------------------------------

                if np.isnan(u[0][0]) is True:
                    print("PIV First-Pass gives no results: (u,v) = Nan")
                    #raise KeyboardInterrupt

                # ------------------------------------------------------------------------------------------------------
                # TODO: Need to simplify masking functions for readability. It's currently very confusing to figure out.
                # Masking (and mask the velocity vector field)
                if self.pivSetup.settings.image_mask and self.piv_mask is not None:
                    if self.piv_mask == 'bpe':
                        image_mask = np.logical_and(img.bpe_mask, img_b.bpe_mask)
                        min_length = 3
                    else:
                        image_mask = np.logical_and(img.mask, img_b.mask)
                        min_length = 10
                    mask_coords = preprocess.mask_coordinates(image_mask, min_length=min_length)
                    # mark those points on the grid of PIV inside the mask
                    grid_mask = preprocess.prepare_mask_on_grid(x, y, mask_coords)
                    grid_mask = (~grid_mask.astype(bool)).astype(int)

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
                # ------------------------------------------------------------------------------------------------------

                if self.pivSetup.settings.smoothn:
                    u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(u, s=self.pivSetup.settings.smoothn_p)
                    v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(v, s=self.pivSetup.settings.smoothn_p)

                if not isinstance(u, np.ma.MaskedArray):
                    raise ValueError("Expected masked array")

                # 3.2.0 - Run multi pass windows deformation loop
                for current_iteration in range(0, self.pivSetup.settings.num_iterations):
                    # --------------------------------------------------------------------------------------------------
                    # TODO: Reduce this section into one simple line using the custom_multipass_img_deform function.
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
                    # --------------------------------------------------------------------------------------------------

                # Masking (perform masking in order to get BPE mask coordinates on refined mesh)
                if self.pivSetup.settings.image_mask and self.piv_mask is not None:
                    if self.piv_mask == 'bpe':
                        image_mask = np.logical_and(img.bpe_mask, img_b.bpe_mask)
                        min_length = 3
                    else:
                        image_mask = np.logical_and(img.mask, img_b.mask)
                        min_length = 10
                    mask_coords = preprocess.mask_coordinates(image_mask, min_length=min_length)
                    # mark those points on the grid of PIV inside the mask
                    grid_mask = preprocess.prepare_mask_on_grid(x, y, mask_coords)

                # ------------------------------------------------------------------------------------------------------
                # TODO: Figure out what the smoothn function is doing, why applied at early passes but not the last?
                # what are the advantages/disadvantages to using smoothing? Is it creating "artificial" data?
                # If the smoothing is active, we do it at each pass but not the last one
                if self.pivSetup.settings.smoothn is True and current_iteration < self.pivSetup.settings.num_iterations - 1:
                    u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(u, s=self.pivSetup.settings.smoothn_p)
                    v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(v, s=self.pivSetup.settings.smoothn_p)
                # ------------------------------------------------------------------------------------------------------

                if not isinstance(u, np.ma.MaskedArray):
                    raise ValueError('not a masked array anymore')

                # Replace Nan with zeros
                u = u.filled(0.)
                v = v.filled(0.)

                # ------------------------------------------------------------------------------------------------------
                # TODO: Make sure the scaling factor is being applied correctly.
                # 3.2.2 - Adjust scaling
                u = u / self.pivSetup.settings.dt
                v = v / self.pivSetup.settings.dt
                x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=self.pivSetup.settings.scaling_factor)
                # ------------------------------------------------------------------------------------------------------

                # ------------------------------------------------------------------------------------------------------
                # TODO: Add option to export PIV data - should it be compiled into one very large .csv or many .csv's?
                """
                1. Need to export PIV data efficiently and effectively to make data analysis easier. 
                2. Need to export important and relevant PIV settings to help figure out why the results are the way
                they are, the map out the effects of each PIV settings, and to optimize the settings for best results.

                1. Export data:
                    Pandas Dataframe:
                        Columns: frame, x, y, u, v
                        frame (int): img.frame
                        x (float): x-coordinate in PIV coordinates
                        y (float): y-coordinate in PIV coordinates
                        u (float): u velocity in PIV coordinates (microns / second)
                        v (float): v velocity in PIV coordinates (microns / second)
                        ---------- FUTURE COLUMNS -----------
                        region (str): is this vector in the BPE or non-BPE region.
                2. Export PIV settings:
                    PIV analysis details (for now but need to export entire PIVsetup b/c everything is important):
                        self.u_min = u_min # microns / second
                        self.u_max = u_max
                        self.v_min = v_min # microns / second
                        self.v_max = v_max
                        .... and many, many more later.
                """
                frame_out = img.frame * np.ones_like(x)
                frame_out = frame_out.flatten('F')
                x_out = x.flatten('F')
                y_out = y.flatten('F')
                u_out = u.flatten('F')
                v_out = v.flatten('F')
                s2n_out = s2n.flatten('F')
                data_out = np.stack(arrays=(frame_out, x_out, y_out, u_out, v_out, s2n_out), axis=1)
                piv_data_out.append(data_out)
                # ------------------------------------------------------------------------------------------------------

                # ------------------------------------------------------------------------------------------------------
                # TODO: Put PIV stats calculation into a sub-function for code readability.
                # calculate the PIV stats
                if self.pivSetup.calculate_zeta:
                    # --- if calculating zeta stats (i.e. no BPE) ---
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
                    # --- perform masking of [non] BPE regions ---
                    column_values = grid_mask[0,:]
                    bpe_columns = np.argwhere(column_values)
                    bpe_leftedge = np.min(bpe_columns)
                    bpe_rightedge = np.max(bpe_columns)
                    bpe_centerline = (bpe_rightedge - bpe_leftedge) / 2
                    if ((bpe_centerline+1) % 2) == 0:
                        print("BPE is spanned by an even number of interrogation windows. Consider +/-1 windows for better PIV-to-centerline analys")
                    else:
                        bpe_centerline = np.ceil(bpe_centerline)
                    bpe_centerline = int(bpe_centerline)

                    # store edges
                    self.bpe_leftedge = bpe_leftedge
                    self.bpe_rightedge = bpe_rightedge
                    self.bpe_centerline = bpe_centerline

                    # mask non-BPE regions
                    u_zeros_masked_array = ma.masked_where(u == 0, u)
                    u_far_left_masked_array = u_zeros_masked_array[:, :bpe_leftedge]
                    u_centerline_masked_array = u_zeros_masked_array[:, bpe_centerline-1:bpe_centerline]
                    u_far_right_masked_array = u_zeros_masked_array[:, bpe_rightedge:]

                    # mask BPE regions
                    u_pos_left_masked_array = ma.masked_less_equal(u, 0)
                    u_pos_left_masked_array2 = u_pos_left_masked_array[:,bpe_leftedge:bpe_centerline-1]
                    u_neg_right_masked_array = ma.masked_greater_equal(u, 0)
                    u_neg_right_masked_array2 = u_neg_right_masked_array[:,bpe_centerline:bpe_rightedge]

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
                # ------------------------------------------------------------------------------------------------------

                # ------------------------------------------------------------------------------------------------------
                # TODO: Simplify zeta vs. u_mean_columns implementation by writing separate implementations?
                if (self.pivSetup.save_plot or self.pivSetup.show_plot) and (self.pivSetup.calculate_zeta):
                    plot_quiver(x, y, u, v, img, self.pivSetup, img_piv_plot=self.img_piv_plot,
                                u_mag_mean = img.M_mean, u_mag_std = img.M_std,
                                locname=locname, testname=testname, runname=runname, seqname=seqname)
                elif self.pivSetup.save_plot or self.pivSetup.save_u_mean_plot or self.pivSetup.show_plot:
                    plot_quiver_and_u_mean(x, y, u_masked_plot, v, img, self.pivSetup, img_piv_plot=self.img_piv_plot,
                                           u_mean_columns=u_mean, locname=locname, testname=testname, runname=runname, seqname=seqname)
                # ------------------------------------------------------------------------------------------------------

                # empty the file to reduce RAM storage
                img.empty_file(to_empty='all')

            counter += 1

        # empty files from seq to reduce RAM storage
        seqs.remove_files(file='all')

        # --------------------------------------------------------------------------------------------------------------
        # TODO: Export all important and relevant details: PIV settings.
        piv_data_out = np.array(piv_data_out)
        piv_data_out = np.reshape(piv_data_out, newshape=(np.shape(piv_data_out)[0]*np.shape(piv_data_out)[1],
                                                          np.shape(piv_data_out)[2]), order='F')
        columns_out = ['frame', 'x', 'y', 'u', 'v', 'snr']
        df_export = pd.DataFrame(data=piv_data_out, index=None, columns=columns_out, dtype=float)
        piv_data_save_name = 'PIV_data_E{}Vmm_f{}Hz_seq{}.csv'.format(testname[0], testname[1], seqname)
        piv_data_save_path = join(self.pivSetup.save_text_path, piv_data_save_name)
        df_export.to_csv(path_or_buf=piv_data_save_path, header=columns_out, index=False)
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # TODO: Reorganize to simplify zeta vs. u_mean columns code into sub-functions and then call?
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
        # --------------------------------------------------------------------------------------------------------------


    def custom_multipass_img_deform(self, frame_a, frame_b, current_iteration,
                                    x_old, y_old, u_old, v_old, mask_coords):

        # --------------------------------------------------------------------------------------------------------------
        # TODO: Figure out what is actually happening in RectBivariateSpline interpolation and image deformation.
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
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # TODO: Write a custom implementation of PIV cross correlation to allow for rectangular interrogation regions.
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
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # TODO: understand what is going on in this section: why u += u_pre?
        shapes = np.array(get_field_shape(frame_a.shape, self.pivSetup.settings.windowsizes[current_iteration], self.pivSetup.settings.overlap[current_iteration]))
        u = u.reshape(shapes)
        v = v.reshape(shapes)
        s2n = s2n.reshape(shapes)
        u += u_pre
        v += v_pre
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # TODO: Write a custom implementation of image masking for BPE and non-BPE regions.
        # reapply the image mask to new grid
        if self.pivSetup.settings.image_mask:
            grid_mask = preprocess.prepare_mask_on_grid(x, y, mask_coords)
            grid_mask = (~grid_mask.astype(bool)).astype(int)
            u = np.ma.masked_array(u, mask=grid_mask)
            v = np.ma.masked_array(v, mask=grid_mask)

            # ----------------------------------------------------------------------------------------------------------
            # TODO: Write custom validation filters that enable specialized operations and wider control.
            # validate
            u, v, mask = validation.typical_validation(u, v, s2n, self.pivSetup.settings)
            # -----------------------------------------------------------------------------------------------------------
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
            grid_mask = (~grid_mask.astype(bool)).astype(int)
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
                "Specified metrics level {} is not one of the valid levels: {}".format(level, valid_levels))

        if level == 'file':

            if None in [loc, test, run, seq, file]:
                raise ValueError("Must specify: loc, test, run, seq, and file for file-level metrics")
            levels = [loc, test, run, seq, file]

        if level == 'seq':

            if None in [loc, test, run, seq]:
                raise ValueError("Must specify: loc, test, run, and seq for seq-level metrics")
            levels = [loc, test, run, seq]

        if level == 'run':

            if None in [loc, test, run]:
                raise ValueError("Must specify: loc, test, and run for run-level metrics")
            levels = [loc, test, run]

        if level == 'test':

            if None in [loc, test]:
                raise ValueError("Must specify: loc, and test for test-level metrics")
            levels = [loc, test]

        if level == 'loc':

            if None in [loc]:
                raise ValueError("Must specify loc for loc-level metrics")
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