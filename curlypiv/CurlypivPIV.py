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
import pandas as pd

# Image Processing
import cv2 as cv
from skimage import io


# Curlypiv
import curlypiv
from curlypiv.CurlypivPIVSetup import CurlypivPIVSetup
from curlypiv.CurlypivFile import CurlypivFile
from curlypiv.CurlypivUtils import find_substring, get_sublevel
from curlypiv.CurlypivPlotting import plot_quiver, plot_per_loc

# OpenPIV
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv"))
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv/openpiv"))
from openpiv import *
from windef import Settings
import openpiv.piv
from openpiv import windef
from openpiv.windef import Settings
from openpiv import tools, scaling, validation, filters, preprocess
from openpiv.pyprocess import extended_search_area_piv, get_field_shape, get_coordinates
from openpiv import smoothn
from openpiv.preprocess import mask_coordinates


# 2.0 define class
class CurlypivPIV(object):

    def __init__(self, testCollection, testSetup, pivSetup=None,
                 cropspecs=None, resizespecs=None, filterspecs=None, backsubspecs=None,
                 backSub_init_frames=15, num_analysis_frames=10,
                 exclude=[]):

        if not isinstance(testCollection, curlypiv.CurlypivTestCollection):
            raise ValueError("Specified test collection {} is not a CurlypivTestCollection".format(testCollection))

        if not isinstance(testSetup, curlypiv.CurlypivTestSetup):
            raise ValueError("Specified test collection {} is not a CurlypivTestSetup".format(testSetup))

        self.collection = testCollection
        self.setup = testSetup
        self.pivSetup = pivSetup

        self.cropspecs = cropspecs
        self.resizespecs = resizespecs
        self.filterspecs = filterspecs
        self.backsubspecs = backsubspecs

        self.bg_method = backsubspecs
        self.backSub = None
        self.backSub_init_frames = backSub_init_frames
        self.num_analysis_frames = num_analysis_frames

    def piv_analysis(self, level,
                     save_texts=False, save_plots=False, show_plots=False,
                     loc=None, test=None, run=None, seq=None, file=None):

        if level == 'all':
            for locs in self.collection.locs.values():
                if self.bg_method['bg_method'] == 'KNN':
                    self.backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
                for tests in locs.tests.values():
                    for runs in tests.runs.values():
                        seq0_bkg = 0
                        for seqs in runs.seqs.values():

                            # per seq sampling:
                            seqs.refresh_files()
                            self.piv(seqs, testname=tests.name, seqname=seqs.name, u_mag_bkg=seq0_bkg)
                            if seqs.name == 0:
                                seq0_bkg = seqs.u_mag_bkg
                            print('Loc: {}, Test: {}, Run: {}, Sequence: {}'.format(locs.name, tests.name, runs.name,seqs.name))
                            print('U magnitude mean = {} +/- {}'.format(seqs.u_mag_mean, seqs.u_mag_std*2))

                        # per run sampling
                        runs.add_piv_data()

                    # per test sampling
                    tests.add_piv_data()
                    print(tests)

                # per loc sampling
                plot_per_loc(locs)




    def piv(self, seqs,
            locname=None, testname=None, runname=None, seqname=None,
            u_mag_bkg=0):

        u_mag_means = []
        u_mag_stds = []
        u_means = []
        v_means = []
        u_bkgs = []


        img_b_list = list(seqs.files.values())
        counter = 0

        for filename, img in seqs.files.items():

            # break condition
            if counter > self.backSub_init_frames + self.num_analysis_frames:
                continue

            # ----- Image Pre-Processing -----


            if counter <= self.backSub_init_frames:

                # crop
                if self.cropspecs:
                    img.image_crop(cropspecs=self.cropspecs)

                # resize
                if self.resizespecs:
                    img.image_resize(method='pyramid_expand', scale=self.resizespecs)

                # filter
                if self.filterspecs:
                    img.image_filter(filterspecs=self.filterspecs, image_input='raw', image_output='filtered', force_rawdtype=True)

                # subtract background
                if self.backsubspecs:
                    img.image_subtract_background(image_input='filtered', backgroundSubtractor=self.backSub)

            if counter > self.backSub_init_frames:
                img_b = img_b_list[counter + 1]

                for im in [img, img_b]:
                    # crop
                    if self.cropspecs:
                        im.image_crop(cropspecs=self.cropspecs)

                    # resize
                    if self.resizespecs:
                        im.image_resize(method='pyramid_expand', scale=self.resizespecs)

                    # filter
                    if self.filterspecs:
                        im.image_filter(filterspecs=self.filterspecs, image_input='raw', image_output='filtered',
                                         force_rawdtype=True)

                    # subtract background
                    if self.backsubspecs:
                        im.image_subtract_background(image_input='filtered', backgroundSubtractor=self.backSub)

                    # plotter
                    fig, ax = plt.subplots()
                    ax.imshow(img.img_bgs)
                    plt.show()
                    print('ha')

                # 3.1.4 - Start First Pass PIV
                x, y, u, v, s2n = windef.first_pass(img.masked,img_b.masked,self.pivSetup.settings)

                if np.isnan(u[0][0]) is True:
                    print("PIV First-Pass gives no results: (u,v) = Nan")
                    raise KeyboardInterrupt

                mask_coords = []
                u = np.ma.masked_array(u, mask=np.ma.nomask)
                v = np.ma.masked_array(v, mask=np.ma.nomask)


                # 3.2.0 - Run multi pass windows deformation loop
                for current_iteration in range(0, self.pivSetup.settings.num_iterations):
                    x, y, u, v, s2n, mask = windef.multipass_img_deform(
                        img.masked,
                        img_b.masked,
                        current_iteration,
                        x,
                        y,
                        u,
                        v,
                        self.pivSetup.settings,
                        mask_coords=mask_coords
                    )

                # If the smoothing is active, we do it at each pass
                # but not the last one
                if self.pivSetup.settings.smoothn is True and current_iteration < self.pivSetup.settings.num_iterations - 1:
                    u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(
                        u, s=self.pivSetup.settings.smoothn_p
                    )
                    v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(
                        v, s=self.pivSetup.settings.smoothn_p
                    )

                # 3.2.2 - Adjust scaling
                u = u / self.pivSetup.settings.dt
                v = v / self.pivSetup.settings.dt
                x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=self.pivSetup.settings.scaling_factor)

                if np.isnan(u[0][0]) == True:
                    print("PIV Multi-Pass gives no results: (u,v) = Nan")
                    raise KeyboardInterrupt

                # calculate the PIV stats
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

                if self.pivSetup.save_plot or self.pivSetup.show_plot:
                    plot_quiver(x, y, u, v, img, self.pivSetup,
                                u_mag_mean = img.M_mean, u_mag_std = img.M_std,
                                locname=locname, testname=testname, runname=runname, seqname=seqname)

                # empty the file to reduce RAM storage
                img.empty_file(to_empty='all')

            counter += 1

        # empty files from seq to reduce RAM storage
        seqs.remove_files(file='all')

        seq_u_mag_mean = np.round(np.mean(u_mag_means),1)
        seq_u_mag_std = np.round(np.mean(u_mag_stds),2)
        seq_u_means = np.round(np.mean(u_means),1)
        seq_v_means = np.round(np.mean(v_means),1)
        if len(u_bkgs) > 1:
            seq_u_bkgs = np.round(np.mean(u_bkgs),1)
        else:
            seq_u_bkgs = None


        seqs.add_piv_data(u_mag_mean=seq_u_mag_mean, u_mag_std=seq_u_mag_std, u_mean=seq_u_means,
                          v_mean=seq_v_means, u_mag_bkg=seq_u_bkgs)



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
