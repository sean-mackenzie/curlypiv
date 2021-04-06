# test CurlypivSyntheticCollection
"""
Notes about program
"""

# 1.0 import modules
# data I/O
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
import pandas as pd

# Image Processing
from skimage import io

# Curlypiv
import curlypiv.CurlypivTestCollection
from curlypiv.CurlypivTestCollection import CurlypivSequence
from curlypiv.CurlypivFile import CurlypivFile
from curlypiv.CurlypivUtils import find_substring


# 2.0 define class
class CurlypivSyntheticCollection(object):

    def __init__(self, testSetup, imgSamplingPath=None, imgIlluminationXYPath=None, num_images=50, img_type='.tif'):

        assert isinstance(testSetup, object)

        self.setup = testSetup
        self.imgSamplePath = imgSamplingPath
        self.imgIlluminationXYPath = imgIlluminationXYPath
        self.img_type = img_type
        self.num_images = num_images
        self.num_images_iXY = num_images

        self.sampleImage(p_bkg=90, p_sig=90)
        self.initializeMicroSigSetup(background_mean=self.bkg_mean, background_noise=self.bkg_noise)

    def initializeMicroSigSetup(self, focal_length=3e-3,
                                background_mean=0, background_noise=0, points_per_pixel=15,
                                gain=3, cyl_focal_length=10, ri_lens=1.5):

        self.microsigSetup = dict(
            magnification=self.setup.optics.microscope.objective.magnification,
            numerical_aperture=self.setup.optics.microscope.objective.numerical_aperture,
            focal_length=focal_length,
            ri_medium=self.setup.optics.microscope.objective._n0,
            ri_lens=ri_lens,
            pixel_size=self.setup.optics.microscope.ccd.pixel_size,
            pixel_dim_x=self.setup.optics.illumination.illumination_distribution.shape[1],
            pixel_dim_y=self.setup.optics.illumination.illumination_distribution.shape[0],
            pixel_dim_z=self.setup.chip.channel.height,
            background_mean=background_mean,
            background_noise=background_noise,
            points_per_pixel=points_per_pixel,
            n_rays=500,
            gain=gain,
            cyl_focal_length=cyl_focal_length,
        )

    def sampleImage(self, p_bkg=75, p_sig=95):

        listSamplePath = glob.glob(self.imgSamplePath+'/*'+self.img_type)

        if len(listSamplePath) > self.num_images:
            listSamplePath = listSamplePath[0:self.num_images]
        else:
            self.num_images = len(listSamplePath)

        sampleSequence = CurlypivSequence(dirpath=self.imgSamplePath, file_type=self.img_type, seqname='sample',
                                          filelist=listSamplePath, load_files=True, frameid='_X', exclude=[])

        # image stats
        bkg_mean = []       # mean of lower 75% of pixels.
        bkg_noise = []      # std of lower 75% of pixels.
        snr_mean = []            # mean of upper 5% of pixels /  bkg_mean

        for filename, img in sampleSequence.files.items():

            # calculate background percentiles
            int_bkg, int_sig = np.percentile(img.raw, [p_bkg, p_sig])

            # calculate masks
            mask_bkg = img.raw > int_bkg  # note: using a Numpy masked array requires "False" for "valid" entries
            mask_sig = img.raw < int_sig  # note: using a Numpy masked array requires "False" for "valid" entries

            # create masked array
            img_bkg = ma.masked_array(img.raw, mask=mask_bkg)
            img_sig = ma.masked_array(img.raw, mask=mask_sig)

            # calculate stats
            mean = img_bkg.mean()   # note: Numpy masked arrays exclude masked entries for mean() and std()
            std = img_bkg.std()
            sig = img_sig.mean()
            snr = sig/std

            # append to list
            bkg_mean.append(mean)
            bkg_noise.append(std)
            snr_mean.append(snr)

        # calculate collection stats
        self.bkg_mean = np.mean(bkg_mean).astype('uint16')
        self.bkg_noise = np.mean(bkg_noise).astype('uint16')
        self.snr_mean = np.round(np.mean(snr_mean),2)

    def illuminationDistributionXY(self, imgIlluminationXYPath=None):

        if self.imgIlluminationXYPath is None and imgIlluminationXYPath is None:
            raise ValueError("Need to enter path for illumination images")
        elif imgIlluminationXYPath:
            self.imgIlluminationXYPath = imgIlluminationXYPath

        listIllumXYPath = glob.glob(self.imgIlluminationXYPath + '/*' + self.img_type)

        if len(listIllumXYPath) > self.num_images_iXY:
            listIllumXYPath = listIllumXYPath[0:self.num_images_iXY]
        else:
            self.num_images_iXY = len(listIllumXYPath)

        imgiXYSequence = CurlypivSequence(dirpath=self.imgIlluminationXYPath, file_type=self.img_type, seqname='iXY',
                                          filelist=listIllumXYPath, load_files=True, frameid='_X')






