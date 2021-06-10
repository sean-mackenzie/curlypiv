# test CurlypivImage
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import sys
import os

# scientific
import numpy as np

# Image processing
import cv2 as cv
# skimage
from skimage import io, img_as_float
from skimage.morphology import disk, white_tophat
from skimage.filters import median, gaussian
from skimage.restoration import denoise_wavelet
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, pyramid_expand
from skimage.feature import blob_log

# plotting
import matplotlib.pyplot as plt

# OpenPIV
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv"))
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv/openpiv"))
from openpiv import *
from windef import Settings
testset = Settings()

# Curlypiv
from curlypiv.CurlypivTestCollection import CurlypivTestCollection
from curlypiv.CurlypivTestSetup import CurlypivTestSetup
from curlypiv.CurlypivPIV import CurlypivPIV
from curlypiv.CurlypivFile import CurlypivFile
from curlypiv.CurlypivImageProcessing import img_resize, img_subtract_background, img_filter


# ------------------------- test CurlypivPIV below ------------------------------------

# inputs
name = 'testCol'
base_path = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test'
test_dir = 'tests'
test_level = 'seq' # ['all','loc','test','run','seq','file']
img_type = '.tif'
loc = 1
test = 2.5
testid = ('E','Vmm')
run = 3
runid = ('run', 'num')
seq = 1
seqid = ('test_', '_X')
frameid = '_X'

# processing inputs
bg_method = 'KNN'

if bg_method == 'KNN':
    backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
elif bg_method == "MOG2":
    backSub = cv.createBackgroundSubtractorMOG2(detectShadows=False)
elif bg_method == 'CMG':
    backSub = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames=None)
elif bg_method == 'GSOC':
    backSub = cv.bgsegm.createBackgroundSubtractorGSOC(blinkingSupressionMultiplier=0.1)

# instantiate PIV class object.
piv = CurlypivPIV(
    testCollection = CurlypivTestCollection(name, base_path, file_type=img_type, dir_tests=test_dir,
                              testid=testid, runid=runid, seqid=seqid, frameid=frameid),
    testSetup=CurlypivTestSetup(name='test')
)

# get appropriate metrics level
imgs = piv.get_analysis_level(level=test_level,loc=loc, test=test, run=run, seq=seq)


# ----- TEST IMAGE PROCESSING METHODS ON SEVERAL CURLYPIV.IMAGES IN A SEQUENCE -----

img_baseline = imgs.get_sublevel(key='test_1_X1.tif')

img1 = imgs.get_sublevel_all()

particles_per_image = []
stats_per_image = []
bg_hist = []
bg_ratio = []
bg_threshold = []
images_analyzed = []

save = False
counter=0

for img in img1:

    images_analyzed.append(img.name)

    # crop
    cropping = {
        'xmin': 80,     # x = 0 is the left of the image
        'xmax': 420,
        'ymin': 250,
        'ymax': 450     # y = 0 is the bottom of the image
    }
    #img.image_crop(cropspecs=cropping)

    # resize
    img.image_resize(method='pyramid_expand', scale=2)

    # filter
    processing = {
        'denoise_wavelet': {'args':[], 'kwargs': dict(method='BayesShrink', mode='soft', rescale_sigma=True)},
        'rescale_intensity': {'args': [(4,99.995),('dtype')]}
        }
    img.image_filter(filterspecs=processing, image_input='raw', image_output='filtered', force_rawdtype=True)

    # subtract background
    img.image_subtract_background(image_input='filtered', backgroundSubtractor=backSub)

    # save image
    if save == True:
        imagePath = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test/tests/loc1/E2.5Vmm/run2num'
        savePath = imagePath + '/writeCurlyFromImages/' + str(img.name)
        cv.imwrite(savePath, img.bg)

    plot = True
    if plot and counter > 10:
        src = [img.raw, img.filtered, img.bg, img.bgs]
        if (src[0].dtype == src[1].dtype) and (src[2].dtype == src[3].dtype):
            #subplots2 = cv.hconcat([src[2], src[3]])
            #subplots2 = cv.hconcat([img.mask, img.mask])
            img_mask = np.asarray(img.mask * 255, dtype='uint8')
            cv.imshow("background subtracted image", img_mask)
    else:
        #img_conv = img.raw*img.masked
        #img_conv = rescale_intensity(img_conv, in_range='image', out_range='float')
        subplots2 = cv.hconcat([img.raw, img.masked])
        cv.imshow('mask', subplots2)

    keyboard = cv.waitKey(15)
    if keyboard == 27:
        break

    counter += 1