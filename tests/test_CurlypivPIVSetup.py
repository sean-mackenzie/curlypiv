# test CurlypivImage
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import sys
import os

# Image processing
import cv2 as cv
from skimage import io
from skimage.morphology import disk, white_tophat
from skimage.filters import median
from skimage.exposure import rescale_intensity

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
from curlypiv.CurlypivImageProcessing import resize, subtract_background, filter
from curlypiv.CurlypivPIVSetup import CurlypivPIVSetup



# inputs
nameTestCol = 'testCol'
nameTestSetup = 'testSet'
namePIVSetup = 'testPIV'
base_path = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test'
test_dir = 'tests'
test_level = 'seq' # ['all','loc','test','run','seq','file']
img_type = '.tif'
loc = 1
test = 2.5
testid = ('E','Vmm')
run = 1
runid = ('run', 'num')
seq = 1
seqid = ('test_', '_X')
frameid = '_X'

# processing inputs
bg_method = 'KNN'

if bg_method == 'KNN':
    backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)

testCol = CurlypivTestCollection(nameTestCol, base_path, file_type=img_type, dir_tests=test_dir,
                                                              testid=testid, runid=runid, seqid=seqid, frameid=frameid)
testSet = CurlypivTestSetup(name=nameTestSetup)

cpiv = CurlypivPIVSetup(name=namePIVSetup, save_text=False, save_plot=True, vectors_on_image=True,
                        testCollection=testCol,
                        testSetup=testSet
                        )


print(cpiv)


