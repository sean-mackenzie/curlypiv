# test CurlypivImage
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import sys
import os

# Image processing
from skimage import io
from skimage.morphology import disk, white_tophat
from skimage.filters import median
from skimage.exposure import rescale_intensity

# plotting
import matplotlib.pyplot as plt

# OpenPIV
sys.path.append(os.path.abspath("/Users/mackenzie/openpiv-python"))
sys.path.append(os.path.abspath("/Users/mackenzie/openpiv-python/openpiv"))
from openpiv import *
from windef import Settings
testset = Settings()

# Curlypiv
from curlypiv.utils import CurlypivUtils
from curlypiv.utils.CurlypivPlotting import draw_particles
from curlypiv.utils import CurlypivImage


# ------------------------- test CurlypivUtils below ------------------------------------

dirsearch = '/Users/mackenzie/Desktop/02.13.21 - zeta testing sio2 - chip2 - PASS/tests/zeta'
filetype = '.tif'
testid=('E','Vmm')
runid=('run','num')
seqid=('test_','_X')
frameid='_X'

filelist = CurlypivUtils.find_testcollection(dirsearch, filetype, testid=testid, runid=runid, seqid=seqid, frameid=frameid)
print(filelist[0])

# ------------------------- test CurlypivImage below ------------------------------------

baseline_path = (filelist[0][20])
filetype = '.tif'

cropping = {
    'xmin': 50,
    'xmax': 250,
    'ymin': 50,
    'ymax': 250
}

processing = {
    'median': {'args': [disk(3)]},
    'white_tophat': {'args': [disk(2.5)]}, # returns bright spots smaller than the structuring element.
    'rescale_intensity': {'args': [(0,8e3),('dtype')]}
    }

# create instance of CurlypivImage class
baseline_img = CurlypivImage(baseline_path, '.tif')
print(baseline_img) # print dimensions

# crop the image
baseline_img.crop_image(cropspecs=None)
print(baseline_img) # check that dimensions have been cropped

# subtract the background image
bgfilepath= '/Users/mackenzie/Desktop/02.13.21 - zeta testing sio2 - chip2 - PASS/background/zeta/bgs/bg_E2.5_min.tif'
baseline_img.subtract_background(bg_method='min', bg_filepath=bgfilepath)

# filter the image
baseline_img.filter_image(filterspecs=processing)

# identify particles
particles = baseline_img.find_particles(min_sigma=0.5, max_sigma=3,num_sigma=20, threshold=0.1,overlap=0.35)

print(len(particles))
baseline_img.calculate_stats()
print(baseline_img.stats)

# draw particles for inspection
#draw_particles(baseline_img.filtered, particles, color='red',title='Laplace of Gaussians',figsize=(10,10))



