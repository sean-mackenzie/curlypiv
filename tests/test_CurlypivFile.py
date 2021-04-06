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


# ------------------------- test CurlypivPIV below ------------------------------------

# inputs
name = 'testCol'
base_path = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test'
test_dir = 'tests'
test_level = 'file' # ['all','loc','test','run','seq','file']
img_type = '.tif'
testid = ('E','Vmm')
runid = ('run', 'num')
seqid = ('test_', '_X')
frameid = '_X'

# instantiate PIV class object.
piv = CurlypivPIV(
    testCollection = CurlypivTestCollection(name, base_path, file_type=img_type, dir_tests=test_dir,
                              testid=testid, runid=runid, seqid=seqid, frameid=frameid),
    testSetup=CurlypivTestSetup(name='test')
)

# get appropriate analysis level
img = piv.get_analysis_level(level=test_level,loc=1, test=2.5, run=1, seq=1, file='test_1_X1.tif')


# ----- TEST IMAGE PROCESSING METHODS ON A SINGLE CURLYPIV.IMAGE-----

# resize
img.image_resize(method='rescale', scale=2)
print("Original image shape: {}".format(img.original.shape))
print("Resized image shape: {}".format(img.raw.shape))

# crop
print("Pre-cropping image shape: {}".format(img.raw.shape))
cropping = {
    'xmin': 50,
    'xmax': 250,
    'ymin': 50,
    'ymax': 250
}
img.image_crop(cropspecs=cropping)
print("Cropped image shape: {}".format(img.raw.shape))

# subtract background
img.image_subtract_background(bg_method="MOG2")

# filter
processing = {
    'median': {'args': [disk(3)]},
    'white_tophat': {'args': [disk(5)]}, # returns bright spots smaller than the structuring element.
    'rescale_intensity': {'args': [(0,8e3),('dtype')]}
    }
img.image_filter(filterspecs=processing, image_input='bgs', image_output='filtered', force_rawdtype=True)

# identify particles
find_particles_in = 'filtered'
particles = img.image_find_particles(image_input=find_particles_in, min_sigma=0.5, max_sigma=3,num_sigma=20, threshold=0.1,overlap=0.35)
print("Found {} particles in image {}".format(len(particles),find_particles_in))
img.calculate_stats()
print("Image statistics")
print(img.stats)


