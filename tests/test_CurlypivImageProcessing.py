"""
Notes about the program
"""

# import modules
# data i/o
from os import listdir
from os.path import join

import curlypiv
from curlypiv.CurlypivImageProcessing import img_resize, img_subtract_background
from curlypiv.CurlypivTestCollection import CurlypivRun, CurlypivTest, CurlypivTestCollection
from curlypiv.CurlypivTestSetup import CurlypivTestSetup
from curlypiv.CurlypivPIV import CurlypivPIV

# test image processing functions

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

# ----- TEST IMAGE PROCESSING METHODS -----

# resize
img_resized = img_resize(img, method='rescale', scale=2)