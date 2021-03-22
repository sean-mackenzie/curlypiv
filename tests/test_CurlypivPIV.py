"""
Notes about the program
"""

# import modules
# data i/o
from os import listdir
from os.path import join

import curlypiv
from curlypiv.CurlypivTestCollection import CurlypivRun, CurlypivTest, CurlypivTestCollection
from curlypiv.CurlypivTestSetup import CurlypivTestSetup
from curlypiv.CurlypivPIV import CurlypivPIV

# test zeta potential measurement via micro particle image velocimetry analysis

# step 1 - load data files, etc
# --- notes ---

print('hello')

# load image
name = 'testCol'
test_dir = 'tests'
base_path = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test'
img_type = '.tif'
testid = ('E','Vmm')
runid = ('run', 'num')
seqid = ('test_', '_X')
frameid = '_X'

# instantiate CurlypivRun class
test = CurlypivTestCollection(name, base_path, file_type=img_type, dir_tests=test_dir,
                              testid=testid, runid=runid, seqid=seqid, frameid=frameid)
setup = CurlypivTestSetup(name='test')

print(type(setup))

piv = CurlypivPIV(test, setup)

test_level = 'all' # ['all','loc','test','run','seq','file']

piv_file = piv.get_analysis_level(level=test_level,loc=1, test=2.5, run=1, seq=1, file='test_1_X1.tif')

print(piv_file)
