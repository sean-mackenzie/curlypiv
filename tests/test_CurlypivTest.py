"""
Notes about the program
"""

# import modules
# data i/o
from os import listdir
from os.path import join

from curlypiv.CurlypivTestCollection import CurlypivRun, CurlypivTest

# test zeta potential measurement via micro particle image velocimetry metrics

# step 1 - load data files, etc
# --- notes ---

# load image
base_path = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test'
tests_folder = 'tests'
loc = 'loc1'
test='E-1Vmm'
dirread = join(base_path,tests_folder,loc,test)

img_type = '.tif'
testid = ('E','Vmm')
runid = ('run', 'num')
seqid = ('test_', '_X')
frameid = '_X'

# instantiate CurlypivRun class
test = CurlypivTest(dirpath=dirread, file_type=img_type, testid=testid, runid=runid, seqid=seqid, frameid=frameid)

print(test)

print('yes sir!')