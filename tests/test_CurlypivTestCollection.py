"""
Notes about the program
"""

# import modules
# data i/o
from os import listdir
from os.path import join

from curlypiv.CurlypivTestCollection import CurlypivRun, CurlypivTest, CurlypivTestCollection

# test zeta potential measurement via micro particle image velocimetry analysis

# step 1 - load data files, etc
# --- notes ---

# load image
name = 'testCol'
base_path = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test'
img_type = '.tif'
testid = ('E','Vmm')
runid = ('run', 'num')
seqid = ('test_', '_X')
frameid = '_X'

# instantiate CurlypivRun class
test = CurlypivTestCollection(name, base_path, file_type=img_type, testid=testid, runid=runid, seqid=seqid, frameid=frameid)

print(test)

print('yes sir!')