"""
Notes about the program
"""

# import modules
# data i/o
from os import listdir
from os.path import join

import curlypiv
from curlypiv.CurlypivTestCollection import CurlypivRun, CurlypivTest, CurlypivTestCollection

# test zeta potential measurement via micro particle image velocimetry metrics

# step 1 - load data files, etc
# --- notes ---

# load image
name = 'testCol'
base_path = '/Users/mackenzie/Desktop/04.23.21-iceo-test'
img_type = '.tif'
testid = ('V','channel', 'f', 'Hz')
runid = ('run', 'num')
seqid = ('test_', '_X')
frameid = '_X'

# instantiate CurlypivRun class
test = CurlypivTestCollection(name, base_path, file_type=img_type, testid=testid, runid=runid, seqid=seqid, frameid=frameid)

print(test)

loc = test.locs

print(loc)

print('yes sir!')