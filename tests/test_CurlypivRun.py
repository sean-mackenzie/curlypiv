"""
Notes about the program
"""

# import modules
# data i/o
from os import listdir
from os.path import join

from curlypiv import CurlypivImageCollection
from curlypiv.CurlypivImageCollection import CurlypivRun
from curlypiv.CurlypivImage import CurlypivImage

# test zeta potential measurement via micro particle image velocimetry analysis

# step 1 - load data files, etc
# --- notes ---

# load image
base_path = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test'
tests_folder = 'tests'
loc = 'loc1'
test='E-1Vmm'
run='run1num'
dirread = join(base_path,tests_folder,loc,test,run)

img_type = '.tif'
runid = ('run', 'num')
seqid = ('test_', '_X')
frameid = '_X'

# instantiate CurlypivRun class
run = CurlypivRun(dirpath=dirread, file_type=img_type, runid=runid, seqid=seqid, frameid=frameid)

print(run)

print('yes sir!')