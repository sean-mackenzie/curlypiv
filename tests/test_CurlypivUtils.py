# test CurlypivUtils
"""
Notes about program
"""

# 1.0 import modules
from curlypiv.utils import CurlypivUtils
from curlypiv.utils.CurlypivUtils import find_testcollection, size_files




dirsearch = '/Users/mackenzie/Desktop/03.03.21 - zeta testing pdms/tests/10s'
filetype = '.tif'
testid=('E','Vmm')
runid=('run','num')
seqid=('test_','_X')
frameid='_X'

filelist = find_testcollection(dirsearch, filetype,testid=testid, runid=runid, seqid=seqid, frameid=frameid)

sizes = size_files(filelist)



