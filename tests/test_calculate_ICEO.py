"""
Notes about the program
"""


# ---------- ----------  STEP 1: LOAD TEST COLLECTION ----------  ---------- ---------- ---------- ---------- ----------

# import modules
from curlypiv.CurlypivTestCollection import CurlypivRun, CurlypivTest, CurlypivTestCollection

# load test files
name = 'testCol'
base_path = '/Users/mackenzie/Desktop/04.23.21-iceo-test'
img_type = '.tif'
testid = ('E','Vmm', 'f', 'Hz')
runid = ('run', 'num')
seqid = ('test_', '_X')
frameid = '_X'

# load calibration files
gridPath = None
illumPath = None
camNoisePath = None

# declare filter settings
cropspecs = None
filterspecs = None
resizespecs = None
backsubspecs = None

# instantiate CurlypivTestCollection class
testCol = CurlypivTestCollection(name, base_path, file_type=img_type, testid=testid, runid=runid, seqid=seqid,
                                 frameid=frameid, load_files=True)

