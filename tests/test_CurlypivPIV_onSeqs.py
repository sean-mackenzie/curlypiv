# test CurlypivImage
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import sys
import os

# scientific
import numpy as np

# Image processing
import cv2 as cv
# skimage
from skimage import io
from skimage.morphology import disk, white_tophat
from skimage.filters import median, gaussian
from skimage.restoration import denoise_wavelet
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, pyramid_expand
import skimage.transform as skt
from skimage.feature import blob_log

# plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import matplotlib.image as mgimg
from matplotlib import animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
matplotlib.rcParams['figure.figsize'] = (10, 9)
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=16, weight='bold')
font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']

# OpenPIV
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv"))
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv/openpiv"))
from openpiv import *
from windef import Settings
import openpiv.piv
from openpiv import windef
from openpiv.windef import Settings
from openpiv import tools, scaling, validation, filters, preprocess
from openpiv.pyprocess import extended_search_area_piv, get_field_shape, get_coordinates
from openpiv import smoothn
from openpiv.preprocess import mask_coordinates
testset = Settings()

# Curlypiv
from curlypiv.CurlypivTestCollection import CurlypivTestCollection
from curlypiv.CurlypivTestSetup import CurlypivTestSetup
from curlypiv.CurlypivPIV import CurlypivPIV
from curlypiv.CurlypivPIVSetup import CurlypivPIVSetup
from curlypiv.CurlypivFile import CurlypivFile
from curlypiv.CurlypivImageProcessing import resize, subtract_background, filter

# rarely changed inputs
nameTestCol = 'testCol'
nameTestSetup = 'testSet'
namePIVSetup = 'testPIV'
base_path = '/Users/mackenzie/Desktop/03.04.21-ZuPIVelastosil'
test_dir = 'tests'
img_type = '.tif'
testid = ('E','Vmm')
runid = ('run', 'num')
seqid = ('test_', '_X')
frameid = '_X'
bg_method = 'KNN'
backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
processing = {'denoise_wavelet': {'args': [], 'kwargs': dict(method='BayesShrink', mode='soft', rescale_sigma=True)}, 'rescale_intensity': {'args': [(4, 99.995), ('dtype')]}}


cropspecs = {
    'xmin': 100,  # x = 0 is the left of the image
    'xmax': 356,
    'ymin': 300,
    'ymax': 428  # y = 0 is the bottom of the image
}

filterspecs = {
    'denoise_wavelet': {'args': [], 'kwargs': dict(method='BayesShrink', mode='soft', rescale_sigma=True)},
    'rescale_intensity': {'args': [(4, 99.995), ('dtype')]}
}

resizespecs = {
        'scale': 2
}

backsubspecs = {
        'bg_method': 'KNN'
}

# ------------------------- test CurlypivPIV below ------------------------------------

# inputs you will change
scale = 2
test_level = 'all' # ['all','loc','test','run','seq','file']
loc = None #1
test = None #2.5
run = None #2
seq = None #1


# create instances
testCol = CurlypivTestCollection(nameTestCol, base_path, file_type=img_type, dir_tests=test_dir, testid=testid, runid=runid, seqid=seqid, frameid=frameid)
testSet = CurlypivTestSetup(name=nameTestSetup)
pivSet = CurlypivPIVSetup(name=namePIVSetup, save_text=False, save_plot=False, show_plot=False, vectors_on_image=True,testCollection=testCol,testSetup=testSet)



piv = CurlypivPIV(testCollection=testCol, testSetup=testSet, pivSetup=pivSet,
                  cropspecs=cropspecs, filterspecs=filterspecs, resizespecs=resizespecs, backsubspecs=backsubspecs,
                  num_analysis_frames=2)


piv.piv_analysis(level='all')












# ----- TEST IMAGE PROCESSING METHODS ON SEVERAL CURLYPIV.IMAGES IN A SEQUENCE -----
#img_baseline = imgs.get_sublevel(key='test_1_X1.tif')
#img1 = imgs.get_sublevel_all()


