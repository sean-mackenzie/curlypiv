# test CurlypivImageProcessing
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import sys
import os

# scientific
import numpy.lib.stride_tricks
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
from numpy import ma
from numpy import log

# image processing
from skimage import io
from skimage.morphology import disk, white_tophat
from skimage.filters import median
from skimage.exposure import rescale_intensity
import scipy.ndimage as scn
from scipy.signal import convolve2d
from skimage.feature import blob_log
from skimage.measure import points_in_poly
from scipy.interpolate.fitpack2 import RectBivariateSpline

# other packages
sys.path.append(os.path.abspath("/Users/mackenzie/PycharmProjects/openpiv/openpiv"))
#from windef import *



# 2.0 functions

def calculate_background(img_list, bg_method='min'):
    """
    This calculates the background image for an entire test collection (CurlypivTestCollection).
    Inputs: CurlypivTestCollection (i.e. one or more CurlypivImageCollections).
    Method: Flatten all the CurlypivImageCollections and calculate background mean, min, and std.
    Output: A background image specific to the input CurlypivTestCollection.
    :param img_list:
    :param bg_method:
    :return:
    """


def subtract_background(img, bg_filepath, bg_method='min'):
    """
    This subtracts a background input image from the signal input image.
    :param bg_method:
    :param bg_img:
    :return:
    """

    img_bgs = img.copy()
    img_bg = img.copy()

    valid_bs_methods = ['min', 'mean', 'min+std', 'ml']

    if bg_method not in valid_bs_methods:
        raise ValueError(
            "{} is not a valid method. Implemented so far are {}".format(bg_method, valid_bs_methods[0:2]))
        img_bgs = None
        img_bg = None
    elif bg_method == 'ml':
        raise ValueError(
            "{} is still under development. Implemented so far are {}".format(bg_method, valid_bs_methods[0:2]))
        img_bgs = None
        img_bg = None
    else:
        img_bg = io.imread(bg_filepath)
        img_bgs = img - img_bg

    return(img_bg, img_bgs)


def filter_image(img, filterspecs):
    """
    This is an image filtering function. The argument filterdict are similar to the arguments of the...
    e.g. filterdict: {'median': 5, 'gaussian':3}
    :param filterspecs:
    :param force_rawdtype:
    :return:

    """

    valid_filters = ['none','median','gaussian','white_tophat','rescale_intensity']

    filtering_sequence = 1

    for process_func in filterspecs.keys():
        if process_func not in valid_filters:
            img_filtered = None
            raise ValueError("{} is not a valid filter. Implemented so far are {}".format(process_func, valid_filters))
        if process_func == "none":
            img_filtered = None
        else:
            func = eval(process_func)
            args = filterspecs[process_func]['args']
            if 'kwargs' in filterspecs[process_func].keys():
                kwargs = filterspecs[process_func]['kwargs']
            else:
                kwargs = {}

            img_filtered = apply_filter(img, func, *args, **kwargs)
            print("Filter #{}: {}".format(filtering_sequence, func))
            filtering_sequence += 1

    return(img_filtered)


def find_particles(img, min_sigma=0.5, max_sigma=5, num_sigma=20, threshold=0.1, overlap=0.85):
    """
    This uses Laplacian of Gaussians method to determine the number and size of particles in the image.
    :return:
    """

    # particles
    particles = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                         threshold=threshold, overlap=overlap)
    particles[:, 2] = particles[:, 2] * np.sqrt(2)  # compute radius in the 3rd column

    return (particles)



def apply_filter(img, func, *args, **kwargs):
    assert callable(func)
    return func(img, *args, **kwargs)

