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
import cv2 as cv
# skimage
from skimage import io
from skimage.morphology import disk, white_tophat
from skimage.filters import median
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, pyramid_expand
from skimage.feature import blob_log
# scipy
import scipy.ndimage as scn
from scipy.signal import convolve2d
from scipy.interpolate.fitpack2 import RectBivariateSpline

# other packages
sys.path.append(os.path.abspath("/Users/mackenzie/PycharmProjects/openpiv/openpiv"))
#from windef import *



# 2.0 functions

def resize(img, method='rescale', scale=2):
    """
    This resizes and interpolates an image. Potentially useful when forced to reduce bit-depth from 16 to 8 for cv2.
    :param im: 16-bit numpy array
    :param method: rescale method
    :return: img_as_float
    """
    valid_methods = ['rescale', 'pyramid_expand']

    if method not in valid_methods:
        raise ValueError(
            "{} is not a valid method. Implemented so far are {}".format(method, valid_methods))
    elif method == "rescale":
        img = rescale(img, scale, order=1, mode='reflect', cval=0, clip=True, preserve_range=False,
                      multichannel=False, anti_aliasing=None, anti_aliasing_sigma=None)
    elif method == 'pyramid_expand':
        img = pyramid_expand(img, upscale=scale, sigma=None, order=1, mode='reflect', cval=0, multichannel=False,
                       preserve_range=False)
    return(img)


def subtract_background(img, bg_filepath, bg_method='min'):
    """
    This subtracts a background input image from the signal input image.
    :param bg_method:
    :param bg_img:
    :return:
    """


    img_bgs = img.copy()
    img_bg = img.copy()

    valid_bs_methods = ['KNN', 'MOG2', 'min', 'mean']

    if bg_method not in valid_bs_methods:
        raise ValueError(
            "{} is not a valid method. Implemented so far are {}".format(bg_method, valid_bs_methods[0:2]))
        img_bgs = None
        img_bg = None
    elif bg_method == 'KNN':
        backSub = cv.createBackgroundSubtractorKNN()
        mask = backSub.apply(img)
    elif bg_method == "MOG2":
        backSub = cv.createBackgroundSubtractorMOG2()
        mask = backSub.apply(img)
    else:
        raise ValueError(
            "{} is still under development. Implemented so far are {}".format(bg_method, valid_bs_methods[0:2]))
        img_bgs = None
        img_bg = None

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

