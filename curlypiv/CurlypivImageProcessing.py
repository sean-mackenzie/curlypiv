# test CurlypivImageProcessing
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import sys
import os
import copy

# scientific
import numpy.lib.stride_tricks
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
from numpy import ma
from numpy import log

# image processing
import cv2 as cv
# skimage
from skimage import img_as_float
from skimage import io
from skimage.morphology import disk, white_tophat
from skimage.filters import median, gaussian, threshold_local
from skimage.restoration import denoise_wavelet
from skimage.exposure import rescale_intensity, equalize_adapthist, equalize_hist
from skimage.transform import rescale, pyramid_expand
from skimage.feature import blob_log
# scipy
import scipy.ndimage as scn
from scipy.signal import convolve2d
from scipy.interpolate.fitpack2 import RectBivariateSpline

import matplotlib.pyplot as plt

# other packages
sys.path.append(os.path.abspath("/Users/mackenzie/PycharmProjects/openpiv/openpiv"))
#from windef import *



# 2.0 functions

def img_correct_flatfield(img, img_flatfield, img_darkfield):
    m = np.mean(img_flatfield - img_darkfield)
    img_corrected = (img - img_darkfield) * m / (img_flatfield - img_darkfield)
    img_corrected = rescale_intensity(img_corrected, in_range='image', out_range='uint16')

    return img_corrected

def img_apply_bpe_filter(img, bpespecs):

    valid_specs = ['bxmin', 'bxmax', 'bymin', 'bymax', 'multiplier']
    # example = [220, 280, 25, 450, 2]

    if bpespecs is None:
        img_bpe_mask = None
    else:
        bymin = img.shape[0] - bpespecs['bymax']
        bymax = img.shape[0] - bpespecs['bymin']

        for bpe_func in bpespecs.keys():
            if bpe_func not in valid_specs:
                raise ValueError("{} is not a valid crop dimension. Use: {}".format(bpe_func, valid_specs))

        img_original = copy.copy(img) # copy.deepcopy(img)

        # bpe mask
        nrows, ncols = np.shape(img)
        row, col = np.ogrid[:nrows, :ncols]
        bpe_mask_left = bpespecs['bxmin'] - col < 0
        bpe_mask_right = bpespecs['bxmax'] - col > 0
        bpe_mask_top = bymax - row > 0
        bpe_mask_bottom = bymin - row < 0
        bpe_mask_cols = np.logical_and(bpe_mask_left, bpe_mask_right)
        bpe_mask_rows = np.logical_and(bpe_mask_top, bpe_mask_bottom)
        bpe_mask = np.logical_and(bpe_mask_cols, bpe_mask_rows)
        img_bpe_masked = np.rint(copy.copy(img)) # TODO: FIX THIS?
        img_bpe_masked[~bpe_mask] = 0

        # filter bpe region
        raw_masked = ma.array(img.copy(), mask=~bpe_mask)
        raw_masked = raw_masked * bpespecs['multiplier']

        # store mask and update raw image
        img_bpe_mask = bpe_mask
        img = raw_masked.data

    return img, img_bpe_mask





def img_resize(img, method='rescale', scale=2, preserve_range=True):
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

    if scale == 1:
        pass
    elif method == "rescale":
        img = rescale(img, scale, order=1, mode='reflect', cval=0, clip=True, preserve_range=preserve_range,
                      multichannel=False, anti_aliasing=None, anti_aliasing_sigma=None)
    elif method == 'pyramid_expand':
        img = pyramid_expand(img, upscale=scale, sigma=None, order=1, mode='reflect', cval=0, multichannel=False,
                       preserve_range=preserve_range)

    if preserve_range is True:
        img = np.rint(img)
        img = img.astype(np.uint16)

    return(img)


def img_subtract_background(img, backgroundSubtractor=None, bg_method='KNN', bg_filepath=None, plot_background_subtraction=False):
    """
    This subtracts a background input image from the signal input image.
    :param img:         the image to calculate and subtract the background from
    :param img:         if algorithmic background subtraction is used, it is stored here.
    :param bg_method:   the method used for background subtraction
    :param bg_filepath: if manual background subtraction is used, the filepath leads to the background image.
    :return:
        img_bg      -   the background image                                        (non-background is set to zero)
        img_bgs     -   the background subtracted                                   (background is set to zero)
        img_mask    -   the binary mask used to calculate the background image      (some post-processing)
        img_masked  -   the binary mask after post-processing to smooth out         (non-binary array)

    Process Pipeline:
        1. if this manual or algorithmic subtraction
            A: if manual background subtraction
                1. if bg_filepath is a filepath (and not numpy array), then read file to disk.
                2. subtract the background image from the input image.
                3. assign img_bg, img_bgs, img_mask, and img_masked variables.
            B: if algorithmic background subtraction
                1. check the image datatype and change to 'uint8' if necessary.
                2. apply the background subtractor to the image.
                3. apply filtering to smooth background subtractor mask.
                3. assign img_bg, img_bgs, img_mask, and img_masked variables.
    """

    # check to make sure a valid background subtraction method is used
    valid_bs_methods = ['KNN', 'MOG2', 'CMG', 'min', 'mean']
    if bg_method is not None and bg_method not in valid_bs_methods:
        raise ValueError(
            "{} is not a valid method. Implemented so far are {}".format(bg_method, valid_bs_methods))

    # if "manual" background subtraction method
    if bg_method in ['min', 'mean']:

        # check to make sure background image filepath is present and the size matches the input image.
        if bg_filepath is None:
            raise ValueError(
                "Background image (bg_filepath) not found. Need to assign array/filepath for the background image.")
        else:
            if isinstance(bg_filepath, str):
                bg = io.imread(fname=bg_filepath)       # if filepath, read to disk.
            elif isinstance(bg_filepath, np.ndarray):
                bg = bg_filepath.copy()                 # if array, instantiate new variable.
            else:
                raise ValueError("Unknown data type of background image. Should filepath (str) or np.ndarray")

            if np.shape(img) != np.shape(bg):
                raise ValueError(
                    "Shape mismatch error: input image {} and background image {} need to be identical.".format(np.shape(img), np.shape(bg)))

        # perform background subtraction
        img_mask = np.ones_like(bg)
        img_bg = bg.copy()
        img_bgs = img - bg
        img_masked = img_bgs.copy()

    # algorithmic background subtraction
    elif bg_method in ['KNN', 'MOG2', 'CMG']:

        # rescale to full range b/c background subtractor performs much better in this manner.
        img_backSub = rescale_intensity(img, in_range='image', out_range='uint8')

        if backgroundSubtractor is None:
            raise ValueError("backgroundSubtractor is None. Need to instantiate backgroundSubtractor before this method")
        else:
            mask = backgroundSubtractor.apply(img_backSub)

        # perform background subtraction
        img_mask = mask
        img_masked = gaussian(mask, sigma=1, preserve_range=False)
        img_small_parts = white_tophat(img_masked, selem=disk(3))
        img_masked = img_masked - img_small_parts
        img_masked = np.asarray(np.rint(img_masked*255*254), dtype='uint16')
        img_bg = img.copy()
        img_bgs = img_bg

    # manual background subtraction (if bg_method == None)
    else:
        mask_manual = False
        adaptive_thresh = threshold_local(img, block_size=59, offset=-11)
        mask = img > adaptive_thresh
        mask_manual = True

        # masking
        img_masked = gaussian(mask, sigma=2.5, preserve_range=False)
        img_masked = np.asarray(np.rint(img_masked*255), dtype='uint16')
        img_mask = img_masked > np.median(img_masked)*1.35

        # "perform" background subtraction
        img_bg = img.copy()
        img_bg[img_mask] = 0
        img_bgs = img.copy()
        img_bgs[~img_mask] = 0

    if plot_background_subtraction:
        # ----- IF YOU WANT TO PLOT THE BACKGROUND SUBTRACTED IMAGE TO CHECK -----
        fig, axes = plt.subplots(ncols=3, figsize=(12,4))
        ax = axes.ravel()
        ax[0].imshow(img)
        ax[1].imshow(img_bg)
        ax[2].imshow(img_bgs)
        ax[0].set_title('input')
        ax[1].set_title('background')
        ax[2].set_title('input - background')
        plt.tight_layout()
        plt.show()

    return(img_bg, img_bgs, img_mask, img_masked)


def img_filter(img, filterspecs):
    """
    This is an image filtering function. The argument filterdict are similar to the arguments of the...
    e.g. filterdict: {'median': 5, 'gaussian':3}
    :param filterspecs:
    :param force_rawdtype:
    :return:

    """
    valid_filters = ['none','median','gaussian','white_tophat','rescale_intensity',
                     'denoise_wavelet', 'equalize_adapthist']
    filtering_sequence = 1

    for process_func in filterspecs.keys():
        if process_func not in valid_filters:
            img_filtered = None
            raise ValueError("{} is not a valid filter. Implemented so far are {}".format(process_func, valid_filters))
        elif process_func == "none":
            img_filtered = img
        elif process_func == "rescale_intensity":
            args = filterspecs[process_func]['args']
            vmin, vmax = np.percentile(img, (args[0][0], args[0][1]))
            img_filtered = rescale_intensity(img, in_range=(vmin, vmax), out_range=args[1])
        elif process_func == "equalize_adapthist":
            vmin, vmax = np.percentile(img, (0, 100))       # store original range of values
            kwargs = filterspecs[process_func]['kwargs']
            img_filtered = equalize_adapthist(img, **kwargs)
            img_filtered = rescale_intensity(img_filtered, out_range=(vmin, vmax))
        else:
            func = eval(process_func)
            args = filterspecs[process_func]['args']
            if 'kwargs' in filterspecs[process_func].keys():
                kwargs = filterspecs[process_func]['kwargs']
            else:
                kwargs = {}

            img_filtered = img_apply_filter(img, func, *args, **kwargs)
            filtering_sequence += 1

    return(img_filtered)

def img_apply_filter(img, func, *args, **kwargs):
    assert callable(func)
    return func(img, *args, **kwargs)

def img_find_particles(img, min_sigma=0.5, max_sigma=5, num_sigma=20, threshold=0.1, overlap=0.85):
    """
    This uses Laplacian of Gaussians method to determine the number and size of particles in the image.
    :return:
    """

    # particles
    particles = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                         threshold=threshold, overlap=overlap)
    particles[:, 2] = particles[:, 2] * np.sqrt(2)  # compute radius in the 3rd column

    return (particles)


def apply_flatfield_correction(img_list, darkfield, flatfield):
    for img in img_list.values():
        img.apply_flatfield_correction_to_img(darkfield, flatfield)

def apply_background_subtractor(img_list, backgroundSubtractor, bg_method='KNN', apply_to='filtered', bg_filepath=None):
    for img in img_list.values():
        img.image_subtract_background(image_input=apply_to, backgroundSubtractor=backgroundSubtractor, bg_method=bg_method, bg_filepath=bg_filepath)

def calculate_background_image(img_list, bg_method='min', apply_to='filtered', bg_filepath=None):
    baseline_img = 1

def analyze_img_quality(img_list):
    means = []
    stds = []
    snrs = []

    for img in img_list.values():
        img.calculate_stats()
        means.append(img.stats.raw_mean.values)
        stds.append(img.stats.raw_std.values)
        snrs.append(img.stats.raw_snr.values)

    mean = int(np.mean(means))
    std = int(np.mean(stds))
    snr = np.round(np.mean(snrs), 2)

    return mean, std, snr