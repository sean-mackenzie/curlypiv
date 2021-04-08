# test CurlypivTestSetup - calibrateCamera
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import sys
import os
from os.path import join
import glob
import re

# scientific
from math import ceil, floor
import numpy as np
from scipy import signal, misc

# Image processing
import cv2 as cv
# skimage
from skimage import img_as_float
from skimage import io
from skimage.util import invert
from skimage.morphology import disk, white_tophat
from skimage.filters import median, gaussian, threshold_multiotsu
from skimage.filters.rank import mean
from skimage.restoration import denoise_wavelet
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, pyramid_expand
from skimage.feature import blob_log
# sci-kit-learn
from sklearn.neighbors import NearestNeighbors

# plotting
import matplotlib
import matplotlib.pyplot as plt
import cycler

# Curlypiv




def measureIlluminationDistributionXY(basePath=None, illumPath=None, show_image=False, save_image=False, save_img_type='.tif',
                                      save_txt=False, show_plot=False, save_plot=False, savePath=None, savename=None):

    if basePath is not None:
        illumPath = join(basePath, 'setup/calibration/illumination')
        savePath = join(basePath, 'setup/calibration/results')
    elif illumPath is None:
        raise ValueError("Most input either a base path with proper data structure or illumination file path")
    if savename is None:
        savename = 'measureFlatField'

    # step 0 - initialize
    particle_diameter_pixels_estimate = 20
    n_gaussians = 5
    sigma_multiplier = 8
    cmap_orig = 'gray'
    cmap_smooth = 'cool'
    fontsize = 18

    # step 1 - get image list
    img_list = glob.glob(illumPath + '/*.tif')
    num_images = len(img_list)

    # step 2 - read image or image collection
    if num_images > 1:
        img = io.imread_collection(load_pattern=img_list, conserve_memory=False, plugin='tifffile')
        img = np.asarray(np.rint(np.mean(img, axis=0)), dtype='uint16')
    elif num_images == 1:
        img = io.imread(illumPath, plugin='tifffile')
        if img is None:
            raise ValueError("if num_images==1, then illumPath must be the direct image file path")
    else:
        raise ValueError("No images were found.")

    img_orig = img.copy()
    vmin_orig, vmax_orig = np.percentile(img_orig, (0, 100))

    # step 2 - smooth out most features in image
    img_small_bright = white_tophat(img, selem=disk(particle_diameter_pixels_estimate))
    img = img - img_small_bright
    img_particles_removed = img.copy()
    for i in range(n_gaussians):
        img = gaussian(img, sigma=i*sigma_multiplier, mode='nearest')

    # step 4 - normalize to 1
    vmin, vmax = np.percentile(img, (0.05, 99.95))
    img = rescale_intensity(img, in_range=(0, vmax), out_range='dtype')

    # calculate flatfield correction
    img_flatfield_correction = 1 / img

    # flatfield correction original image
    img_orig_flatfield_corrected = img_orig * img_flatfield_correction
    img_orig_flatfield_corrected = rescale_intensity(img_orig_flatfield_corrected, in_range='image', out_range='uint16')

    # save text
    if save_txt:
        np.savetxt(fname=savePath+'/'+savename+'.txt', X=img, fmt='%f')

    if save_image:
        img_save = rescale_intensity(img, in_range='image', out_range='uint16')
        io.imsave(fname=savePath+'/'+savename+save_img_type, arr=img_save, plugin='tifffile')
        io.imsave(fname=savePath+'/'+'measureFlatFieldCorrection'+save_img_type, arr=img_flatfield_correction, plugin='tifffile')
    if show_image:
        io.imshow(img)

    # step 3 - plot images
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(11,10), sharex=True, sharey=True, tight_layout=True)
    ax = axes.ravel()

    ax[0].imshow(img_orig, cmap=cmap_orig)
    ax[0].set_title('Mean of {} raw images'.format(num_images), fontsize=fontsize)

    ax[1].imshow(img_particles_removed, cmap=cmap_smooth)
    ax[1].set_title(r'$p_{diameter}<$'+'{} pixels removed'.format(particle_diameter_pixels_estimate), fontsize=fontsize)

    ax[2].imshow(img_orig_flatfield_corrected, cmap=cmap_orig)
    ax[2].set_title('Flatfield corrected', fontsize=fontsize)

    ax[3].imshow(img, cmap=cmap_smooth)
    ax[3].set_title('Smoothed: {} Gaussians, {} * sigma'.format(n_gaussians, sigma_multiplier), fontsize=fontsize)

    plt.suptitle('Evaluation of Illumination Distribution')

    if save_plot:
        plt.savefig(fname=savePath+'/'+savename+'_image'+'.jpg')

    if show_plot:
        plt.show()

    # step 3 - plot intensity profiles
    n = 5
    cmap = matplotlib.cm.cool(np.linspace(0,1,n))
    plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', cmap)
    fig, axes = plt.subplots(ncols=2, figsize=(12,5), sharey=True, tight_layout=True)
    ax = axes.ravel()

    if np.shape(img)[0] > np.shape(img)[1]:
        segs = floor(np.shape(img)[0] / n)
    else:
        segs = floor(np.shape(img)[1] / n)

    for j in range(len(np.shape(img))):
        for i in range(n):
            i = int(segs * (i + 0.5))
            if j == 0:
                ints = img[:,i]
                lbl = 'y'
            elif j == 1 and i <= np.shape(img)[0]:
                ints = img[i,:]
                lbl = 'x'

            ax[j].plot(ints, label=i)
            ax[j].set_xlabel(lbl)
        ax[0].set_ylabel('Normalized intensity')

    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.suptitle('Pixel Row/Column Intensity Profiles')
    plt.tight_layout()

    if save_plot:
        plt.savefig(fname=savePath+'/'+savename+'_profile'+'.jpg')

    if show_plot:
        plt.show()

    plt.close('all')

    img = np.asarray(np.rint(rescale_intensity(img, in_range='image', out_range=(vmin_orig, vmax_orig))), dtype='uint16')

    return img, img_flatfield_correction


def particle_illumination_distribution(x, y, z=None,
                illum_xy=None, framex=None, framey=None, startx=None, starty=None,
                scale_z=False, focal_z=None, testSetup=None):

    # --- XY plane illumination ---

    # step 1: ensure there is a mask
    if illum_xy is not None:
        mask = illum_xy
    else:
        # if no mask but mask is inside illumination frame, we must create the mask distribution

        # convolve the frame area with an equal size gaussian
        if framex <= framey:
            r = framex
        else:
            r = framey

        # initialize
        sigma =2
        p_array = np.ones((framey, framex))
        kernel = np.outer(signal.windows.gaussian(M=r, std=sigma), signal.windows.gaussian(M=r, std=sigma ,))

        # convolve ones array with square gaussian of smallest side length
        mask = signal.fftconvolve(p_array, kernel, mode='valid')

    # rescale to max of 1
    g_max = np.max(mask)
    mask = rescale_intensity(image=mask, in_range=(0 ,g_max), out_range=(0 ,1))

    # step 2: compare the particle location to mask location
    if x > startx and x < startx + framex and y > starty and y < starty + framey:
        xstep = x - startx
        ystep = y - starty
        c_xy_int = mask[int(ystep), int(xstep)]   # assign c_int
    else:
        c_xy_int = 0

    # step 5: particle intensity
    c_int = c_xy_int

    # --- Z plane illumination ---
    if scale_z is True:

        # step 0: adjust for units
        if np.mean(z) > 1:
            z = z * 1e-6
        if np.mean(focal_z) > 0.1:
            focal_z = focal_z * 1e-6

        # step 1: calculate the depth of correlation for the optical setup
        depth_of_correlation = testSetup.optics.microscope.objective.depth_of_correlation

        # step 3: compare particle z-height and weighting function distribution
        c_z_int = calculate_particle_correlation(z, z_focal=focal_z, depth_of_corr=depth_of_correlation)

        # step 5: combined particle intensity
        c_int = c_xy_int * c_z_int

    return c_int

def calculate_depth_of_correlation(M, NA, dp, n=1, lmbda=595e-9, eps=0.01):

    z_corr = ((1 - np.sqrt(eps)) / np.sqrt(eps) * ((n ** 2 * dp ** 2) / (4 * NA ** 2) +
                                                       (5.95 * (M + 1) ** 2 * lmbda ** 2 * n ** 4) / (
                                                               16 * M ** 2 * NA ** 4))) ** 0.5
    depth_of_correlation = 2 * z_corr

    return depth_of_correlation

def calculate_distribution_of_correlation(depth_of_corr, plot_weight=False):
    """
    Note: z_weigth is an array.
    """

    z_corr_space = np.linspace(-depth_of_corr/2, depth_of_corr/2, num=200)
    z_weight = 1 / (1 + (3 * z_corr_space / depth_of_corr) ** 2) ** 2

    # step 3: plot the weighting function
    if plot_weight is True:
        plot_distribution_of_correlation()

    return z_weight, z_corr_space

def calculate_particle_correlation(z, z_focal, depth_of_corr):

    z_field = np.linspace(-depth_of_corr/2, depth_of_corr/2, num=250) + z_focal

    if z < np.max(z_field) and z > np.min(z_field):
        c_z_corr = 1 / (1 + (3 * np.abs(z_focal - z) / depth_of_corr/2) ** 2) ** 2
    else:
        c_z_corr = 0

    return c_z_corr

def plot_distribution_of_correlation(z_corr_space, z_weight, focal_z=0, fullz=20e-6):
    fig, ax = plt.subplots()

    # plot weighting function
    ax.scatter(z_corr_space, z_weight, s=5)
    ax.plot(z_corr_space, z_weight, alpha=0.25, linewidth=2)

    # plot focal plane
    plt.vlines(x=focal_z, ymin=0, ymax=1, colors='r', linestyles='dashed', alpha=0.25, label='focal plane')

    # plot channel walls
    plt.vlines(x=-focal_z * 1e6, ymin=0, ymax=1, colors='gray', linestyles='solid', alpha=0.25,
               label='channel walls')
    plt.vlines(x=(fullz - focal_z) * 1e6, ymin=0, ymax=1, colors='gray', linewidth=2, linestyles='solid',
               alpha=0.5)

    ax.set_xlabel('z-position (um)')
    ax.set_ylabel('weighted-contribution to PIV')
    plt.title("Real Depth-Correlated Weight")
    plt.legend()

    plt.show()

def plot_field_depth(depth_of_corr, depth_of_field=None, show_depth_plot=False, save_depth_plot=False, basePath=None, savename=None,
                     channel_height=None, objective=None):

    z_weight, z_space = calculate_distribution_of_correlation(depth_of_corr, plot_weight=False)

    if show_depth_plot or save_depth_plot:

        fig, ax = plt.subplots(figsize=(10,4))

        # plot focal plane
        plt.vlines(x=0, ymin=0, ymax=100, colors='r', linestyles='dashed', alpha=0.75, label='Focal Plane')

        # plot weighting function
        ax.plot(z_space * 1e6, z_weight*100, color='blue', alpha=1, linewidth=2, label='Depth of Correlation')

        # plot depth of field
        plt.vlines(x=-depth_of_field/2 * 1e6, ymin=0, ymax=100, colors='lightblue', linewidth=2, linestyles='solid', alpha=0.9, label='Depth of Field')
        plt.vlines(x=depth_of_field/2 * 1e6, ymin=0, ymax=100, colors='lightblue', linewidth=2, linestyles='solid', alpha=0.9)

        # plot channel walls
        plt.vlines(x=-channel_height/2 * 1e6, ymin=0, ymax=100, colors='gray', linewidth=2, linestyles='solid', alpha=0.5, label='Channel walls')
        plt.vlines(x=channel_height/2 * 1e6, ymin=0, ymax=100, colors='gray', linewidth=2, linestyles='solid', alpha=0.5)

        # figure setup
        plt.title('Interrogation Volume for {}X Objective'.format(objective))
        ax.set_xlabel('z (um)')
        ax.set_ylabel('% Correlation')
        plt.legend()    # bbox_to_anchor=(1.04,1), loc="upper left"
        plt.tight_layout()

        if save_depth_plot:
            if savename is None:
                savename = 'measure{}XObjectiveInterrogationVolume'.format(objective)
            plt.savefig(join(basePath,'setup/calibration/results', savename+'.jpg'))

        if show_depth_plot:
            plt.show()

def calculate_darkfield(basePath=None, darkframePath=None, show_image=False, save_image=False, save_img_type='.tif',
                                      savePath=None, savename=None, save_plot=False):

    if basePath is not None:
        darkframePath = join(basePath, 'setup/calibration/cameraNoise/darkfield')
        savePath = join(basePath, 'setup/calibration/results')
    elif darkframePath is None:
        raise ValueError("Most input either a base path with proper data structure or dark frame file path")
    if savename is None:
        savename = 'measureDarkField'

    # step 1 - get image list
    img_list = glob.glob(darkframePath + '/*.tif')
    num_images = len(img_list)

    if num_images > 1:
        img = io.imread_collection(load_pattern=img_list, conserve_memory=False, plugin='tifffile')
        img = np.asarray(np.rint(np.mean(img, axis=0)), dtype='uint16')

    elif num_images == 1:
        img = io.imread(img_list[0], plugin='tifffile')

        if len(np.shape(img)) > 2:
            img = np.asarray(np.rint(np.mean(img, axis=0)), dtype='uint16')

    else:
        raise ValueError("No images were found.")


    # measure image mean and std
    darkfield_mean = np.round(np.mean(img),1)
    darkfield_std = np.round(np.std(img),2)

    # save image
    if save_image:
        io.imsave(join(savePath, savename+save_img_type), img, plugin='tifffile', check_contrast=False)

    # save or show image
    if save_plot or show_image:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title('Dark field: Mean={}, Std={}'.format(darkfield_mean, darkfield_std))

        if save_plot:
            plt.savefig(join(savePath, savename+'.jpg'))

        if show_image:
            plt.show()

    return img, darkfield_mean, darkfield_std











