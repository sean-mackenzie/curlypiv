# test CurlypivTestSetup - calibrateCamera
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import sys
import os
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
from curlypiv.CurlypivTestCollection import CurlypivTestCollection
from curlypiv.CurlypivTestSetup import CurlypivTestSetup
from curlypiv.CurlypivUtils import round_to_odd
from curlypiv.utils.calibrateCamera import measureIlluminationDistributionXY


# Inputs
analyses = ['illumination']     # ['illumination', 'grid', 'scaling', 'distortion']
est_particle_diameter = 5

illumPath = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test/tests/loc1/E2.5Vmm/run3num/1'

illumSavePath = '/Users/mackenzie/Desktop/testSynthetics/save/illumXY'
illumSaveName = 'iXY'


gridPath = '/Users/mackenzie/Box/2019-2020/Research/Data Storage/BPE/Experimental/03.10.21 - 50X objective ' \
           'microgrids/grid2_2.7mmPDMS_cc=0/z=0/grid2.tif'
gridDepth = True # True if the image is a stacked tiff
gridPoints = 'circles'



# --- Test 1: Measure illumination distribution across image ---
img_iXY = measureIlluminationDistributionXY(illumPath=illumPath, num_images=50, show_image=False, save_image=True, save_img_type='.tif',
                                            save_txt=True, show_plot=False, save_plot=True, savePath=illumSavePath, savename=illumSaveName)


# --- TEST 1: Measure Grid and Pixel Scaling ---
if 'grid' in analyses:

    # step 1 - read the input image
    grid = io.imread(gridPath, plugin='tifffile')

    if gridDepth == True or len(grid.shape) > 2:
        grid = np.mean(grid, axis=0)

    # step 2 - smooth out most features in image
    grid = invert(grid)
    grid = median(grid, selem=disk(3))
    grid = gaussian(grid, sigma=1)
    grid = denoise_wavelet(grid, method='BayesShrink', mode='soft', rescale_sigma=True)

    # step 3 - use bandpass filter to get correct sized bright spots
    large_spots = white_tophat(grid, selem=disk(5))
    small_spots = white_tophat(grid, selem=disk(1))
    newgrid = large_spots - small_spots

    # step 4 - find location of particles
    if gridPoints == 'circles':

        # crop image to find particles not near the edge
        shift = est_particle_diameter * 3
        grid_search = newgrid[0+shift:newgrid.shape[0]-shift, 0+shift:newgrid.shape[1]-shift]

        # step 4.1 - find particle locations and compute radius
        particles = blob_log(grid_search, min_sigma=4, max_sigma=5, num_sigma=5, threshold=200, overlap=0.85)

        # adjust particle positions and compute radius in 3rd column
        particles[:,0] = particles[:,0] + shift
        particles[:,1] = particles[:,1] + shift
        particles[:, 2] = particles[:, 2] * np.sqrt(2)  # compute radius in the 3rd column
        print(len(particles))

        # step 5 - calculate mean of particle intensity

        sorted(particles, key=lambda rad: rad[2])
        newgrid_int = np.asarray(np.rint(newgrid), dtype='uint16')
        n = 1  # constant multiplier for particle identification region
        mean_ints = []
        mean_psfs = []
        corrs = []

        for p in particles:
            y, x, r = p

            # slice the array to just the particle
            ymin = round_to_odd(y-r*n)
            ymax = round_to_odd(y+r*n)
            xmin = round_to_odd(x-r*n)
            xmax = round_to_odd(x+r*n)
            p_array = newgrid_int[ymin:ymax, xmin:xmax]     # [ymin:ymax, xmin:xmax]

            # calculate the mean particle intensity across the array slice
            mean_array = mean(p_array, selem=disk(r))
            mean_int = mean_array[mean_array.shape[0]//2, mean_array.shape[1]//2]
            mean_ints.append(mean_int)

            # convolve the particle with a gaussian
            r_upper = ceil(r)
            sigma=2
            kernel = np.outer(signal.windows.gaussian(M=r_upper, std=sigma),
                                signal.windows.gaussian(M=r_upper, std=sigma,))
            blurred = signal.fftconvolve(p_array, kernel, mode='valid')
            mean_gauss = np.mean(blurred)
            mean_psfs.append(mean_gauss)

            corr_array = signal.correlate2d(p_array, kernel, mode='valid')
            corr_max = np.max(corr_array)
            corr_norm = np.mean(corr_array)/np.max(corr_array)
            corrs.append(corr_norm)



        # filter particles graphically by adjusting n
        n = 0.75
        x = range(len(mean_ints))

        # mean intensity
        mean_int = np.mean(mean_ints)
        std_int = np.std(mean_ints)

        # psf convolution
        mean_psf = np.mean(mean_psfs)
        std_psf = np.std(mean_psfs)


        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,6))
        ax = axes.ravel()

        ax[0].imshow(p_array, cmap='viridis')

        ax[2].plot(mean_ints)
        ax[2].hlines(mean_int,0,len(mean_ints))
        ax[2].fill_between(x, mean_int-std_int*n, mean_int+std_int*n, alpha=0.2)

        ax[1].imshow(blurred, cmap='viridis')

        #ax[3].plot(mean_psfs)
        ax[3].plot(corrs)



        plt.show()

        # create new list from mean +/- std filter
        def condition(x, mean, std, n): return x > mean - std*n and x < mean + std*n
        filtered_particles = [idx for idx, value in enumerate(mean_ints) if condition(value, mean_int, std_int, n)]
        particles = [particles[i] for i in filtered_particles]

        # step 5 - find the nearest neighbors



    # step  - threshold to get contours
    thresholds = threshold_multiotsu(newgrid, classes=2)
    regions = np.digitize(newgrid, bins=thresholds)




    # show image
    fig, axes = plt.subplots(ncols=2)
    ax = axes.ravel()

    # original-ish
    ax[0].imshow(regions, cmap='gray')
    ax[0].set_axis_off()

    # particles
    ax[1].imshow(newgrid, cmap='gray')
    ax[1].set_axis_off()
    for p in particles:
        y, x, r = p
        c = plt.Circle((x,y), r, color='red', linewidth=1, fill=False)
        ax[1].add_patch(c)

    plt.show()

    radius_mean = np.mean(particles[:,2])
    radius_med = np.median(particles[:,2])

    print(radius_mean)
    print(radius_med)


# --- TEST 2: Test Distortion on Grid ---

