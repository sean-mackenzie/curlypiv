# test CurlypivImage
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
import numpy as np

# Image processing
import cv2 as cv
# skimage
from skimage import img_as_float
from skimage import io
from skimage.morphology import disk, white_tophat
from skimage.filters import median, gaussian
from skimage.exposure import rescale_intensity, equalize_adapthist, equalize_hist
from skimage.transform import pyramid_expand, rescale

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def image_details(frame):
    #print(frame)
    test_num = re.search('test_(.*)_X', frame).group(1)
    if test_num == '_': test_num = int(0)
    frame_num = int(re.search('_X(.*).tif', frame).group(1))
    return(test_num, frame_num)



videoPath2 = '/Users/mackenzie/Desktop/fm250Hz_square.avi'
imagePath = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test/tests/loc1/E2.5Vmm/run3num'

bg_method = 'KNN'
read_type = 'video'
cmap = 'hot'
alpha = 0.15


sigma = 0.75
save = False
show = True
crop = False
filt = False
scale = False
bg_subtract = True


if bg_method == 'KNN':
    backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
elif bg_method == "MOG2":
    backSub = cv.createBackgroundSubtractorMOG2(detectShadows=False)
elif bg_method == 'CMG':
    backSub = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames=None)
elif bg_method == 'GSOC':
    backSub = cv.bgsegm.createBackgroundSubtractorGSOC(blinkingSupressionMultiplier=0.1)

list_imgs = glob.glob(imagePath+'/*'+'.tif')
list_imgs.sort(key = image_details)

counter = 0

title = "{} background subtraction applied to {}".format(bg_method, read_type)


if read_type == 'images':

    for i in list_imgs:

        frame = cv.imread(i, 0)
        orig = frame.copy()
        #frame = io.imread(i, plugin='tifffile')

        if crop:
            # crop to reduce size and time to calculate
            cropspecs = {
                'xmin': 80,  # x = 0 is the left of the image
                'xmax': 450,
                'ymin': 200,
                'ymax': 450  # y = 0 is the bottom of the image
            }
            ymin = frame.shape[1] - cropspecs['ymax']
            ymax = frame.shape[1] - cropspecs['ymin']
            frame = frame[ymin:ymax, cropspecs['xmin']:cropspecs['xmax']]

        if filt:
            # perform CLAHE histogram equilization
            frame = equalize_adapthist(frame, clip_limit=0.02)

        if scale:
            # upscale 2X to improve spatial resolution and smooth with pyramid_expand
            frame = pyramid_expand(frame, upscale=2, sigma=None, order=1, mode='reflect', cval=0, multichannel=False,
                                   preserve_range=True)

        if bg_subtract:
            # subtract the background
            # calculate the median for background subtraction
            med = np.median(frame)

            frame = frame - med / 2
            frame = np.where(frame < 0, 0, frame)
            #frame = np.asarray(np.rint(frame), dtype='uint8')

        # rescale to improve background subtractor
        if filt or bg_subtract or scale:
            n = 255
        else:
            n = 1
        vmin, vmax = np.percentile(frame, (4, 99.995))
        frame = rescale_intensity(frame, in_range=(vmin, vmax), out_range='dtype')

        frame = np.asarray(np.rint(frame*n), dtype='uint8')

        # BG #1 - apply background subtractor
        fgMask = backSub.apply(frame)

        #cv.imshow('FG Mask', fgMask)

        # BG #2 - smooth mask
        fgMask = gaussian(fgMask, sigma=sigma, preserve_range=True)
        fgMask = np.asarray(np.rint(fgMask), dtype='uint8')

        # save image
        if save == True and counter > 50:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.imshow(orig, cmap='gray')
            #ax.imshow(fgMask, cmap=cmap, alpha=alpha)

            plt.title(title + ', Frame: {}'.format(counter))
            ax.set_axis_off()
            plt.tight_layout()

            #saveMaskPath = imagePath+'/writeFromImages{}/'.format(bg_method)+str(counter)+'.png'
            saveMaskPath = imagePath + '/raw/' + str(counter) + '.png'
            plt.savefig(saveMaskPath, dpi=100)

            if counter > 60:
                break

        print(counter)
        counter += 1

elif read_type == 'video':

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(videoPath2))

    if not capture.isOpened:
        print('Unable to open the video')
        exit(0)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        # BG #2 - smooth mask
        fgMask = gaussian(fgMask, sigma=sigma, preserve_range=True)
        fgMask = np.asarray(np.rint(fgMask), dtype='uint8')

        # save image
        if save == True and counter > 50:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.imshow(fgMask, cmap='gray')
            #ax.imshow(fgMask, cmap=cmap, alpha=alpha)

            plt.title(title + ', Frame: {}'.format(counter))
            ax.set_axis_off()
            plt.tight_layout()

            svpath = '/Users/mackenzie/Desktop/results'
            saveMaskPath = svpath+ '/'.format(bg_method)+'mask'+str(counter)+'.png'
            plt.savefig(saveMaskPath, dpi=100)

            if counter > 100:
                break

        if counter > 50 and show == True and save == False:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.imshow(fgMask, cmap='gray')
            #ax.imshow(fgMask, cmap=cmap, alpha=alpha)

            plt.title(title + ', Frame: {}'.format(counter))
            ax.set_axis_off()
            plt.tight_layout()

            plt.show()

        print(counter)
        counter += 1
        
