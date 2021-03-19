from __future__ import print_function
import cv2 as cv
import argparse

pth = '/Users/mackenzie/Desktop/03.04.21-ZuPIVelastosil/tests_loc1/E2.5Vmm/imgs/%1d.tiff'

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='pth')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# read image sequence or video
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))

if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

###


"""
this file..
"""

# import modules
import numpy as np
import cv2
import ffmpeg
import tifffile as tiff
from skimage.morphology import disk
from skimage.filters import median, gaussian
import matplotlib.pyplot as plt


# set path variables
img_dir = 'Users/mackenzie/Desktop/03.04.21-ZuPIVelastosil/tests_loc1/E2.5Vmm/12/'
img_basename = 'test_1_X%2d'
img_type = '.tif'
img_path = img_dir+img_basename+img_type

vid_dir = img_dir
vid_basename = '12'
vid_type = '.avi'
vid_path = vid_dir+vid_basename+vid_type



# read video sequence
cap = cv2.VideoCapture(img_path, cv2.CAP_IMAGES)

# create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)


time=0
frame_id = 0
while(time<0.99):
    time = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)

    #fed = fgmask.copy()
    #fed = median(fgmask, disk(2))
    #fed = gaussian(fgmask,sigma=0.5, preserve_range=False)

    gray_masked = cv2.bitwise_or(gray, gray, mask=fgmask)

    # contrast enhancement
    # global
    #gray_enhanced = cv2.equalizeHist(gray_masked)
    # local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray_masked)

    flip = cv2.flip(gray_enhanced, 0)
    # write the flipped frame
    img_save = img_dir+'imgs/'+str(frame_id)+img_type
    #cv2.imwrite(img_save,gray_enhanced)
    tiff.imsave(img_save, gray_enhanced)
    frame_id += 1

    cv2.imshow("original", gray_enhanced)
    if cv2.waitKey(1) == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()



