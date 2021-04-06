# test CurlypivImage
"""
Notes about program
"""

# 1.0 import modules
# data I/O
from os.path import isfile, basename

# Maths/Scientifics
import numpy as np
import pandas as pd

# Image Processing
import cv2 as cv
import imutils
from skimage import io
from skimage import data, filters, measure, morphology

# Curlypiv
from curlypiv.CurlypivUtils import find_substring
from curlypiv import CurlypivImageProcessing
from curlypiv.CurlypivImageProcessing import resize, subtract_background, filter



# 2.0 define class

class CurlypivFile(object):
    """
    This class holds an image along with its properties such as:
    raw image, filtered, image, path, filename, and statistics.
    """

    def __init__(self, path, img_type, pre_calc_bg=None, pre_calc_mask=None):
        super(CurlypivFile, self).__init__()

        # Attributes with an underscore as first character are for internal use only.

        if not isfile(path):
            raise ValueError("{} is not a valid file".format(path))

        self._filepath = path
        self._filename = basename(path)
        self._filetype = img_type

        # Read the file path to determine test collection identifiers
        self._sequence = find_substring(self._filename,leadingstring='test_',trailingstring='_X', dtype=int)
        self._frame = find_substring(self._filename,leadingstring='X',trailingstring=self._filetype, dtype=int)

        # Load the image. This sets the ._raw attribute which is the primary raw/original image.
        self.load(path)

        # Crop the raw image to a new size. This sets the ._original attribute to store the original image file. If the
        # image is not cropped or resized, then the ._original attribute is left as None.
        self._original = None   # store original image
        self._bg = pre_calc_bg  # background image
        self._bgs = None        # store background subtracted image
        self._filtered = None
        self._processing_stats = None
        self._mask = pre_calc_mask
        self._masked = None

    def __repr__(self):
        class_ = 'CurlypivImage'
        repr_dict = {'Dimensions': self.shape,
                     'Sequence': self._sequence,
                     'Frame': self._frame}

        out_str = "{}, {} \n".format(class_, self.filename)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def load(self, path):
        img = io.imread(self._filepath, plugin='tifffile')
        self._raw = img.copy()

    def _update_processing_stats(self, names, values):
        if not isinstance(names, list):
            names = [names]
        new_stats = {}

        for name, value in zip(names, values):
            new_stats.update({name: [value]})
        new_stats = pd.DataFrame(new_stats)

        if self._processing_stats is None:
            self._processing_stats = new_stats
        else:
            self._processing_stats = new_stats.combine_first(self._processing_stats)

    def image_resize(self, method='rescale', scale=2):

        if self._original is None: self._original = self._raw.copy()

        self._raw = resize(self._raw, method=method, scale=scale)

    def image_crop(self, cropspecs):
        """
        This crops the image.
        The argument cropsize is a dictionary of coordinates with (0,0) as the bottom left corner of the image.
        :param cropspecs:
        :return:
        """

        valid_crops = ['xmin','xmax','ymin','ymax']
        # example = [20, 500, 0, 400]

        ymin = self.shape[1] - cropspecs['ymax']
        ymax = self.shape[1] - cropspecs['ymin']

        if cropspecs == None:
            pass
        else:
            for crop_func in cropspecs.keys():
                if crop_func not in valid_crops:
                    raise ValueError("{} is not a valid crop dimension. Use: {}".format(crop_func, valid_crops))

            if self._original is None: self._original = self._raw.copy()
            self._raw = self._raw[ymin:ymax, cropspecs['xmin']:cropspecs['xmax']]

    def image_subtract_background(self, image_input='raw', backgroundSubtractor=None, bg_filepath=None, bg_method="MOG2"):
        """
        This subtracts a background input image from the signal input image.
        :param bg_method:
        :param bg_img:
        :return:
        """
        valid_images = ['raw', 'filtered']

        if image_input not in valid_images:
            raise ValueError("{} not a valid image for filtering. Use: {}".format(image_input, valid_images))
        elif image_input == 'raw':
            input = self._raw.copy()
        elif image_input == 'filtered':
            if self._filtered is None:
                ValueError("This file has no filtered image")
            input = self._filtered.copy()

        (self._bg, self._bgs, self._mask, self._masked) = subtract_background(input,
                                                                              backgroundSubtractor=backgroundSubtractor,
                                                                              bg_filepath=bg_filepath,
                                                                              bg_method=bg_method)

    def image_filter(self, filterspecs, image_input='raw', image_output='filtered', force_rawdtype=True):
        """
        This is an image filtering function. The argument filterdict are similar to the arguments of the...
        e.g. filterdict: {'median': 5, 'gaussian':3}
        :param filterspecs:
        :param force_rawdtype:
        :return:

        This method should self-assign self._processing_stats)
        """
        raw_dtype = self._raw.dtype

        valid_images = ['raw', 'bgs', 'filtered', 'mask', 'masked', 'bg']

        if image_input == 'bg':
            user_confirmed_image = input("Are you sure you want to filter the background image? Enter 'y' or other image in {}".format(valid_images))
            if user_confirmed_image != 'y':
                image = user_confirmed_image

        if image_input not in valid_images:
            raise ValueError("{} not a valid image for filtering. Use: {}".format(image_input, valid_images))
        elif image_input == 'raw':
            input = self._raw.copy()
        elif image_input == 'bgs':
            if self._bgs is None:
                ValueError("This file has no background subtracted image (bgs)")
            input = self._bgs.copy()
        elif image_input == 'filtered':
            if self._filtered is None:
                ValueError("This file has no filtered image")
            input = self._filtered.copy()
        elif image_input == 'mask':
            if self._mask is None:
                ValueError("This file has no mask")
            input = self._mask.copy()
        elif image_input == 'masked':
            if self._masked is None:
                ValueError("This file has no masked image")
            input = self._masked.copy()
        elif image_input == 'bg':
            if self._bg is None:
                ValueError("This file has no background image")
            input = self._bg.copy()
        else:
            ValueError("A matching image input from {} was not found".format(valid_images))

        # perform filtering
        output = filter(input, filterspecs=filterspecs)

        if force_rawdtype and output.dtype != raw_dtype:
            output = output.astype(raw_dtype)

        valid_outputs = ['bgs', 'filtered', 'masked', 'bg']
        if image_output not in valid_outputs:
            raise ValueError("{} not a valid image for output. Use: {}".format(image_output, valid_outputs))

        if image_output == 'bgs':
            self._bgs = output
        elif image_output == 'filtered':
            self._filtered = output
        elif image_output == 'mask':
            self._mask = output
        elif image_output == 'masked':
            self._masked = output
        elif image_output == 'bg':
            self._bg = output

    def calculate_stats_zeta(self, u, v):
        """
        this calculate...
        """
        M = np.hypot(u, v)
        n = 1

        self.u_mean = np.round(np.mean(u),1)
        self.v_mean = np.round(np.mean(v),1)
        if self.u_mean < 0:
            n = -1
        self.M_mean = np.round(np.mean(M*n),1)
        self.u_std = np.round(np.std(u),2)
        self.v_std = np.round(np.std(v),2)
        self.M_std = np.round(np.std(M),2)

    def calculate_stats(self):
        """
        This calculates some basic image statistics and updates the processing stats
        :return:
        """
        raw_mean = self._raw.mean()
        raw_std = self._raw.std()

        if self._filtered is not None:
            # pixel intensities
            filt_mean = self._filtered.mean()
            filt_std = self._filtered.std()
        else:
            filt_mean = None
            filt_std = None

        self._update_processing_stats(['raw_mean','raw_std','filt_mean','filt_std'],
                                      [raw_mean, raw_std, filt_mean, filt_std])

    def identify_particles(self, min_size=None, max_size=None, shape_tol=0.1, overlap_threshold=0.3):

        if shape_tol is not None:
            assert 0 < shape_tol < 1

        # Identify particles
        contours = cv.findContours(self.mask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        bboxes = [cv.boundingRect(contour) for contour in contours]

        id_ = 0
        # Sort contours and bboxes by x-coordinate:
        skipped_cnts = []
        for cont_bbox in sorted(zip(contours, bboxes), key=lambda b: b[1][0], reverse=True):
            contour = cont_bbox[0]
            contour_area = abs(cv.contourArea(contour))
            # get perimeter
            contour_perim = cv.arcLength(contour, True)

            # If specified, check if contour is too small or too large. If true, skip the creation of the particle
            if min_size is not None:
                if contour_area < min_size:
                    skipped_cnts.append(contour)
                    continue
            if max_size is not None:
                if contour_area > max_size:
                    skipped_cnts.append(contour)
                    continue

            bbox = cont_bbox[1]

            if shape_tol is not None:
                # Discard contours that are clearly not a circle just by looking at the aspect ratio of the bounding box
                bbox_ar = bbox[2] / bbox[3]
                if abs(np.maximum(bbox_ar, 1 / bbox_ar) - 1) > shape_tol:
                    skipped_cnts.append(contour)
                    continue
                # Check if circle by calculating thinness ratio
                tr = 4 * np.pi * contour_area / contour_perim**2
                if abs(np.maximum(tr, 1 / tr) - 1) > shape_tol:
                    skipped_cnts.append(contour)
                    continue



    def image_find_particles(self, image_input='filtered', min_sigma=0.5, max_sigma=5,num_sigma=20, threshold=0.1,overlap=0.85):
        """
        This uses Laplacian of Gaussians method to determine the number and size of particles in the image.
        :return: 
        """

        valid_images = ['raw', 'bgs', 'filtered', 'mask', 'masked']

        if image_input not in valid_images:
            raise ValueError("{} not a valid image for filtering. Use: {}".format(image_input, valid_images))
        elif image_input == 'raw':
            input = self._raw.copy()
        elif image_input == 'bgs':
            if self._bgs is None:
                ValueError("This file has no background subtracted image (bgs)")
            input = self._bgs.copy()
        elif image_input == 'filtered':
            if self._filtered is None:
                ValueError("This file has no filtered image")
            input = self._filtered.copy()
        elif image_input == 'mask':
            if self._mask is None:
                ValueError("This file has no mask")
            input = self._filtered.copy()
        elif image_input == 'masked':
            if self._masked is None:
                ValueError("This file has no masked image")
            input = self._filtered.copy()
        else:
            ValueError("A matching image input from {} was not found".format(valid_images))

        # particle identification
        particles = CurlypivImageProcessing.find_particles(img=input,min_sigma=min_sigma, max_sigma=max_sigma,
                                                           num_sigma=num_sigma, threshold=threshold, overlap=overlap)

        # stats
        num_particles = len(particles)
        particle_density = np.sum(np.square(particles[:,2])*np.pi, axis=0)/np.size(input)*100

        if particle_density >= 100:
            raise ValueError("Particle density should not be > 100%. Need to reduce overlap or increase threshold")

        self._update_processing_stats(['num_particles','particle_density'],[num_particles,particle_density])

        return(particles)

    def empty_file(self, to_empty='all'):
        # code for removing a specific attributes
        # would go here

        if to_empty == 'all':
            self.filtered = None
            self.mask = None
            self.masked = None
            self.raw = None



    @property
    def name(self):
        return self._filename

    @property
    def path(self):
        return self._filepath

    @property
    def filename(self):
        return self._filename

    @property
    def filepath(self):
        return self._filepath

    @property
    def sequence(self):
        return self._sequence

    @property
    def frame(self):
        return self._frame

    @property
    def raw(self):
        return self._raw

    @property
    def original(self):
        return self._original

    @property
    def bg(self):
        return self._bg

    @property
    def bgs(self):
        return self._bgs

    @property
    def filtered(self):
        return self._filtered

    @property
    def mask(self):
        return self._mask

    @property
    def masked(self):
        return self._masked

    @property
    def processed(self):
        return self._processed

    @property
    def shape(self):
        return self.raw.shape

    @property
    def stats(self):
        return self._processing_stats

    @raw.setter
    def raw(self, value):
        self._raw = value


    @bg.setter
    def bg(self, value):
        self._bg = value

    @bgs.setter
    def bgs(self, value):
        self._bg = value

    @filtered.setter
    def filtered(self, value):
        self._filtered = value

    @mask.setter
    def mask(self, value):
        self._mask = value

    @masked.setter
    def masked(self, value):
        self._masked = value
