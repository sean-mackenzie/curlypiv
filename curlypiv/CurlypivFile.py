# test CurlypivImage
"""
Notes about program
"""

# 1.0 import modules
# data I/O
from os.path import isfile, basename
import copy

# Maths/Scientifics
import numpy as np
import numpy.ma as ma
import pandas as pd

# Image Processing
import cv2 as cv
import imutils
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.measure import find_contours, approximate_polygon, points_in_poly
from skimage import data, filters, measure, morphology

# plotting
import matplotlib.pyplot as plt

# Curlypiv
from curlypiv.CurlypivUtils import find_substring
from curlypiv.CurlypivImageProcessing import img_resize, img_subtract_background, img_filter, img_apply_bpe_filter, img_find_particles
from curlypiv.CurlypivPlotting import plot_image_process



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
        self.original = None
        self._original = None
        self.raw = None
        self.load(path)

        # Crop the raw image to a new size. This sets the ._original attribute to store the original image file. If the
        # image is not cropped or resized, then the ._original attribute is left as None.
        self.bg = pre_calc_bg  # background image
        self.bgs = None        # store background subtracted image
        self.filtered = None
        self.processing_stats = None
        self.bpe_mask = None
        self.mask = pre_calc_mask
        self.masked = None

    def __repr__(self):
        class_ = 'CurlypivImage'
        repr_dict = {'Dimensions': self.shape,
                     'Sequence': self.sequence,
                     'Frame': self.frame}

        out_str = "{}, {} \n".format(class_, self._filename)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def load(self, path):
        img = io.imread(self.filepath, plugin='tifffile')

        if len(np.shape(img)) > 2:
            raise ValueError("Image is three-dimensional. Need to investigate more.")

        self.original = img.copy()
        self._original = img.copy()
        self.raw = img.copy()

    def update_processing_stats(self, names, values):
        if not isinstance(names, list):
            names = [names]
        new_stats = {}

        for name, value in zip(names, values):
            new_stats.update({name: [value]})
        new_stats = pd.DataFrame(new_stats)

        if self.processing_stats is None:
            self.processing_stats = new_stats
        else:
            self.processing_stats = new_stats.combine_first(self.processing_stats)

    def image_bpe_filter(self, bpespecs=None):
        """
        Filter the BPE specific region
        """
        if bpespecs is not None and self.original is None:
            self.original = self.raw.copy()

        #self.raw, self.bpe_mask = img_apply_bpe_filter(self, bpespecs=bpespecs)
        ###
        img_original = copy.copy(self.raw)   # copy.deepcopy(img)
        img = copy.copy(self.raw)            # copy.deepcopy(img)

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

            if isinstance(img, CurlypivFile):  # TODO: FIX THIS?
                img_bpe_masked = np.rint(img.raw)
            elif isinstance(img, np.ndarray):
                img_bpe_masked = np.rint(copy.copy(img))
            else:
                raise ValueError("Need to check what type the input image array is.")

            img_bpe_masked[~bpe_mask] = 0

            # filter bpe region
            raw_masked = ma.array(img.copy(), mask=~bpe_mask)
            raw_masked = raw_masked * bpespecs['multiplier']

            # store mask and update raw image
            img_bpe_mask = bpe_mask
            img = raw_masked.data

            self.raw, self.bpe_mask = img, img_bpe_mask
            #return img, img_bpe_mask


    def image_crop(self, cropspecs, show_crop_plot=False):
        """
        This crops the image.
        The argument cropsize is a dictionary of coordinates with (0,0) as the bottom left corner of the image.

        NOTES:
            np.shape() returns (y-dim, x-dim)
            self.shape() returns (y-dim, x-dim)

        :param cropspecs:
        :return:
        """

        if cropspecs is None:
            pass
        else:
            valid_crops = ['xmin', 'xmax', 'ymin', 'ymax'] # example = [20, 500, 0, 400]

            for crop_func in cropspecs.keys():
                if crop_func not in valid_crops:
                    raise ValueError("{} is not a valid crop dimension. Use: {}".format(crop_func, valid_crops))

            if cropspecs['xmax'] > self.shape[1]:
                cropspecs['xmax'] = self.shape[1]

            if cropspecs['ymax'] > self.shape[0]:
                cropspecs['ymax'] = self.shape[0]

            ymin = self.shape[0] - cropspecs['ymax']
            ymax = self.shape[0] - cropspecs['ymin']

            if self.original is None:
                self.original = self.raw.copy()

            self.raw = self.raw[ymin:ymax, cropspecs['xmin']:cropspecs['xmax']] # TODO: fix cropping mistakes - the x and y axes are getting mixed.

            if self.bpe_mask is not None:   # TODO: fix when bpe_masking does and doesn't occur
                self.bpe_mask = self.bpe_mask[ymin:ymax, cropspecs['xmin']:cropspecs['xmax']]

            if show_crop_plot:
                # plot to confirm cropping
                fig, axes = plt.subplots(ncols=2, figsize=(10,6))
                ax = axes.ravel()
                ax[0].imshow(self.original, cmap='viridis')
                ax[0].set_title('Original')
                ax[1].imshow(self.raw, cmap='viridis')
                ax[1].imshow(self.bpe_mask, cmap='Reds', alpha=0.5)
                ax[1].set_title('Raw + BPE mask')
                plt.show()

    def image_resize(self, resizespecs=None):

        if resizespecs is not None:
            if self.original is None:
                self.original = self.raw.copy()

            self.raw = img_resize(self.raw, method=resizespecs['method'], scale=resizespecs['scale'])

    def image_subtract_background(self, image_input='raw', backgroundSubtractor=None, bg_method="KNN", bg_filepath=None, plot_background_subtraction=False):
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
            input = self.raw.copy()

        elif image_input == 'filtered':
            if self.filtered is None:
                ValueError("This file has no filtered image")
            input = self.filtered.copy()

        # perform background subtraction
        (self.bg, self.bgs, self.mask, self.masked) = img_subtract_background(input, backgroundSubtractor=backgroundSubtractor, bg_filepath=bg_filepath, bg_method=bg_method, plot_background_subtraction=plot_background_subtraction)

        # apply background subtraction to the input image as well
        if bg_method in ['min', 'mean']:
            if self.original is None:
                self.original = self.raw
            self.raw = self.bgs


    def image_filter(self, filterspecs, image_input='raw', image_output='filtered', force_rawdtype=True):
        """
        This is an image filtering function. The argument filterdict are similar to the arguments of the...
        e.g. filterdict: {'median': 5, 'gaussian':3}
        :param filterspecs:
        :param force_rawdtype:
        :return:

        This method should self-assign self._processing_stats)
        """
        raw_dtype = self.raw.dtype

        valid_images = ['raw', 'bgs', 'filtered', 'mask', 'masked', 'bg']

        if image_input == 'bg':
            user_confirmed_image = input("Are you sure you want to filter the background image? Enter 'y' or other image in {}".format(valid_images))
            if user_confirmed_image != 'y':
                image = user_confirmed_image

        if image_input not in valid_images:
            raise ValueError("{} not a valid image for filtering. Use: {}".format(image_input, valid_images))
        elif image_input == 'raw':
            input = self.raw.copy()
        elif image_input == 'bgs':
            if self.bgs is None:
                ValueError("This file has no background subtracted image (bgs)")
            input = self.bgs.copy()
        elif image_input == 'filtered':
            if self.filtered is None:
                ValueError("This file has no filtered image")
            input = self.filtered.copy()
        elif image_input == 'mask':
            if self.mask is None:
                ValueError("This file has no mask")
            input = self.mask.copy()
        elif image_input == 'masked':
            if self.masked is None:
                ValueError("This file has no masked image")
            input = self.masked.copy()
        elif image_input == 'bg':
            if self.bg is None:
                ValueError("This file has no background image")
            input = self.bg.copy()
        else:
            ValueError("A matching image input from {} was not found".format(valid_images))

        # perform filtering
        output = img_filter(input, filterspecs=filterspecs)

        if force_rawdtype and output.dtype != raw_dtype:
            output = output.astype(raw_dtype)

        valid_outputs = ['bgs', 'filtered', 'masked', 'bg']
        if image_output not in valid_outputs:
            raise ValueError("{} not a valid image for output. Use: {}".format(image_output, valid_outputs))

        if image_output == 'bgs':
            self.bgs = output
        elif image_output == 'filtered':
            self.filtered = output
        elif image_output == 'mask':
            self.mask = output
        elif image_output == 'masked':
            self.masked = output
        elif image_output == 'bg':
            self.bg = output

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

    def apply_flatfield_correction_to_img(self, darkfield, flatfield=None):

        if np.shape(self.raw) != np.shape(darkfield):
            darkfield = np.mean(darkfield)

        if flatfield is None:
            img_corrected = self.raw - darkfield
        else:
            vmin, vmax = np.percentile(self.raw, (0, 100))
            img_corrected = (self.raw - darkfield) * np.mean((flatfield - darkfield)) / (flatfield - darkfield)
            img_corrected = np.asarray(rescale_intensity(img_corrected, in_range='image', out_range=(0, vmax)), dtype='uint16')

        self.raw = img_corrected

    def calculate_stats(self):
        """
        This calculates some basic image statistics and updates the processing stats
        :return:
        """
        raw_mean = self.raw.mean()
        raw_std = self.raw.std()

        # calculate approximate signal to noise ratio
        # background and signal values
        bkg, sig = np.percentile(self.raw, (50, 99.5))
        # masks
        bkg_mask = self.raw < bkg
        sig_mask = self.raw > sig
        # mask the raw image
        ma_bkg = ma.masked_array(self.raw, mask=bkg_mask)
        ma_sig = ma.masked_array(self.raw, mask=sig_mask)
        # compute the approximate signal to noise ratio
        raw_snr = ma_sig.mean() / ma_bkg.std()

        if self.filtered is not None:
            # pixel intensities
            filt_mean = self.filtered.mean()
            filt_std = self.filtered.std()
        else:
            filt_mean = None
            filt_std = None

        self.update_processing_stats(['raw_mean','raw_std','raw_snr', 'filt_mean','filt_std'],
                                      [raw_mean, raw_std, raw_snr, filt_mean, filt_std])


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
            input = self.raw.copy()
        elif image_input == 'bgs':
            if self.bgs is None:
                ValueError("This file has no background subtracted image (bgs)")
            input = self.bgs.copy()
        elif image_input == 'filtered':
            if self.filtered is None:
                ValueError("This file has no filtered image")
            input = self.filtered.copy()
        elif image_input == 'mask':
            if self.mask is None:
                ValueError("This file has no mask")
            input = self.filtered.copy()
        elif image_input == 'masked':
            if self.masked is None:
                ValueError("This file has no masked image")
            input = self.filtered.copy()
        else:
            ValueError("A matching image input from {} was not found".format(valid_images))

        # particle identification
        particles = img_find_particles(img=input, min_sigma=min_sigma, max_sigma=max_sigma,
                                                               num_sigma=num_sigma, threshold=threshold, overlap=overlap)

        # stats
        num_particles = len(particles)
        particle_density = np.sum(np.square(particles[:,2])*np.pi, axis=0)/np.size(input)*100

        if particle_density >= 100:
            raise ValueError("Particle density should not be > 100%. Need to reduce overlap or increase threshold")

        self.update_processing_stats(['num_particles','particle_density'],[num_particles,particle_density])

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
    def filename(self):
        return self._filename

    @property
    def frame(self):
        return self._frame

    @property
    def filepath(self):
        return self._filepath

    @property
    def path(self):
        return self._filepath

    @property
    def filetype(self):
        return self._filetype

    @property
    def shape(self):
        return self.raw.shape

    @property
    def stats(self):
        return self.processing_stats

    @property
    def sequence(self):
        return self._sequence