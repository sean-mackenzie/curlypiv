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
from skimage import io

# Curlypiv
from curlypiv.CurlypivUtils import find_substring
from curlypiv import CurlypivImageProcessing



# 2.0 define class

class CurlypivImage(object):
    """
    This class holds an image along with its properties such as:
    raw image, filtered, image, path, filename, and statistics.
    """

    def __init__(self, path, img_type, ):
        super(CurlypivImage, self).__init__()

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
        # image is not cropped, then the ._original attribute is left as None.
        self._original = None   # store original image
        self._bg = None         # background image
        self._bgs = None        # store background subtracted image
        self._filtered = None
        self._processing_stats = None
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

    def crop_image(self, cropspecs):
        """
        This crops the image.
        The argument cropsize is a dictionary of coordinates with (0,0) as the bottom left corner of the image.
        :param cropspecs:
        :return:
        """

        valid_crops = ['xmin','xmax','ymin','ymax']

        if cropspecs == None:
            pass
        else:
            for crop_func in cropspecs.keys():
                if crop_func not in valid_crops:
                    raise ValueError("{} is not a valid crop dimension. Use: {}".format(crop_func, valid_crops))

            self._original  = self._raw.copy()
            self._raw = self._raw[cropspecs['ymin']:cropspecs['ymax'], cropspecs['xmin']:cropspecs['xmax']]

    def subtract_background(self, bg_method, bg_filepath):
        """
        This subtracts a background input image from the signal input image.
        :param bg_method:
        :param bg_img:
        :return:
        """

        (self._bg, self._bgs) = CurlypivImageProcessing.subtract_background(self._raw, bg_filepath, bg_method='min')

    def filter_image(self, filterspecs, force_rawdtype=True):
        """
        This is an image filtering function. The argument filterdict are similar to the arguments of the...
        e.g. filterdict: {'median': 5, 'gaussian':3}
        :param filterspecs:
        :param force_rawdtype:
        :return:

        This method should self-assign self._processed and self._processing_stats)
        """
        raw_dtype = self._raw.dtype

        if self._bgs is not None:
            img_filtered = self._bgs.copy()
        else:
            img_filtered = self._raw.copy()

        img_filtered = CurlypivImageProcessing.filter_image(img_filtered, filterspecs=filterspecs)

        if force_rawdtype and img_filtered.dtype != raw_dtype:
            img_filtered = img_filtered.astype(raw_dtype)

        self._filtered = img_filtered

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

    def find_particles(self,min_sigma=0.5, max_sigma=5,num_sigma=20, threshold=0.1,overlap=0.85):
        """
        This uses Laplacian of Gaussians method to determine the number and size of particles in the image.
        :return: 
        """

        if self._filtered is not None:
            img = self._filtered.copy()
            print("Using filtered image for particle identification")
        else:
            img = self._raw.copy()

        # particle identification
        particles = CurlypivImageProcessing.find_particles(img,min_sigma=min_sigma, max_sigma=max_sigma,
                                                           num_sigma=num_sigma, threshold=threshold, overlap=overlap)

        # stats
        num_particles = len(particles)
        particle_density = np.sum(np.square(particles[:,2])*np.pi, axis=0)/np.size(img)*100

        print("{} particles identified".format(num_particles))

        if particle_density >= 100:
            raise ValueError("Particle density should not be > 100%. Need to reduce overlap or increase threshold")

        self._update_processing_stats(['num_particles','particle_density'],[num_particles,particle_density])

        return(particles)


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
    def bg(self):
        return self._bg

    @property
    def bgs(self):
        return self._bgs

    @property
    def filtered(self):
        return self._filtered

    # @property
    # def masked(self):
    #    return self._masked

    @property
    def original(self):
        return self._original

    @property
    def processed(self):
        return self._filtered

    @property
    def raw(self):
        return self._raw

    @property
    def shape(self):
        return self.raw.shape

    @property
    def stats(self):
        return self._processing_stats
