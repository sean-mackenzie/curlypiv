# test CurlypivImageCollection
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import os
from os import listdir
from os.path import isfile, basename, join, isdir
from collections import OrderedDict

# quality control and debugging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Maths/Scientifics
import numpy as np
import pandas as pd

# Image Processing
from skimage import io

# Curlypiv
from curlypiv import CurlypivImage
from curlypiv.CurlypivUtils import find_substring


# 2.0 define class
class CurlypivImageCollection(object):

    def __init__(self, dirpath, file_type,
                 process=None, electric_field_strengths=None, frequencies=None,
                 runs=None, seqs=None, files=None,
                 backgroundsub_specs=None, processing_specs=None, thresholding_specs=None, cropping_specs=None,
                 dir_tests='tests',dir_bg='background',dir_results='results',
                 testid = ('E', 'Vmm'), runid = ('run', 'num'), seqid = ('test_', '_X'), frameid = '_X',
                 exclude=[]):
        super(CurlypivImageCollection, self).__init__()
        if not isdir(dirpath):
            raise ValueError("Specified folder {} does not exist".format(dirpath))

        self.dirpath = dirpath
        self.dir_tests = dir_tests
        self.dir_bg = dir_bg
        self.dir_results = dir_results

        self.file_type = file_type
        self.testid = testid
        self.runid = runid
        self.seqid = seqid
        self.frameid = frameid
        self._find_testcollection(exclude=exclude)
        #self._add_files()

        self.processing_specs = processing_specs
        self.thresholding_specs = thresholding_specs
        self.cropping_specs = cropping_specs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        key = list(self.files.keys())[item]
        return self.files[key]

    def __repr__(self):
        class_ = 'CurlypivFileCollection'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'Number of files': len(self),
                     'Cropping': self.cropping_specs,
                     'Preprocessing': self.processing_specs}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _find_testcollection(self, exclude=[]):
        # initialize list for files
        filelist = []
        filelen = 0

        for dirpath, dirnames, filenames in os.walk(self.dirpath+'/'+self.dir_tests):

            # get list of all subdirectories
            if len(dirnames) > 0:
                subdirs = dirnames
                subdirs.sort(key=lambda subdirs: find_substring(string=subdirs, leadingstring=self.testid[0],
                                                                trailingstring=self.testid[1], dtype=float,
                                                                magnitude=True),
                             reverse=False)

            # get list of all files
            filesub = []
            for filename in [f for f in filenames if f.endswith(self.file_type)]:
                filesub.append(os.path.join(dirpath, filename))

            if len(filesub) > 1:
                filesub.sort(key=lambda filesub: find_substring(string=filesub, leadingstring=self.seqid[0],
                                                                trailingstring=self.seqid[1], dtype=int,
                                                                magnitude=False,
                                                                leadingsecond=self.frameid,
                                                                trailingsecond=self.file_type))
                filelen += len(filesub)
                filelist.append(filesub)

        print("Sub directories: " + str(subdirs))
        print("File list shape: " + str(np.shape(filelist)))
        print("Test collection size: " + str(filelen))

        logger.warning(
            "Found {} files with filetype {} in folder {}".format(len(filelist), self.file_type,
                                                                  self.dirpath+'/'+self.dir_tests))
        # Save all the files of the right filetype in this attribute
        self.filepaths = filelist


    def _find_filepaths(self, exclude=[]):
        """
        Identifies all files of filetype in folder
        :param exclude:
        :return:
        """
        all_files = listdir(self.dirpath)
        save_files = []
        for file in all_files:
            if file.endswith(self.file_type):
                if file in exclude:
                    continue
                save_files.append(file)

        logger.warning(
            "Found {} files of type {} in directory {}".format(len(save_files), self.file_type, self.dirpath))
        # save all the files of the right filetype in this attribute
        self.filepaths = save_files

    def _add_files(self):
        files = OrderedDict()
        for f in self.filepaths:
            file = CurlypivImage(join(self.dirpath, f))
            files.update({file.filename: file})
            logger.warning('Loaded image {}'.format(file.filename))
        self.files = files

    def filter_images(self):
        for image in self.files.values():
            image.filter_image
