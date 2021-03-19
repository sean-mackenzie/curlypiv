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
from curlypiv.CurlypivImage import CurlypivImage
from curlypiv.CurlypivUtils import find_substring


# 2.0 define class
class CurlypivImageCollection(object):

    def __init__(self, dirpath, file_type,
                 process=None, electric_field_strengths=None, frequencies=None,
                 runs=None, seqs=None, files=None,
                 backgroundsub_specs=None, processing_specs=None, thresholding_specs=None, cropping_specs=None,
                 dir_tests='tests',dir_bg='background',dir_results='results',
                 locid = None, testid = ('E', 'Vmm'), runid = ('run', 'num'), seqid = ('test_', '_X'), frameid = '_X',
                 exclude=[]):
        super(CurlypivImageCollection, self).__init__()
        if not isdir(dirpath):
            raise ValueError("Specified folder {} does not exist".format(dirpath))

        self.dirpath = dirpath
        self.dir_tests = dir_tests
        self.dir_bg = dir_bg
        self.dir_results = dir_results

        self.file_type = file_type
        self.locid = locid
        self.testid = testid
        self.runid = runid
        self.seqid = seqid
        self.frameid = frameid
        self._find_testCollection(exclude=exclude)
        self._add_files()

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

    def _find_testCollection(self, exclude=[]):
        # initialize list for files
        filelist = []
        filelen = 0

        # get list of all locations in tests directory
        for locpath, locnames, locs in os.walk(self.dirpath+'/'+self.dir_tests):
            if len(locnames) > 0:
                loctests = locnames

            for l in loctests:

                # get list of all electric field strengths at each test location
                for dirpath, dirnames, filenames in os.walk(self.dirpath+'/'+self.dir_tests+'/'+l):
                    if len(dirnames) > 0:
                        subdirs = dirnames
                        subdirs.sort(key=lambda subdirs: find_substring(string=subdirs, leadingstring=self.testid[0],
                                                                        trailingstring=self.testid[1], dtype=float,
                                                                        magnitude=True),
                                     reverse=False)

                    # get list of all files in each electric field strength directory
                    filesub = []
                    for filename in [f for f in filenames if f.endswith(self.file_type)]:
                        filesub.append(os.path.join(dirpath, filename))

                # sort file list
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
            "Found {} tests with filetype {} in folder {}".format(len(filelist), self.file_type,
                                                                  self.dirpath+'/'+self.dir_tests))
        # Save all the files of the right filetype in this attribute
        self.filepaths = filelist


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

# 2.0 define class
class CurlypivTest(object):

    def __init__(self, dirpath, file_type,
                 testid = ('E', 'Vmm'), runid = ('run', 'num'), seqid = ('test_', '_X'), frameid = '_X',
                 exclude=[]):

        self.dirpath = dirpath
        self.file_type = file_type
        self.testid = testid
        self.runid = runid
        self.seqid = seqid
        self.frameid = frameid

        self._find_runs()
        self._add_runs()
        self._files()

    def __len__(self):
        return len(self.run_list)

    def __repr__(self):
        class_ = 'CurlypivTest'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'Run list': self.run_list,
                     'Total number of images': self.files}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _find_runs(self, exclude=[]):
        """
        Identifies all runs in a test folder
        :param exclude:
        :return:
        """
        # step 1: find all files in directory and add to list
        all_runs = listdir(self.dirpath)
        save_runs = []
        for run in [r for r in all_runs if r.endswith(self.runid[1])]:
            if run in exclude:
                continue
            save_runs.append(run)

        if len(save_runs) < 1:
            raise ValueError("No runs found in {} with run...{} format".format(self.dirpath, self.runid[1]))

        # step 2: sort files based on sequence id and frame id
        save_runs.sort(key=lambda save_runs: find_substring(string=save_runs, leadingstring=self.runid[0],
                                                        trailingstring=self.runid[1], dtype=int,
                                                        magnitude=False))
        # save all the files of the right filetype in this attribute
        self.run_list = save_runs

        logger.warning(
            "Found {} runs in directory {}".format(len(self.run_list), self.dirpath))

    def _add_runs(self):
        runs = OrderedDict()
        for f in self.run_list:
            file = CurlypivRun(join(self.dirpath,f), file_type=self.file_type,
                               runid = self.runid, seqid = self.seqid, frameid = self.frameid)
            runs.update({file._runname: file})
            logger.warning('Loaded run {}'.format(file._runname))

        self.runs = runs

    def _files(self):
        numfiles_key = ['NumberOfFiles']
        files = 0

        for key, values in self.runs.items():
            files += len(values.filepaths)

        self.files = files










# 2.0 define class
class CurlypivRun(object):

    def __init__(self, dirpath, file_type,
                 runid = ('run', 'num'), seqid = ('test_', '_X'), frameid = '_X',
                 exclude=[]):

        self.dirpath = dirpath
        self.file_type = file_type
        self.runid = runid
        self.seqid = seqid
        self.frameid = frameid

        self._runname = find_substring(string=self.dirpath, leadingstring=self.runid[0], trailingstring=self.runid[1],
                                       dtype=int)[0]

        self._find_filepaths()
        self._add_files()

    def __len__(self):
        return self.files

    def __repr__(self):
        class_ = 'CurlypivRun'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'SequenceList': self.seqs,
                     'NumberOfFiles': len(self.files)}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _find_filepaths(self, exclude=[]):
        """
        Identifies all files of filetype in folder
        :param exclude:
        :return:
        """
        # step 1: find all files in directory and add to list
        all_files = listdir(self.dirpath)
        save_files = []
        for file in [f for f in all_files if f.endswith(self.file_type)]:
            if file in exclude:
                continue
            save_files.append(file)

        if len(save_files) < 1:
            raise ValueError("No files found in {}".format(self.dirpath))

        # step 2: sort files based on sequence id and frame id
        save_files.sort(key=lambda save_files: find_substring(string=save_files, leadingstring=self.seqid[0],
                                                        trailingstring=self.seqid[1], dtype=int,
                                                        magnitude=False,
                                                        leadingsecond=self.frameid,
                                                        trailingsecond=self.file_type))
        # save all the files of the right filetype in this attribute
        self.filepaths = save_files

        # step 3: identify the sequences
        seq_list = []
        for f in self.filepaths:
            seq = find_substring(string=f, leadingstring=self.seqid[0], trailingstring=self.seqid[1],
                             dtype=int)[0]
            if seq not in seq_list: seq_list.append(seq)
        self.seqs = seq_list

        logger.warning(
            "Found {} files across {} sequences of type {} in directory {}".format(len(save_files), len(self.seqs),
                                                                                   self.file_type, self.dirpath))

    def _add_files(self):
        files = OrderedDict()
        for f in self.filepaths:
            file = CurlypivImage(join(self.dirpath,f), img_type=self.file_type)
            files.update({file.filename: file})
            logger.warning('Loaded image {}'.format(file.filename))
        self.files = files







# 2.0 define class
class CurlypivSequence(object):

    def __init__(self, dirpath, file_type,
                 seqid = ('test_', '_X'), frameid = '_X',
                 exclude=[]):

        self.dirpath = dirpath
        self.file_type = file_type
        self.seqid = seqid
        self.frameid = frameid
        self.seqs = None
        self.files = None

    def __len__(self):
        return self.files

    def __repr__(self):
        class_ = 'CurlypivRun'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'Number of files': len(self)}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str