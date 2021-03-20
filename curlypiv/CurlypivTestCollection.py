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
import glob

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
from curlypiv.CurlypivFile import CurlypivFile
from curlypiv.CurlypivUtils import find_substring


# 2.0 define class
class CurlypivTestCollection(object):

    def __init__(self, collectionName, dirpath, file_type,
                 process=None, electric_field_strengths=None, frequencies=None,
                 runs=None, seqs=None, tests=None,
                 backgroundsub_specs=None, processing_specs=None, thresholding_specs=None, cropping_specs=None,
                 dir_tests='tests',dir_bg='background',dir_results='results',
                 locid = 'loc', testid = ('E', 'Vmm'), runid = ('run', 'num'), seqid = ('test_', '_X'), frameid = '_X',
                 exclude=[]):
        super(CurlypivTestCollection, self).__init__()
        if not isdir(dirpath):
            raise ValueError("Specified folder {} does not exist".format(dirpath))

        self._collectionName = collectionName
        self.dirpath = dirpath
        self.dir_tests = dir_tests
        self._path = join(self.dirpath,self.dir_tests)
        self.dir_bg = dir_bg
        self.dir_results = dir_results

        self.file_type = file_type
        self.locid = locid
        self.testid = testid
        self.runid = runid
        self.seqid = seqid
        self.frameid = frameid

        self._find_locs(exclude=exclude)
        self._add_locs()
        self._get_size()

        self.processing_specs = processing_specs
        self.thresholding_specs = thresholding_specs
        self.cropping_specs = cropping_specs

    def __len__(self):
        return self._size

    def __getitem__(self, item):
        key = list(self.files.keys())[item]
        return self.files[key]

    def __repr__(self):
        class_ = 'CurlypivFileCollection'
        repr_dict = {'Name': self._collectionName,
                     'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'Number of files': len(self),
                     'Cropping': self.cropping_specs,
                     'Preprocessing': self.processing_specs}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _find_locs(self, exclude=[]):
        all_locs = listdir(join(self.path))
        all_locs.sort()
        save_locs = []
        for lc in [l for l in all_locs if l.startswith(self.locid)]:
            if lc in exclude:
                continue
            save_locs.append((lc))

        if len(save_locs) < 1:
            raise ValueError("No locs found in /{} with {} loc id".format(self.dir_tests, self.locid))

        self.loclist = save_locs


    def _add_locs(self):
        locs = OrderedDict()
        for lc in [l for l in self.loclist if l.startswith(self.locid)]:
            loc = CurlypivLocation(join(self.path,lc), file_type=self.file_type,
                                testid=self.testid,runid=self.runid,seqid=self.seqid,frameid=self.frameid)
            locs.update({loc._locname: loc})
            logger.warning('Loaded loc{}'.format(loc._locname))

        if len(locs) < 1:
            raise ValueError("No locs found in {} with ...{} loc id".format(self.dir_tests, self.locid))
        self.locs = locs

    #def get_files(self, loc=None, test=None, run=None, sequence=None, file=None):
        # method to retrieve all files in the specified container

    def filter_images(self):
        for image in self.tests.values():
            image.filter_image

    def _get_size(self):
        size = 0
        for key, values in self.locs.items(): size += values.size
        self._size = size

    @property
    def name(self):
        return self._collectionName

    @property
    def path(self):
        return self._path

    @property
    def size(self):
        return self._size

    @property
    def collectionname(self):
        return self._collectionName


# 2.0 define class
class CurlypivLocation(object):

    def __init__(self, dirpath, file_type,
                 locid = 'loc', testid = ('E', 'Vmm'), runid = ('run', 'num'), seqid = ('test_', '_X'), frameid = '_X',
                 exclude=[]):

        self.dirpath = dirpath
        self.file_type = file_type
        self.locid = locid
        self.testid = testid
        self.runid = runid
        self.seqid = seqid
        self.frameid = frameid

        self._locname = find_substring(string=self.dirpath, leadingstring=self.locid, dtype=int)[0]

        self._find_tests()
        self._add_tests()
        self._get_size()

    def __repr__(self):
        class_ = 'CurlypivLocation'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'Test list': self.test_list,
                     'Total number of images': self._size}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _find_tests(self, exclude=[]):
        """
        Identifies all tests in a location folder
        :param exclude:
        :return:
        """
        # step 1: find all files in directory and add to list
        all_tests = listdir(self.dirpath)
        save_tests = []
        for test in [t for t in all_tests if t.endswith(self.testid[1])]:
            if test in exclude:
                continue
            save_tests.append(test)

        if len(save_tests) < 1:
            raise ValueError("No runs found in {} with run...{} format".format(self.dirpath, self.testid[1]))

        # step 2: sort files based on sequence id and frame id
        save_tests.sort(key=lambda save_tests: find_substring(string=save_tests, leadingstring=self.testid[0],
                                                        trailingstring=self.testid[1], dtype=float,
                                                        magnitude=True))
        # save all the files of the right filetype in this attribute
        self.test_list = save_tests

        logger.warning(
            "Found {} runs in directory {}".format(len(self.test_list), self.dirpath))


    def _add_tests(self):
        tests = OrderedDict()
        for tst in [t for t in self.test_list if t.endswith(self.testid[1])]:
            test = CurlypivTest(join(self.dirpath,tst), file_type=self.file_type,
                                testid=self.testid,runid=self.runid,seqid=self.seqid,frameid=self.frameid)
            tests.update({test._testname: test})
            logger.warning('Loaded test {}'.format(test._testname))

        if len(tests) < 1:
            raise ValueError("No tests found in test {} with ...{} test id".format(self.dir_tests, self.testid))
        self.tests = tests

    def _get_size(self):
        size = 0
        for key, values in self.tests.items(): size += values.size
        self._size = size

    @property
    def name(self):
        return self._locname

    @property
    def path(self):
        return self.dirpath

    @property
    def size(self):
        return self._size

    @property
    def locname(self):
        return self._locname


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

        self._testname = find_substring(string=self.dirpath, leadingstring=self.testid[0], trailingstring=self.testid[1],
                                       dtype=float)[0]

        self._find_runs()
        self._add_runs()
        self._get_size()

    def __len__(self):
        return len(self.run_list)

    def __repr__(self):
        class_ = 'CurlypivTest'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'Run list': self.run_list,
                     'Total number of images': self._size}
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

    def _get_size(self):
        size = 0
        for key, values in self.runs.items(): size += values.size
        self._size = size

    @property
    def name(self):
        return self._testname

    @property
    def path(self):
        return self.dirpath

    @property
    def testname(self):
        return self._testname

    @property
    def size(self):
        return self._size



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
        self._find_seqpaths()
        self._add_seqs()
        #self._add_files()
        self._get_size()

    def __len__(self):
        return self.files

    def __repr__(self):
        class_ = 'CurlypivRun'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'SequenceList': self.seqs,
                     'Number of files': self._size}
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
        self.seq_list = seq_list

        logger.warning(
            "Found {} files across {} sequences of type {} in directory {}".format(len(save_files), len(self.seq_list),
                                                                                   self.file_type, self.dirpath))

    def _find_seqpaths(self):
        seq_files = []
        for seq_uniq in self.seq_list:
            seeqing = str(self.seqid[0])+str(seq_uniq)+str(self.seqid[1])
            files = []
            for file in [f for f in self.filepaths if f.find(seeqing)!=-1]:
                files.append(file)
            seq_files.append([seq_uniq, files])
        self._seqpaths = (seq_files)

    def _add_seqs(self):
        seqs = OrderedDict()
        for s in self._seqpaths:
            seq = CurlypivSequence(self.dirpath,file_type=self.file_type, seqname=s[0],
                                   filelist=s[1],frameid = '_X')
            seqs.update({seq._seqname: seq})
            logger.warning('Loaded sequence {}'.format(seq._seqname))
        self.seqs = seqs


    def _add_files(self):
        files = OrderedDict()
        for f in self.filepaths:
            file = CurlypivFile(join(self.dirpath,f), img_type=self.file_type)
            files.update({file.filename: file})
            logger.warning('Loaded image {}'.format(file.filename))
        self.files = files

    def _get_size(self):
        self._size = len(self.filepaths)

    @property
    def name(self):
        return self._runname

    @property
    def path(self):
        return self.dirpath

    @property
    def size(self):
        return self._size


# 2.0 define class
class CurlypivSequence(object):

    def __init__(self, dirpath, file_type, seqname, filelist,
                 frameid = '_X',
                 exclude=[]):

        self.dirpath = dirpath
        self.file_type = file_type
        self.frameid = frameid
        self._seqname = seqname
        self.file_list = filelist

        self._add_files()
        self._get_size()

        self._add_files()

    def _add_files(self):
        files = OrderedDict()
        for f in self.file_list:
            file = CurlypivFile(join(self.dirpath,f), img_type=self.file_type)
            files.update({file.filename: file})
            logger.warning('Loaded image {}'.format(file.filename))
        self.files = files

    def _get_size(self):
        self._size = len(self.file_list)

    @property
    def name(self):
        return self._seqname

    @property
    def path(self):
        return self.dirpath

    @property
    def size(self):
        return self._size
