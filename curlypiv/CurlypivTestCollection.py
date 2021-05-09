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
import cv2 as cv

# Curlypiv
from curlypiv.CurlypivFile import CurlypivFile
from curlypiv.CurlypivUtils import find_substring
from curlypiv.CurlypivImageProcessing import analyze_img_quality, apply_flatfield_correction, apply_darkfield_correction, apply_background_subtractor


# 2.0 define class
class CurlypivTestCollection(object):

    def __init__(self, collectionName, dirpath, file_type,
                 process=None, electric_field_strengths=None, frequencies=None,
                 runs=None, seqs=None, tests=None,
                 bpe_specs=None, backsub_specs=None, processing_specs=None, thresholding_specs=None, cropping_specs=None, resizing_specs=None,
                 dir_tests='tests', dir_bg='background', dir_results='results',
                 locid = 'loc', testid = ('E', 'Vmm'), runid = ('run', 'num'), seqid = ('test_', '_X'), frameid = '_X',
                 load_files=False, exclude=[],
                 calibration_grid_path=None, calibration_illum_path=None, calibration_camnoise_path=None,
                 ):
        super(CurlypivTestCollection, self).__init__()
        if not isdir(dirpath):
            raise ValueError("Specified folder {} does not exist".format(dirpath))

        self._collectionName = collectionName
        self.dirpath = dirpath
        self.dir_tests = dir_tests
        self._path = join(self.dirpath, self.dir_tests)
        self.dir_bg = dir_bg
        self.dir_results = dir_results

        self.file_type = file_type
        self.locid = locid
        self.testid = testid
        self.runid = runid
        self.seqid = seqid
        self.frameid = frameid

        self.load_files = load_files

        self._find_locs(exclude=exclude)
        self._add_locs()
        self._get_size()

        self.bpe_specs = bpe_specs
        self.cropping_specs = cropping_specs
        self.resizing_specs = resizing_specs
        self.backsub_specs = backsub_specs
        self.processing_specs = processing_specs
        self.thresholding_specs = thresholding_specs


        # data structure dependent files
        if calibration_grid_path is None:
            self.grid_path = join(self.dirpath, 'setup/calibration/microgrid')
        if calibration_illum_path is None:
            self.illum_path = join(self.dirpath, 'setup/calibration/illumination')
        else:
            self.illum_path = join(self.dirpath, 'setup/calibration/illumination', calibration_illum_path)
        if calibration_camnoise_path is None:
            self.camnoise_path = join(self.dirpath, 'setup/calibration/cameraNoise')


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
                     'Test collection identifier': self.dir_tests,
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
                                testid=self.testid,runid=self.runid,seqid=self.seqid,frameid=self.frameid,
                                   load_files=self.load_files)
            locs.update({loc._locname: loc})
            logger.warning('Loaded loc{}'.format(loc._locname))

        if len(locs) < 1:
            raise ValueError("No locs found in {} with ...{} loc id".format(self.dir_tests, self.locid))
        self.locs = locs

    #def get_files(self, loc=None, test=None, run=None, sequence=None, file=None):
        # method to retrieve all files in the specified container

    def add_img_testset(self, loc, test, run, seq, level='seq'):
        lev = self
        if level == 'seq':
            levels = [loc, test, run, seq]
        for l in levels:
            lev = lev.get_sublevel(l)

        self.img_testset = lev


    def filter_images(self):
        for image in self.tests.values():
            image.filter_image

    def _get_size(self):
        size = 0
        for key, values in self.locs.items(): size += values.size
        self._size = size

    def get_sublevel(self, key):

        sub = self.locs

        for k, v in sub.items():
            if k == key:
                item = v

        return item

    def get_sublevel_all(self):
        sub = self.locs
        all_subs = []

        for k, v in sub.items():
            all_subs.append(v)

        return all_subs

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
                 load_files=False, exclude=[]):

        self.dirpath = dirpath
        self.file_type = file_type
        self.locid = locid
        self.testid = testid
        self.runid = runid
        self.seqid = seqid
        self.frameid = frameid

        self.load_files = load_files

        self._locname = find_substring(string=self.dirpath, leadingstring=self.locid, dtype=int)[0]

        self._find_tests()
        self._add_tests()
        self._get_size()

    def __repr__(self):
        class_ = 'CurlypivLocation'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'Location identifier': self._locname,
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
        # step 0: asses the length of the test_id
        test_ids = len(self.testid) // 2

        # step 1: find all files in directory and add to list
        all_tests = listdir(self.dirpath)
        save_tests = []
        for test in [t for t in all_tests if t.endswith(self.testid[-1])]:
            if test in exclude:
                continue
            save_tests.append(test)

        if len(save_tests) < 1:
            raise ValueError("No runs found in {} with run...{} format".format(self.dirpath, self.testid[-1]))

        # step 2: sort files based on sequence id and frame id
        if test_ids > 1:
            save_tests.sort(key=lambda save_tests: find_substring(string=save_tests, leadingstring=self.testid[0],
                                                        trailingstring=self.testid[1], leadingsecond=self.testid[2],
                                                        trailingsecond=self.testid[3], dtype=float, magnitude=False))
        else:
            save_tests.sort(key=lambda save_tests: find_substring(string=save_tests, leadingstring=self.testid[0],
                                                        trailingstring=self.testid[1], dtype=float, magnitude=True))

        # save all the files of the right filetype in this attribute
        self.test_list = save_tests

        logger.warning(
            "Found {} runs in directory {}".format(len(self.test_list), self.dirpath))


    def _add_tests(self):
        tests = OrderedDict()
        for tst in [t for t in self.test_list if t.endswith(self.testid[-1])]:
            test = CurlypivTest(join(self.dirpath,tst), file_type=self.file_type,
                                testid=self.testid, runid=self.runid, seqid=self.seqid, frameid=self.frameid,
                                load_files=self.load_files)
            j = 1
            tests.update({test._testname: test})
            logger.warning('Loaded test {}'.format(test._testname))

        if len(tests) < 1:
            raise ValueError("No tests found in test {} with ...{} test id".format(self.dir_tests, self.testid))
        self.tests = tests

    def _get_size(self):
        size = 0
        for key, values in self.tests.items(): size += values.size
        self._size = size

    def get_sublevel(self, key):

        sub = self.tests

        for k, v in sub.items():
            if k == key:
                item = v

        return item

    def get_sublevel_all(self):
        sub = self.tests
        all_subs = []

        for k, v in sub.items():
            all_subs.append(v)

        return all_subs


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
                 load_files=False, exclude=[]):

        self.dirpath = dirpath
        self.file_type = file_type
        self.testid = testid
        self.runid = runid
        self.seqid = seqid
        self.frameid = frameid

        self.load_files = load_files

        # step 0: asses the length of the test_id
        test_ids = len(self.testid) // 2

        # step 1: assign the test name as a tuple (E,) or (E, f)
        if test_ids == 1:
            self._testname = (find_substring(string=self.dirpath, leadingstring=self.testid[0], trailingstring=self.testid[1],
                                       dtype=float)[0],)
            self._E = self._testname[0]
        else:
            testname = find_substring(string=self.dirpath, leadingstring=self.testid[0], trailingstring=self.testid[1],
                                            leadingsecond=self.testid[2], trailingsecond=self.testid[3], dtype=float)
            self._testname = (testname[0], testname[1])
            self._E = self._testname[0]
            self._f = self._testname[1]

        self._find_runs()
        self._add_runs()
        self._get_size()

    def __len__(self):
        return len(self.run_list)

    def __repr__(self):
        class_ = 'CurlypivTest'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'Test identifier': self._testname,
                     'Electric field': self._E,
                     'Frequency': self._f,
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
                               runid = self.runid, seqid = self.seqid, frameid = self.frameid,
                               load_files=self.load_files)
            runs.update({file._runname: file})
            logger.warning('Loaded run {}'.format(file._runname))

        self.runs = runs

    def _get_size(self):
        size = 0
        for key, values in self.runs.items(): size += values.size
        self._size = size

    def get_sublevel(self, key):

        sub = self.runs

        for k, v in sub.items():
            if k == key:
                item = v

        return item

    def get_sublevel_all(self):
        sub = self.runs
        all_subs = []

        for k, v in sub.items():
            all_subs.append(v)

        return all_subs

    def add_piv_data(self, zeta=False, x=None, testname=None):

        if zeta:
            u_mag_bkg = []
            u_mag_mean = []
            u_mag_std = []
            u_mean = []
            v_mean = []

            for run in self.runs.values():
                u_mag_bkg.append(run.u_mag_bkg)
                u_mag_mean.append(run.u_mag_mean)
                u_mag_std.append(run.u_mag_std)
                u_mean.append(run.u_mean)
                v_mean.append(run.v_mean)

            self.u_mag_bkg = np.round(np.mean(u_mag_bkg),1)
            self.u_mag_mean = np.round(np.mean(u_mag_mean), 1)
            self.u_mag_std = np.round(np.mean(u_mag_std), 1)
            self.u_mean = np.round(np.mean(u_mean), 1)
            self.v_mean = np.round(np.mean(v_mean), 1)
        else:
            u_mean_columns = []
            u_mean_columns_std = []
            for run in self.runs.values():
                u_mean_columns.append(run.u_mean_columns)
                u_mean_columns_std.append(run.u_mean_columns_std)
            self.u_mean_x = x
            self.u_mean_columns = np.round(np.mean(run.u_mean_columns, axis=0), 1)
            self.u_mean_columns_std = np.round(np.mean(run.u_mean_columns_std, axis=0), 2)

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
                 load_files=False, exclude=[]):

        self.dirpath = dirpath
        self.file_type = file_type
        self.runid = runid
        self.seqid = seqid
        self.frameid = frameid

        self.load_files = load_files

        self._runname = find_substring(string=self.dirpath, leadingstring=self.runid[0], trailingstring=self.runid[1],
                                       dtype=int)[0]

        # first loop through file list to expand any stacked files
        self._find_filepaths()
        self._find_seqpaths()
        self._add_seqs()
        logger.warning("First loop through file list complete")
        # second loop through file list to reorganize file list
        self._find_filepaths()
        self._find_seqpaths()
        self._add_seqs()
        logger.warning("Second loop through file list complete")
        self._get_size()

    def __len__(self):
        return self.files

    def __repr__(self):
        class_ = 'CurlypivRun'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'Run identifier': self._runname,
                     'Sequence list': self.seq_list,
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
            if file in exclude or file.startswith('multifile'):
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
                                   filelist=s[1], seqid=self.seqid, frameid = self.frameid, load_files=self.load_files)

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

    def update_run_filelist(self):
        print('yah')


    def _get_size(self):
        self._size = len(self.filepaths)

    def get_sublevel(self, key):

        sub = self.seqs

        for k, v in sub.items():
            if k == key:
                item = v

        return item

    def get_sublevel_all(self):
        sub = self.seqs
        all_subs = []

        for k, v in sub.items():
            all_subs.append(v)

        return all_subs

    def add_piv_data(self, zeta=False, x=None, testname=None):

        if zeta:
            u_mag_bkg = []
            u_mag_mean = []
            u_mag_std = []
            u_mean = []
            v_mean = []

            for seq in self.seqs.values():
                if seq.name == 0:
                    u_mag_bkg.append(seq.u_mag_bkg)
                u_mag_mean.append(seq.u_mag_mean)
                u_mag_std.append(seq.u_mag_std)
                u_mean.append(seq.u_mean)
                v_mean.append(seq.v_mean)

            self.u_mag_bkg = np.round(np.mean(u_mag_bkg),1)
            self.u_mag_mean = np.round(np.mean(u_mag_mean), 1)
            self.u_mag_std = np.round(np.mean(u_mag_std), 1)
            self.u_mean = np.round(np.mean(u_mean), 1)
            self.v_mean = np.round(np.mean(v_mean), 1)

        else:
            u_mean_columns = []
            u_mean_columns_std = []
            for seq in self.seqs.values():
                u_mean_columns.append(seq.u_mean_columns)
                u_mean_columns_std.append(seq.u_mean_columns_std)
            self.u_mean_x = seq.u_mean_x
            self.u_mean_columns = np.round(np.mean(u_mean_columns, axis=0), 1)
            self.u_mean_columns_std = np.round(np.mean(u_mean_columns_std, axis=0), 2)
            self.testname = testname

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
                 load_files=False, seqid = ('test_', '_X'), frameid = '_X',
                 file_lim=600, exclude=[]):

        self.dirpath = dirpath
        self.file_type = file_type
        self.seqid = seqid
        self.frameid = frameid
        self._seqname = seqname
        self.file_lim = file_lim

        self.file_list = filelist
        self.check_for_multifiles()
        self.get_size()
        self.add_files(load_files)

    def __repr__(self):
        class_ = 'CurlypivSequence'
        repr_dict = {'Dirpath': self.dirpath,
                     'Filetype': self.file_type,
                     'Sequence identifier': self._seqname,
                     'Sequence ID': self.seqid,
                     'Frame ID': self.frameid,
                     'Number of files': self._size}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def check_for_multifiles(self, init_frame=0):
        for f in self.file_list:

            img = io.imread(join(self.dirpath,f), plugin='tifffile')

            multifile_num = 0
            if len(np.shape(img)) > 2:
                shape = np.shape(img)
                os.rename(join(self.dirpath,f),join(self.dirpath,'multifile_'+str(multifile_num)+self.file_type))
                multifile_num += 1

                imgs_in_stack = np.shape(img)[0]
                max_imgs = self.file_lim
                if imgs_in_stack > max_imgs:
                    logger.warning("{} images in stack. Keeping only {} for now.".format(imgs_in_stack, max_imgs))
                    imgs_in_stack = max_imgs

                img_num = 1
                for i in range(imgs_in_stack):
                    if i >= init_frame:
                        img_sub = img[i,:,:]
                        io.imsave(join(self.dirpath,self.seqid[0]+str(self._seqname)+self.seqid[1]+str(img_num)+self.file_type),
                                  img_sub, plugin='tifffile', check_contrast=False)
                    img_num += 1


    def add_files(self, load_file, file_lim=None):

        if file_lim is not None:
            file_limit = file_lim
        else:
            file_limit = self.file_lim

        files = OrderedDict()
        if load_file:
            file_num = 0
            for f in self.file_list:
                file = CurlypivFile(join(self.dirpath,f), img_type=self.file_type)
                files.update({file.filename: file})

                if file_num == file_limit:
                    logger.warning("Added the maximum number of files to sequence")
                    continue

                file_num += 1

        self.files = files

    def refresh_files(self):
        if len(self.files) < 1:
            self.add_files(load_file=True)

    def get_subset(self, num_files):

        # update file limit
        self.file_lim = num_files

        # get full file collection
        full_set = self.files.items()

        # get subset from 0 to file_lim by index
        subset = list(full_set)[:self.file_lim]

        # split into two lists
        subset_filenames, subset_files = map(list, zip(*subset))

        # update sequence properties
        self.files = dict(zip(subset_filenames, subset_files))
        self.file_list = self.file_list[:self.file_lim]
        self.get_size()

    def get_sublevel(self, key):

        if len(self.files) < 1:
            self.add_files(load_file=True)

        sub = self.files

        for k, v in sub.items():
            if k == key:
                item = v

        return item

    def get_sublevel_all(self):
        if len(self.files) < 1:
            self.add_files(load_file=True)

        sub = self.files
        all_subs = []

        for k, v in sub.items():
            all_subs.append(v)

        return all_subs

    def get_size(self):
        self._size = len(self.file_list)

    # ----- ----- image processing functions below ----- -----

    def get_img_quality(self):
        self.raw_mean, self.raw_std, self.raw_snr = analyze_img_quality(self.files)

    def calc_seq_mean_file(self):
        means = np.zeros_like(self.first_file.raw)
        for f in self.files:
            means += f.raw
        mean = np.divide(means, len(self.files))
        return mean

    def apply_flatfield_correction(self, flatfield, darkfield):
        apply_flatfield_correction(self.files, flatfield, darkfield)

    def apply_darkfield_correction(self, darkfield):
        apply_darkfield_correction(self.files, darkfield)

    def apply_image_processing(self, bpespecs=None, cropspecs=None, resizespecs=None, filterspecs=None, backsubspecs=None):
        for img in self.files.values():
            img.image_bpe_filter(bpespecs=bpespecs)    # mask bpe region
            img.image_crop(cropspecs=cropspecs)          # crop
            img.image_resize(resizespecs=resizespecs)    # resize
            img.image_filter(filterspecs=filterspecs, image_input='raw', image_output='filtered', force_rawdtype=True)


    def apply_background_subtractor(self, bg_method='KNN', apply_to='raw'):
        if bg_method == 'KNN':
            backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
        elif bg_method == "MOG2":
            backSub = cv.createBackgroundSubtractorMOG2(detectShadows=False)
        else:
            backSub = None

        apply_background_subtractor(self.files, backgroundSubtractor=backSub, bg_method=bg_method, apply_to=apply_to, bg_filepath=None)

    def animate_images(self, img_animate='bgs', start=0, stop=200, savePath=None):
        valid_imgs = ['bg', 'bgs', 'filtered', 'mask', 'masked', 'original', 'raw']
        if img_animate not in valid_imgs:
            raise ValueError("Animation image must be one of {}".format(valid_imgs))

        counter = start
        if stop > len(self.files):
            stop = len(self.files)

        for name, img in self.files.items():

            if counter >= stop:
                break

            if img_animate == 'bg':
                frame = img.bg
            elif img_animate == 'bgs':
                frame = img.bgs
            elif img_animate == 'filtered':
                frame = img.filtered
            elif img_animate == 'mask':
                frame = img.mask
            elif img_animate == 'masked':
                frame = img.masked
            elif img_animate == 'original':
                frame = img.original
            elif img_animate == 'raw':
                frame = img.raw


            if savePath and counter > (stop - start) / 2:
                cv.imwrite(savePath + '/{}_'.format(img_animate) + name, frame)

            cv.imshow('Background subtracted {} image'.format(img_animate), frame)
            counter += 1
            keyboard = cv.waitKey(50)
            if keyboard == 'q' or keyboard == 27:
                break

        cv.destroyAllWindows()

    def add_piv_data(self, zeta=False, u_mag_mean=None, u_mag_std=None, u_mean=None, v_mean=None, u_mag_bkg=None,
                     u_mean_x=None, u_mean_columns=None, u_mean_columns_std=None, testname=None):
        self._E = int(np.round(testname[0],0))
        if zeta:
            self.u_mag_mean = np.mean(u_mag_mean)
            self.u_mag_std = np.mean(u_mag_std)
            self.u_mean = np.mean(u_mean)
            self.v_mean = np.mean(v_mean)
            if u_mag_bkg is not None:
                self.u_mag_bkg = np.mean(u_mag_bkg)
        else:
            self.u_mean_x = u_mean_x
            self.u_mean_columns = u_mean_columns
            self.u_mean_columns_std = u_mean_columns_std


    # ----- ----- deleting files and removing data functions below ----- -----

    def remove_files(self, file='all'):
        # code for removing a single file
        # would go here

        if file == 'all':
            files = OrderedDict()

        self.files = files


    @property
    def name(self):
        return self._seqname

    @property
    def path(self):
        return self.dirpath

    @property
    def size(self):
        return self._size

    @property
    def first_file(self):
        return self.files[0][1]

    @property
    def mean_file(self):
        return self.calc_seq_mean_file()
