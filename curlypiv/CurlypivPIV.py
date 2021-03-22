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
import curlypiv
from curlypiv.CurlypivFile import CurlypivFile
from curlypiv.CurlypivUtils import find_substring, get_sublevel


# 2.0 define class
class CurlypivPIV(object):

    def __init__(self, testCollection, testSetup,
                 exclude=[]):

        if not isinstance(testCollection, curlypiv.CurlypivTestCollection):
            raise ValueError("Specified test collection {} is not a CurlypivTestCollection".format(testCollection))

        if not isinstance(testSetup, curlypiv.CurlypivTestSetup):
            raise ValueError("Specified test collection {} is not a CurlypivTestSetup".format(testSetup))

        self.collection = testCollection
        self.setup = testSetup

    def get_analysis_level(self, level='seq',
                            loc=None, test=None, run=None, seq=None, file=None):

        lev = self.collection

        valid_levels = ['all','loc','test','run','seq','file']

        if level not in valid_levels:
            raise ValueError("Specified analysis level {} is not one of the valid levels: {}".format(level, valid_levels))

        if level == 'file':

            if None in [loc, test, run, seq, file]:
                raise ValueError("Must specify: loc, test, run, seq, and file for file-level analysis")
            levels = [loc, test, run, seq, file]

        if level == 'seq':

            if None in [loc, test, run, seq]:
                raise ValueError("Must specify: loc, test, run, and seq for seq-level analysis")
            levels = [loc, test, run, seq]

        if level == 'run':

            if None in [loc, test, run]:
                raise ValueError("Must specify: loc, test, and run for run-level analysis")
            levels = [loc, test, run]

        if level == 'test':

            if None in [loc, test]:
                raise ValueError("Must specify: loc, and test for test-level analysis")
            levels = [loc, test]

        if level == 'loc':

            if None in [loc]:
                raise ValueError("Must specify loc for loc-level analysis")
            levels = [loc]

        if level == 'all':
            pass
        else:
            for l in levels:
                lev = get_sublevel(lev, l)

        return lev


