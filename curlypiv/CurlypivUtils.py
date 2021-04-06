# CurlypivUtils
"""
Notes about program
"""

# 1.0 import modules
import glob, os, os.path
import re
import numpy as np
from skimage import io


def find_testcollection(dirsearch, filetype,
                        testid=('E', 'Vmm'), runid=('run', 'num'), seqid=('test_', '_X'), frameid='_X'):
    # initialize list for files
    filelist = []

    for dirpath, dirnames, filenames in os.walk(dirsearch):

        # get list of all subdirectories
        if len(dirnames) > 0:
            subdirs = dirnames
            subdirs.sort(key=lambda subdirs: find_substring(string=subdirs, leadingstring=testid[0],
                                                            trailingstring=testid[1], dtype=float, magnitude=True),
                         reverse=False)

        # get list of all files
        filesub = []
        for filename in [f for f in filenames if f.endswith(filetype)]:
            filesub.append(os.path.join(dirpath, filename))

        if len(filesub) > 1:
            filesub.sort(key=lambda filesub: find_substring(string=filesub, leadingstring=seqid[0],
                                                            trailingstring=seqid[1], dtype=int, magnitude=False,
                                                            leadingsecond=frameid, trailingsecond=filetype))
            filelist.append(filesub)

    print("Sub directories: " + str(subdirs))
    print("File list shape: " + str(np.shape(filelist)))

    return (filelist)


# def create_imagecollections(filelist):

def find_substring(string, leadingstring='', trailingstring='', dtype=float, magnitude=False,
                   leadingsecond=None,trailingsecond=None):

    ids = [[leadingstring, trailingstring],[leadingsecond, trailingsecond]]

    keys = []
    for index, k in enumerate(ids):

        # need to add a check if the string is empty
        if k == [None, None]:
            break
        if dtype == int:
            key = int(re.search(k[0] + '(.*)' + k[1], string).group(1))
        else:
            key = float(re.search(k[0] + '(.*)' + k[1], string).group(1))
            if magnitude == True and index == 0:
                key_magnitude_sorter = np.sqrt(key ** 2) - key / 1000
                key = key_magnitude_sorter
        keys.append(key)

    return (keys)


def size_files(filelist):
    # initialize size list
    sizes = []

    # flatten the filelist
    filelist = [i for sub in filelist for i in sub]

    # loop through file list
    for i in filelist:
        img = io.imread(i)
        img_size = np.shape(img)
        if img_size not in sizes: sizes.append(img_size)

    print("Image sizes in file list: " + str(sizes))
    return (sizes)


def get_sublevel(fileCollection, key):

    sub = fileCollection.get_sublevel(key)

    return sub

def round_to_odd(f):
    return int(np.round(f) // 2 * 2 + 1)