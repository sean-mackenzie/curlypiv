import os

import pandas as pd

from curlypiv.utils.read_XML import read_Squires_XML
from curlypiv.datasets import BpeIceoDataset


def import_squires_dataset(return_rows=1, save=False):
    filter_squires = {
        "electric_fields": [5000, 25000],
        "frequencys": [100, 5000],
        "dielectrics": ['SiO2'],
        "buffers": ['KCl'],
        "d_thick": [0, 50],
        "b_conduct": [10, 50],
    }
    df = read_Squires_XML(return_rows=return_rows, write_to_disk=save, kwargs=filter_squires)
    return df


# prepare squires dataset
SQUIRES_SAMPLED = import_squires_dataset(return_rows=1, save=True)

# create CurlypivTestSetup classes from sampled Squires data

BPE = None
DIELECTRIC_COATING = None
STANDARD_DATASET = {
    "CHANNEL": None,
    "BUFFER": None,
    "MICROSCOPE": None,
    "CCD": None,
    "FLUORESCENT_PARTICLES": None,
    "METADATA": None
}

class SquiresDataset(BpeIceoDataset):
    """
    Pascall & Squires (2010) dataset:

    See: https://pubs.rsc.org/en/content/articlelanding/2010/LC/c004926c#!divAbstract
    """
    def __init__(self, bpe=BPE, dielectric_coating=DIELECTRIC_COATING):

        super(SquiresDataset, self).__init__(bpe=BPE, dielectric_coating=DIELECTRIC_COATING, kwargs=STANDARD_DATASET)