from logging import warning

import numpy as np
import pandas as pd

from curlypiv.datasets.dataset import Dataset
from curlypiv.datasets.standard_dataset import StandardDataset
from curlypiv.datasets.bpe_iceo_dataset import BpeIceoDataset


class BpeIceoFluoroQuenchDataset(BpeIceoDataset):
    """
    Base class for all standard datasets for BPE-ICEO experiments...
    that employ fluorescein quenching to observe the BPE charging/discharging dynamics and the Faradaic reaction rate
    """

    def __init__(self, fluorescein, test_collection, **kwargs):
        """
        Args:
            fluorescein (material liquid class):        class with material information on the fluorescein/electrolyte.
                NOTES: this material liquid class should contain a list for species and concentration:
                species (list): ['fluorescein species name', 'electrolyte species name']
                concentration (list): [conc. in mmol of fluorescein, conc. in mmol of electrolyte)
            **kwargs (StandardDataset class):           StandardDataset arguments
        """
        # STORAGE VARIABLES
        # fluorescein
        self.fluorescein_species = fluorescein.species[0]
        self.fluorescein_concentration = fluorescein.concentration[0]
        self.fluor_electrolyte_species = fluorescein.species[1]
        self.fluor_electrolyte_concentration = fluorescein.concentration[1]

        super(BpeIceoFluoroQuenchDataset, self).__init__(**kwargs)


    def validate_dataset(self):
        """
        Error checking and type validation

        Raise:
            ValueError: some condition
            ValueError: some condition
        """
        if np.all(self.L_bpe == self.bpe_l):
            pass

        super(BpeIceoDataset, self).validate_dataset()