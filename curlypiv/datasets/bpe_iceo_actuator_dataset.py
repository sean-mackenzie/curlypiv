from logging import warning

import numpy as np
import pandas as pd

from curlypiv.datasets.dataset import Dataset
from curlypiv.datasets.standard_dataset import StandardDataset
from curlypiv.datasets.bpe_iceo_dataset import BpeIceoDataset


class BpeIceoActuatorDataset(BpeIceoDataset):
    """
    Base class for all standard datasets for BPE-ICEO experiments
    """

    def __init__(self, membrane, **kwargs):
        """
        Args:
            membrane            (material solid class):     class with material and geometric information.
            **kwargs (StandardDataset class):               StandardDataset arguments
        """
        # STORAGE VARIABLES
        # membrane
        self.membrane_l = membrane.length
        self.membrane_w = membrane.width
        self.membrane_h = membrane.thickness
        self.membrane_mat = membrane.name
        self.membrane_youngs_modulus = membrane.youngs_modulus
        self.membrane_density = membrane.density
        self.membrane_poissons_ratio = membrane.poissons_ratio
        self.membrane_conductivity = membrane.conductivity
        self.membrane_permittivity = membrane.permittivity # relative permittivity
        self.membrane_zeta = membrane.zeta
        self.membrane_dielectric_strength = membrane.dielectric_strength

        super(BpeIceoActuatorDataset, self).__init__(**kwargs)

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