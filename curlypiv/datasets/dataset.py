from abc import ABC, abstractmethod
import copy


class Dataset(ABC):
    """
    Abstract base class for datasets

    # ----- ----- ----- ----- PHYSICAL CONSTANTS ----- ----- ----- ----- -----
        these ACTIVE variables are used to reference physical constants

    # physical constants
    eps_0 = -8.854e-12  # (F/m) permittivity of free space
    e = -1.602e-19      # (C) charge of an electron
    kb = 1.3806e-23     # (J/K) Boltzmann constant
    Na = 6.022e23       # (1/mol) Avogadro's number
    R = 8.1345          # (J/K mol) Ideal gas constant
    F = 96500           # (C/mol) Faraday's constant - charge and number of moles of electrons
    """

    @abstractmethod
    def __init__(self, **kwargs):
        self.metadata = kwargs.pop('metadata', dict()) or dict()
        self.metadata.update({
            'transformer': '{}.__init__'.format(type(self).__name__),
            'params': kwargs,
            'previous': []
        })

        # define PHYSICAL CONSTANTS
        self.eps_0 = 8.854e-12  # (F/m)       permittivity of free space
        self.e = -1.602e-19  # (C)         charge of an electron
        self.kb = 1.3806e-23  # (J/K)       Boltzmann constant
        self.Na = 6.022e23  # (1/mol)     Avogadro's number
        self.R = 8.1345  # (J/K mol) Ideal gas constant
        self.F = self.e * self.Na  # (Coulombs/mol) Faraday's constant

        self.validate_dataset()

    def validate_dataset(self):
        """
        Error checking and type validation.
        """
        pass

    def copy(self, deepcopy=False):
        """
        Convenience method to return a copy of this dataset

        Args:
            deepcopy (bool, optional): :func:'~copy.deepcopy' this dataset if 'True', shallow copy otherwise.

        Returns:
            Dataset: A new dataset with fields copied from this object and metadata set accordingly
        """
        cpy = copy.deepcopy(self) if deepcopy else copy.copy(self)
        # preserve any user-created fields
        cpy.metadata = cpy.metadata.copy()
        cpy.metadata.update({
            'transformer': '{}.copy'.format(type(self).__name__),
            'params': {'deepcopy': deepcopy},
            'previous': [self]
        })
        return cpy

    @abstractmethod
    def export_dataset(self):
        """Save this Dataset to disk."""
        raise NotImplementedError

    @abstractmethod
    def split(self, num_or_size_splits, shuffle=False):
        """Split this dataset into multiple partitions.

        Args:
            num_or_size_splits (array or int): If `num_or_size_splits` is an
                int, *k*, the value is the number of equal-sized folds to make
                (if *k* does not evenly divide the dataset these folds are
                approximately equal-sized). If `num_or_size_splits` is an array
                of type int, the values are taken as the indices at which to
                split the dataset. If the values are floats (< 1.), they are
                considered to be fractional proportions of the dataset at which
                to split.
            shuffle (bool, optional): Randomly shuffle the dataset before
                splitting.

        Returns:
            list(Dataset): Splits. Contains *k* or `len(num_or_size_splits) + 1`
            datasets depending on `num_or_size_splits`.
        """
        raise NotImplementedError