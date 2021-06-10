from logging import warning

import numpy as np
import pandas as pd

from curlypiv.datasets import Dataset


class StandardDataset(Dataset):
    """
    Base class for all structured datasets.

    A StructuredDataset requires data to be stored in :obj:'numpy.ndarray' objects with :obj:'~numpy.dtype' as :obj:'~numpy.float64'.

    Attributes:

    # ----- ----- ----- ----- -----  STORAGE VARIABLES ----- ----- ----- ----- -----
    these variables are used to keep historical data about experimental parameters

    # channel
    channel_l:          length of channel           (m)             CurlypivTestSetup.chip...
    channel_w:          width of channel            (m)             CurlypivTestSetup.chip.
    channel_h:          height of channel           (m)             CurlypivTestSetup.chip.
    channel_mat_bw:     bottom wall material        ()
    channel_mat_tw:     bottom wall material        ()

    # bpe
    bpe_l:              length of channel           (m)             CurlypivTestSetup.chip.bpe
    bpe_w:              width of channel            (m)             CurlypivTestSetup.chip.bpe
    bpe_h:              height of channel           (m)             CurlypivTestSetup.chip.bpe
    bpe_mat:            material of BPE             ()
    bpe_adh_h:          height of adhesion layer    (m)
    bpe_adh_mat:        adhesion material           ()

    # dielectric coating
    dielectric_h:       height of dielectric layer  (m)
    dielectric_mat:     dielectric material         ()
    dielectric_perm:    relative permittivity       ()

    # electrolyte / buffer
    buffer_mat:         buffer composition          ()
    buffer_cond:        buffer conductivity         (S/m)
    buffer_pH:          buffer pH                   ()
    buffer_perm:        relative permittivity       ()

    # microscope
    scope_type:         type of microscope used     ()
    scope_objective:    microscope objective        ()
    scope_mag:          magnification               ()
    scope_na:           numerical aperture          ()
    scope_um_per_pix:   microns per pixel           ()

    # ccd
    ccd_type:           type of ccd used            ()
    ccd_img_acq_type:   ccd or emccd                ()
    ccd_exp_time:       exposure time               (s)
    ccd_img_acq_rate:   image acquisition rate      (Hz)
    ccd_gain:           emccd gain                  ()

    # fluorescent particles
    fp_type:            name of particles           ()
    fp_diameter:        diameter of particles       (m)
    fp_concentration:   concentration in buffer     (?)
    fp_mobility:        ep mobility of particles    (um m / V s)

    # ----- ----- ----- ----- CALCULATION VARIABLES ----- ----- ----- ----- -----
            these ACTIVE variables are used to calculate ICEO stats
        (many are repeated from above in short script for readability)

    # physical constants
    eps_fluid:      permittivity of water           (F/m2)          CurlypivTestSetup.chip.material_fluid.permittivity
    eps_dielectric: permittivity of sio2            ()              CurlypivTestSetup.chip.bpe.dielectric_coating.permittivity
    T:              temperature                     (K)             CurlypivTestSetup.chip.material_fluid.temperature

    # material properties
    rho:            density                         (kg/m3)         depends on the instance
    mu:             dynamic viscosity               (m2/s)          CurlypivTestSetup.chip.material_fluid.viscosity
    sigma:          electrical conductivity         (S/m)           CurlypivTestSetup.chip.material_fluid.conductivity
    z:              electrolyte valence             ()              CurlypivTestSetup.chip.material_fluid.valence
    zeta:           zeta potential                  (V)             depends on the instance
    Ns:             surface site density            (#/nm2)         CurlypivTestSetup.chip.material_fluid.reaction_site_density
    Ka:             reaction equilibrium constant   ()              CurlypivTestSetup.chip.material_fluid.Ka
    a_h:            bulk concentration of protons   (mmols)         (I think this is just pH) CurlypivTestSetup.chip.material_fluid.pH

    # geometries
    l:              characteristic length scale     (m)             CurlypivTestSetup.chip.channel.height
    l_bpe:          length of bpe                   (m)             CurlypivTestSetup.chip.bpe.length
    d:              thickness of sio2 dielectric    (m)             CurlypivTestSetup.chip.bpe.dielectric_coating.thickness

    # experimental inputs
    x:              location                        (m)             * need to write * array of locations along BPE length for instanteous induced zeta calc.
    t:              time                            (s)             * need to write * array of times in a periodic cycle for instanteous zeta calc.
    f:              frequency                       (1/s)           * need to write * CurlypivTestCollection.locs.tests.test_id[1]
    E:              electric field strength         (V/m)           * need to write * CurlypivTestCollection.locs.tests.test_id[0]

    metadata (dict): Details about the creation of this dataset. For
    example::

        {
            'transformer': 'Dataset.__init__',
            'params': kwargs,
            'previous': None
        }
    """

    def __init__(self, channel=None, buffer=None, microscope=None, ccd=None, fluorescent_particles=None, metadata=None):
        """
        Args:
            channel: channel class
            buffer: material fluid class
            microscope: micrscope class
            ccd: ccd class
            fluorescent_particles: fluorescent particles class
            metadata (optional): additional metadata to append
        """

        # define STORAGE VARIABLES

        # channel
        self.channel_l = channel.length                              #     length of channel           (m)             CurlypivTestSetup.chip...
        self.channel_w = channel.width                               #     width of channel            (m)             CurlypivTestSetup.chip.
        self.channel_h = channel.height                              #     height of channel           (m)             CurlypivTestSetup.chip.
        self.channel_mat_bw = channel.material_bottom_wall_surface.name   #     bottom wall material        ()
        self.channel_mat_tw = channel.material_top_wall_surface.name      #     bottom wall material        ()

        # electrolyte / buffer
        self.buffer_mat = buffer.species                             #   buffer composition          ()
        self.buffer_conc = buffer.concentration                      #   buffer concentration        (mmol)
        self.buffer_cond = buffer.conductivity                       #   buffer conductivity         (S/m)
        self.buffer_pH = buffer.pH                                   #   buffer pH                   ()
        self.buffer_perm = buffer.permittivity                       #   relative permittivity       ()
        self.buffer_visc = buffer.viscosity                          #   viscosity                   ()
        self.buffer_temp = buffer.temperature                        #   temperature                 ()

        # microscope
        self.scope_type = microscope.type                            #   type of microscope used     ()
        self.scope_objective = microscope.objective.name             #   objective used              ()
        self.scope_mag = microscope.objective.magnification          #   objective magnification     ()
        self.scope_na = microscope.objective.numerical_aperture      #   objective NA                ()
        self.scope_um_per_pix = microscope.objective.pixel_to_micron #   objective microns per pixel ()

        # ccd
        self.ccd_type = ccd.name                                     #   type of ccd camera used     ()
        self.ccd_img_acq_type = ccd.img_acq_type                     #   image acquisition type      ()
        self.ccd_exp_time = ccd.exposure_time                        #   exposure time               (s)
        self.ccd_img_acq_rate = ccd.img_acq_rate                     #   image acquisition rate      (Hz)
        self.ccd_gain = ccd.em_gain                                  #   emccd gain                  ()

        # fluorescent particles
        self.fp_type = fluorescent_particles.name                    #   name of particles           ()
        self.fp_diameter = fluorescent_particles.diameter            #   diameter of particles       (m)
        self.fp_concentration = fluorescent_particles.concentration  #   concentration in buffer     (?)
        self.fp_mobility = fluorescent_particles.electrophoretic_mobility#   ep mobility of particles    (um m / V s)


        # define CALCULATION VARIABLES

        # mechanical
        self.mu = channel.material_fluid.viscosity  # fluid viscosity
        self.rho = channel.material_fluid.density  # fluid density
        self.T = channel.material_fluid.temperature  # standard temperature
        self.D = channel.material_fluid.diffusivity  # mass diffusivity

        # electro/chemical
        self.c = channel.material_fluid.concentration
        self.sigma = channel.material_fluid.conductivity
        self.pH = channel.material_fluid.pH
        self.c_H = 10 ** -self.pH * 1e3  # (mmol) = (mmol/L) = (mol/m3); (concentration of Hydrogen ions (H+)
        self.eps_fluid = channel.material_fluid.permittivity
        self.zeta_wall = (channel.material_bottom_wall_surface.zeta + channel.material_top_wall_surface.zeta) / 2
        self.z = channel.material_fluid.valence

        # geometry
        self.L = channel.length
        self.W = channel.width
        self.H = channel.height

        # imaging
        self.dt = 1 / ccd.img_acq_rate  # (s) time between images
        self.p_d = fluorescent_particles.diameter
        self.microns_to_pixels = 1 / microscope.objective.pixel_to_micron

        # sets metadata
        super(StandardDataset, self).__init__(metadata=metadata)


    def __repr__(self):
        # return repr(self.metadata)
        return str(self)

    def validate_dataset(self):
        """
        Error checking and type validation

        Raise:
            TypeError: certain fields must be ...
        """
        super(StandardDataset, self).validate_dataset()

        # ==== TYPE CHECKING ====
        # code for checking the data type and correcting

        # ==== SHAPE CHECKING ====

    def export_dataset(self, export_metadata=False):
        """
        Export the dataset and supporting attributes
        TODO: the preferred file format is HDF? XML? CSV?
        """

        if export_metadata:
            raise NotImplementedError("The option to export metadata has not been implemented yet")

        return None

    def import_dataset(self, import_metadata=False):
        """
        Import the dataset and supporting attributes
        TODO: The preferred file format is HDF? XML? CSV?
        """

        if import_metadata:
            raise NotImplementedError("The option to import metadata has not been implemented yet")
        return None

    def split(self, num_or_size_splits, shuffle=False, seed=None):
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
            seed (int or array_like): Takes the same argument as
                :func:`numpy.random.seed()`.

        Returns:
            list: Splits. Contains *k* or `len(num_or_size_splits) + 1`
            datasets depending on `num_or_size_splits`.
        """
        return None