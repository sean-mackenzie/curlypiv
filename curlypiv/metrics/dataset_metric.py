import csv

import numpy as np

from curlypiv.datasets import StandardDataset
from curlypiv.metrics import Metric, utils


class DatasetMetric(Metric):
    """
    Class for computing metrics based on one StandardDataset.
    """

    def __init__(self, dataset, electric_field_strength=10e3):
        """
        Notes:
            Characteristic length scale: --> It would probably be smart to choose a "characteristic length scale" as the
            smallest length scale in the system: min(channel height, bpe length, channel width, etc)
        Args:
            dataset (StandardDataset): A StandardDataset.
            electric_field_strength (int): an example electric field strength to compute basic stats.
            arg2 (...): ...

        Raise:
            TypeError: 'dataset' must be a :obj: '~curlypiv.datasets.StandardDataset' type.

        Examples
            >>> from curlypiv.datasets.squires_dataset import SquiresDataset
            >>> squires = SquiresDataset()
            >>> dm = DatasetMetric(dataset=squires)
        """
        if not isinstance(dataset, StandardDataset):
            raise TypeError("'dataset' should be a StandardDataset")

        # sets self.dataset
        super(DatasetMetric, self).__init__(dataset)

        # set example variables: electric field strength and frequency
        self.electric_field_strength = electric_field_strength

        # calculate DatasetMetrics

        # individual quantities
        self.lamb = self.calc_lamb()

        # characteristic values
        self.characteristic = self.calc_total_characteristic()

        # geometries and their relationships
        self.geometry = self.calc_total_geometry()

        # timescales
        self.timescale = self.calc_total_timescale()

        # transport phenomena (mass, charge, heat, etc...)
        self.transport = self.calc_total_transport_properties()

        # resistance
        self.resistance = self.calc_total_resistance()

        # current / current densities / charge transport
        self.current = self.calc_total_current()

        # slip velocities
        self.u_slip = self.calc_total_u_slip()

        self.calc_export_to_csv()


    # ---------- CHARACTERISTIC ----------------

    def calc_characteristic_length_scale(self, characteristic_L=None):
        """
        The characteristic length scale is the smallest meaningful (defining) length scale. In microfluidic systems,
        this is generally the channel height.
        units: (m)
        """
        if characteristic_L is None:
            characteristic_length_scale = self.dataset.H
        else:
            if characteristic_L > self.dataset.H:
                characteristic_length_scale = self.dataset.H
            else:
                characteristic_length_scale = characteristic_L

        return characteristic_length_scale

    def calc_characteristic_velocity(self, characteristic_u=None):
        """
        Calculate the characteristic velocity (rate of mass transport) in the dataset. If none is given, then take the
        u_slip velocity because we are working with electrokinetic systems.
        units: (m/s)
        """
        if characteristic_u is None:
            characteristic_u = self.calc_u_eof()
        else:
            characteristic_u = characteristic_u

        return characteristic_u

    def calc_total_characteristic(self):
        """
        Calculate all of the characteristic quantities for the dataset.
        Outputs:
            characteristic length scale (m): the smallest meaningful length scale of the system. Usually channel height.
        """
        length_scale = self.calc_characteristic_length_scale()
        velocity_scale = self.calc_characteristic_velocity()

        total_characteristic = {
            "length_scale": length_scale,
            "u": velocity_scale,
        }

        return total_characteristic

    # ------ GEOMETRIC / CONFINEMENT ----------

    def calc_confinement_channel(self):
        """
        Channel confinement describes the ratio of the channel height to the channel length
        units:
        Notes:
            Olsen discusses the effect of confinement on charge transfer to/from the BPE.
        Inputs:
            H = channel height
            L = channel length
        Returns:
            channel confinement = H / L
        """
        confinement_channel = self.dataset.H / self.dataset.L

        return confinement_channel

    def calc_total_geometry(self):
        """
        Calculate all of the geometric length scales and ratios for the dataset.
        Outputs:
            channel_confinement (): the ratio of the channel height to length.
        """
        channel_confinement = self.calc_confinement_channel()

        total_geometry = {
            "channel_confinement": channel_confinement
        }

        return total_geometry

    # ---------- MATERIAL PROPERTIES ----------------

    def calc_reynolds_number(self):
        """
        Reynolds number
        """
        Re = np.round(self.dataset.rho * self.characteristic["u"] * self.characteristic["length_scale"] / self.dataset.mu, 3)

        return Re

    def calc_conductivity_electrolyte(self):
        """
        Calculates the expected conductivity for a symmetric monovalent electrolyte
        Inputs:
            D (m^2/s):      mass diffusivity
            n_0 (#/m^3)     number density of electrolyte: n_0 = Na * c where c is concentration (mol/m^3 ~ mmol)
        """
        est_sigma = 2 * self.dataset.z ** 2 * self.dataset.e ** 2 * self.dataset.Na * self.dataset.c * \
                    self.dataset.D / (self.dataset.kb * self.dataset.T)

        return est_sigma

    def calc_total_transport_properties(self):
        """
        Calculate all of the transport related properties (phenomena) including some mass, charge, heat, etc...
        Outputs:
            --- mass ---
            Re (): Reynolds Number - defines the ratio of inertial to viscous effects.
            --- charge ---
            est_sigma (S/m): calculated conductivity of the buffer electrolyte.
        """
        # mass
        Re = self.calc_reynolds_number()
        # charge
        est_sigma = self.calc_conductivity_electrolyte()

        total_transport_properties = {
            "Re": Re,
            "est_sigma": est_sigma
        }

        return total_transport_properties

    # ---------- TIME SCALES ----------------

    def calc_timescale_L_characteristic_by_bulk_diffusion(self, L_characteristic=None):
        """
        Calculates the time for mass to diffuse a characteristic length, L
        Inputs:
            L (m):      characteristic length scale
            D (m^2/s):  mass diffusivity (most usually taken as 2e-9 (m^2/s) in microfluidics.
        Returns:
            tau_bulk_diffusion (s):
        """
        if L_characteristic is None:
            tau_bulk_diffusion = self.characteristic["length_scale"] ** 2 / self.dataset.D
        else:
            tau_bulk_diffusion = L_characteristic ** 2 / self.dataset.D

        return tau_bulk_diffusion

    def calc_total_timescale(self):
        """
        Calculates the relevant timescales for the dataset.
        Outputs:
            tau_bulk_diffusion (s): the time required to diffusive transport a distance 'l_characteristic'.
        """
        tau_bulk_diffusion = self.calc_timescale_L_characteristic_by_bulk_diffusion()

        total_timescale = {
            "tau_bulk_diffusion": tau_bulk_diffusion
        }

        return total_timescale

    # ---------- CURRENT ----------------

    def calc_channel_current(self, E=None):
        """
        Calculates the expected channel current.
        Inputs:
            Electric field strength (V/m):
            Cross-sectional area (m**2): channel width * channel height
            Buffer conductivity (S/m):
        Returns:
            Channel current (A): E * sigma * A
        """
        # channel current
        if E is None:
            est_channel_current = self.electric_field_strength * self.dataset.sigma * self.dataset.H * self.dataset.W
        else:
            est_channel_current = E * self.dataset.sigma * self.dataset.H * self.dataset.W

        return est_channel_current

    def calc_total_current(self):
        """
        Calculates the relevant current magnitudes (and eventually current density and charge transport) in dataset.
        Outputs:
            est_channel_current (A): the estimated channel current given an applied field and conductance.
        """
        est_channel_current = self.calc_channel_current()

        total_current = {
            "est_channel_current": est_channel_current
        }

        return total_current

    # ---------- RESISTANCE ----------------

    def calc_channel_resistance(self, E=None):
        """
        Calculates the expected channel current.
        Inputs:
            Electric field strength (V/m):
            Cross-sectional area (m**2): channel width * channel height
            Buffer conductivity (S/m):
        Returns:
            Channel current (A): E * sigma * A
        """
        # channel resistance
        est_channel_resistance = 1 / self.dataset.sigma * self.dataset.L / (self.dataset.H * self.dataset.W)

        return est_channel_resistance

    def calc_total_resistance(self):
        """
        Calculates the relevant resistances in the dataset.
        Outputs:
            est_channel_resistance (Ohms): the estimated channel resistance due geometry and buffer conductivity.
        """
        est_channel_resistance = self.calc_channel_resistance()

        total_resistance = {
            "est_channel_resistance": est_channel_resistance
        }

        return total_resistance

    # ---------- SLIP VELOCITIES ----------------

    def calc_u_eof(self, electric_field_strength=None):
        """
        Helmholtz-Smoluchowski / characteristic EOF velocity
        units: (m/s)
        """
        if electric_field_strength is None:
            u_eof = -self.dataset.eps_fluid * self.dataset.eps_0 * self.dataset.zeta_wall * \
                    self.electric_field_strength / self.dataset.mu
        else:
            u_eof = -self.dataset.eps_fluid * self.dataset.eps_0 * self.dataset.zeta_wall * \
                    electric_field_strength / self.dataset.mu

        return u_eof

    def calc_total_u_slip(self):
        """
        Calculates the relevant slip velocities for the electrokinetic system.
        Outputs:
            u_eof (m/s): the Helmholtz-Smoluchowski slip velocity for an applied field and small zeta potential (<25 mV)
        """
        u_eof = self.calc_u_eof()

        total_u_slip = {
            "u_eof": u_eof
        }

        return total_u_slip

    # ---------- INDIVIDUAL QUANTITIES ----------------
    # in some cases, we leave the quantities to be defined directly as attributes for easier access.

    def calc_lamb(self):
        """
        Debye length (m) for symmetric monovalent electrolyte
        """
        lamb = np.sqrt(self.dataset.eps_fluid * self.dataset.eps_0 * self.dataset.kb * self.dataset.T / (
                2 * (self.dataset.z ** 2 * self.dataset.Na * self.dataset.c) * self.dataset.e ** 2))

        return lamb

    # ---------- EXPORT DATASET METRICS ----------------
    # Package all of the dataset values and metrics into a DataFrame

    def calc_export_to_csv(self, savename='dataset_metric'):
        """
        Package all dataset values and metrics into an export-ready DataFrame
        """
        # Dataset values
        with open('/Users/mackenzie/PythonProjects/curlypiv-master/curlypiv/data/results/'+savename+'.csv', 'w') as f:
            for attribute, value in self.__dict__.items():

                if isinstance(value, StandardDataset):
                    for attr, val in value.__dict__.items():
                        if isinstance(val, np.ndarray):
                            pass
                        else:
                            f.write("%s,%s\n" % (attr, val))
                else:
                    if isinstance(value, dict):
                        for atr, vl in value.items():
                            if isinstance(vl, np.ndarray):
                                pass
                            else:
                                f.write("%s,%s\n" % (atr, vl))
                    else:
                        f.write("%s,%s\n" % (attribute, value))