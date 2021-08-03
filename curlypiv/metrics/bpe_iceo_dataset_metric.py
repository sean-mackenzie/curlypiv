from logging import warning

import numpy as np
import pandas as pd

from curlypiv.datasets.dataset import Dataset
from curlypiv.datasets.standard_dataset import StandardDataset
from curlypiv.datasets.bpe_iceo_dataset import BpeIceoDataset
from curlypiv.metrics import DatasetMetric, utils


class BpeIceoDatasetMetric(DatasetMetric):
    """
    Class for computing BPE-ICEO related metrics on a single :obj: '~curlypiv.datasets.BpeIceoDataset'
    """

    def __init__(self, dataset, electric_field_strength=10e3, frequency=100, waveform='square'):
        """
        Args:
            dataset (StandardDataset): A StandardDataset.
            electric_field_strength (int): an example electric field strength to compute basic stats.
            arg2 (...): ...

        Raise:
            TypeError: 'dataset' must be a :obj: '~curlypiv.datasets.StandardDataset' type.
        """
        if not isinstance(dataset, BpeIceoDataset):
            raise TypeError("'dataset' should be a BpeIceoDataset")

        # sets the self.dataset and frequency
        super(BpeIceoDatasetMetric, self).__init__(dataset, electric_field_strength=electric_field_strength)
        self.frequency = frequency
        self.waveform = waveform

        # basic quantities (directly accessible)
        self.w = self.calc_angular_frequency()
        self.zeta = self.calc_zeta_induced()

        # calculate BPE-ICEO stats
        self.geometry.update(self.calc_total_geometry_bpe())
        self.characteristic.update(self.calc_total_characteristic_bpe())
        self.capacitance = self.calc_total_capacitance_bpe()
        self.current.update(self.calc_total_current_bpe())
        self.resistance.update(self.calc_total_resistance_bpe())
        self.timescale.update(self.calc_total_timescale_bpe())
        self.zeta_induced = self.calc_total_zeta_induced()
        self.u_slip.update(self.calc_total_induced_charge_u_slip())

        # export dataset to csv
        self.calc_export_to_csv(savename='bpe_iceo_dataset_metric')

    def __repr__(self):
        class_ = 'BpeIceoDatasetMetric'
        repr_dict = {
            'Geometry': self.geometry,
            'Characteristic': self.characteristic,
            'Capacitance': self.capacitance,
            'Current': self.current,
            'Resistance': self.resistance,
            'Timescale': self.timescale,
            'Induced Zeta': self.zeta_induced,
            'Slip Velocity': self.u_slip
        }
        """
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str
        """
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            if type(val) is list:
                print('list')
            elif type(val) is dict:
                for keyy, vall in val.items():
                    if type(vall) is np.ndarray:
                        pass
                    else:
                        out_str += '{}: {} \n'.format(keyy, str(vall))
            else:
                out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    # ------------- BASICS -----------------

    # ---------- CHARACTERISTIC ---------------

    def calc_total_characteristic_bpe(self):
        """
        Calculate all of the characteristic quantities for the dataset.
        Outputs:
            characteristic length scale (m): the smallest meaningful length scale of the system. Usually channel height.
        """
        length_scale = self.calc_characteristic_length_scale(characteristic_L=self.dataset.L_bpe)

        total_characteristic_bpe = {
            "length_scale": length_scale,
        }

        return total_characteristic_bpe


    # ------ GEOMETRIC / CONFINEMENT ----------

    def calc_confinement_bpe(self):
        """
        BPE confinement describes the ratio of the channel height to the BPE length
        units:
        Notes:
            Olsen discusses the effect of confinement on charge transfer to/from the BPE.
        Inputs:
            H = channel height
            L_bpe = BPE length
        Returns:
            BPE confinement = H / L_bpe
        """
        confinement_bpe = self.dataset.H / self.dataset.L_bpe
        return confinement_bpe

    def calc_length_ratio_bpe_to_channel(self):
        """
        BPE length ratio describes the ratio of the channel length to the BPE length
        units:
        Notes:
            Alex Eden says this plays a significant role in observed BPE-ICEO phenomena like pressure generation.
        Inputs:
            L = channel length
            L_bpe = BPE length
        Returns:
            BPE confinement = H / L_bpe
        """
        length_ratio_bpe_to_channel = self.dataset.L_bpe / self.dataset.L
        return length_ratio_bpe_to_channel

    def calc_total_geometry_bpe(self):
        """
        Calculate the relevant length scales for the BPE system.
        """
        confinement_bpe = self.calc_confinement_bpe()
        length_ratio_bpe_to_channel = self.calc_length_ratio_bpe_to_channel()

        total_geometry_bpe = {
            "bpe_confinement": confinement_bpe,
            "length_ratio_bpe_to_channel": length_ratio_bpe_to_channel
        }

        return total_geometry_bpe

    # ---------- CAPACITANCES ----------------

    def calc_linear_doublelayer_capacitance(self):
        """
        Capacitance due to the linearized electric double layer
        units: F/m^2
        Notes:
            Squires, 2010   -   "linearized electric double layer capacitance"
            Adjari, 2006    -   "Debye layer capacitance (per unit area) at low voltage"
        Inputs:
            eps     =   permittivity of the fluid   (F/m)
            lamb    =   linearized Debye length     (m)
        """
        Cdl_linear = self.dataset.eps_fluid * self.dataset.eps_0 / self.lamb

        return Cdl_linear

    def calc_nonlinear_doublelayer_capacitance(self, zeta=None):
        """
        Nonlinear electric double layer capacitance         (when zeta > thermal voltage)
        units: F/m^2
        Notes:
            Squires, 2010   -   "   "
        Inputs:
            eps     =   permittivity of fluid       (F/m)
            lamb    =   Debye length                (m)
            zeta    =   zeta potential              (V)
        """
        if zeta is not None:
            Cdl_nonlinear = (self.dataset.eps_fluid * self.dataset.eps_0 / self.lamb) * np.cosh(zeta / 2)
        else:
            Cdl_nonlinear = (self.dataset.eps_fluid * self.dataset.eps_0 / self.lamb) * np.cosh(self.zeta / 2)

        return Cdl_nonlinear

    def calc_nonlinear_differential_capacitance(self):
        """
        Nonlinear differential capacitance
        units: F/m^2
        Notes:
            Bazant, ()   -   "   "
        Inputs:
            eps     =   permittivity of fluid       (F/m)
            lamb    =   Debye length                (m)
            zeta    =   zeta potential              (V)
        """
        Cdl_nonlinear_differential = (self.dataset.eps_fluid * self.dataset.eps_0 / self.lamb) * np.cosh(self.dataset.z
                                      * self.dataset.e * self.zeta / (2 * self.dataset.kb * self.dataset.T))

        return Cdl_nonlinear_differential

    def calc_dielectric_capacitance(self):
        """
        Capacitance due to a dielectric coating on the BPE
        units: F/m^2
        Notes:
            Squires, 2010   -   "additional capacitance due to dielectric layer"
                --> The addition of a dielectric layer (instead of using the Stern layer) is direct experimental control.
            Adjari, 2006    -   "additional surface capacitance due to dielectric layer"
        Inputs:
            eps =   permittivity of dielectric layer    (F/m)
            d   =   thickness of the dielectric layer   (m)
        """
        Cd = self.dataset.dielectric_perm * self.dataset.eps_0 / self.dataset.dielectric_h

        return Cd

    def calc_buffer_capacitance(self, Cbuff_input=0.024):
        """
        Equilibrium buffer capacitance
        units: F/m^2
        Notes:
            Squires, 2010   -   "binding of counterions due to equilibrium reaction at charged electrode surface"
                --> This acts in parallel to the double layer capacitance
        Inputs:
            Cbuff_input:    define a specific buffer capacitance    (F/m)
            Ns:             surface density of reactive groups      (#/m2)
            T:              temperature                             (K)
            Ka:             reaction equilibrium constant           (ranges between 2-6)
            a_h:            bulk concentration of protons           (#/m3)
            zeta:           zeta potential at surface               (V)
        """
        Cbuff = Cbuff_input  # (F/m2) taken from Squires but should be fit to data.
        Cbuff_calc = (self.dataset.e ** 2 * self.dataset.Ns / (self.dataset.kb * self.dataset.T)) * (
                    self.dataset.Ka * self.dataset.c_H * np.exp(-self.dataset.e * self.zeta / (
                    self.dataset.kb * self.dataset.T)) / (self.dataset.Ka + self.dataset.c_H * np.exp(
                -self.dataset.e * self.zeta / (self.dataset.kb * self.dataset.T))))

        return Cbuff, Cbuff_calc

    def calc_delta_capacitance_ratio(self, capacitance_doublelayer, capacitance_dielectric):
        """
        Ratio of the double layer to dielectric capacitance
        units: ~
        Notes:
            Squires, 2010   -   "Ratio of the double layer to dielectric capacitance"
            Adjari, 2006    -   "Surface capacitance ratio"
                --> At larger potentials the Debye layer capacitance becomes very large and,
                --> total capacitance is dominated by the Stern layer only.
                --> Capacitance ratios: C_total < C_debye layer < C_dielectric layer
        """
        delta = capacitance_doublelayer / capacitance_dielectric
        return delta

    def calc_beta_capacitance_ratio(self, capacitance_buffer, capacitance_dielectric):
        """
        Ratio of the buffer capacitance to the dielectric capacitance
        units: ~
        Notes:
            Squires, 2010   -   "   "
        """
        beta = capacitance_buffer / capacitance_dielectric
        return beta

    def calc_total_capacitance_bpe(self):
        """
        Total capacitance from Debye layer, dielectric layer, and buffer.
        units: F/m^2
        Notes:
            Squires, 2010   -   "the total capacitance per unit area (in the Debye-Huckel limit)"
        Inputs:
            permittivity of fluid
            Debye length
            Capacitance - electric double layer
            Capacitance - dielectric layer
            Capacitance - buffer (ion adsorption and migration)
        """
        Cdl_linear = self.calc_linear_doublelayer_capacitance()
        Cdl_nonlinear = self.calc_nonlinear_doublelayer_capacitance()
        Cdl_nonlinear_differential = self.calc_nonlinear_differential_capacitance()
        Cd = self.calc_dielectric_capacitance()
        Cbuff, Cbuff_calc = self.calc_buffer_capacitance()

        delta_linear = self.calc_delta_capacitance_ratio(Cdl_linear, Cd)
        delta_nonlinear = self.calc_delta_capacitance_ratio(Cdl_nonlinear, Cd)
        beta = self.calc_beta_capacitance_ratio(Cbuff, Cd)
        beta_calc = self.calc_beta_capacitance_ratio(Cbuff_calc, Cd)
        Ctotal_linear = (self.dataset.eps_fluid * self.dataset.eps_0 / self.lamb) * ((1 + beta / delta_linear) / (1 + delta_linear + beta))
        Ctotal_nonlinear = (self.dataset.eps_fluid * self.dataset.eps_0 / self.lamb) * ((1 + beta / delta_nonlinear) / (1 + delta_nonlinear
                                                                                                     + beta))
        total_capacitance_bpe = {
            "Cdl_linear": Cdl_linear,
            "Cdl_nonlinear": Cdl_nonlinear,
            "Cdl_nonlinear_differential": Cdl_nonlinear_differential,
            "Cd": Cd,
            "Cbuff": Cbuff,
            "Cbuff_calc": Cbuff_calc,
            "delta_linear": delta_linear,
            "delta_nonlinear": delta_nonlinear,
            "beta": beta,
            "beta_calc": beta_calc,
            "Ctotal_linear": Ctotal_linear,
            "Ctotal_nonlinear": Ctotal_nonlinear
        }

        return total_capacitance_bpe

    # ------ CURRENT / CURRENT DENSITIES / CHARGE TRANSPORT  ------

    def calc_exchange_current_bpe(self):
        """
        The area specific exchange current through the BPE.

        IMPORTANT: I am using an equation from Bard and Faulkner with a fudge factor to get results similar to those
        provided by Alex Eden in his figure.

        units: A*m2
        Notes:
            See Olsen paper for some details and Bard and Faulkner for full details.
        """
        j_0 = self.dataset.F * self.dataset.L_bpe * self.dataset.Ka * self.dataset.c * 1e-3  # convert mmol to mol
        fudge_factor = 40
        j_0 = j_0 * fudge_factor

        return j_0

    def calc_total_current_bpe(self):
        """
        Calculate all BPE-related currents and current densities
        Outputs:
            j_0_exchange_current_bpe: the area specific exchange current due to Faradaic reactions at the BPE surface.
        """
        j_0_exchange_current_bpe = self.calc_exchange_current_bpe()

        total_current_bpe = {
            "j_0_exchange_current_bpe": j_0_exchange_current_bpe
        }

        return total_current_bpe

    # ------------ RESISTANCES  ------------------

    def calc_bpe_bulk_electrolyte_resistance(self):
        """
        Bulk resistance of the electrolyte (area-specific) in the region above the BPE.
            units: Ohm*m2

        Notes:
        This terms is derived from a RC-model of a bipolar electrode in parallel with the bulk electrolyte for an
        equivalent electric circuits model.
            Adjari, 2006    -   "(area specific) bulk electrolyte resistance"
            Squires, 2010   -   does not explicitly define this but uses the same equation

        Inputs:
            char_length:    (m)     length of  BPE
            sigma           (S/m)   conductivity of electrolyte/buffer

        Output:
            Resistance:     (Ohm*m^2)
        """
        R_bpe_bulk_electrolyte = self.dataset.L_bpe / (2 * self.dataset.sigma)
        R_bpe_bulk_electrolyte_confinement = R_bpe_bulk_electrolyte / self.geometry['bpe_confinement']

        return R_bpe_bulk_electrolyte, R_bpe_bulk_electrolyte_confinement

    def calc_bpe_charge_transfer_resistance(self):
        """
        The area specific charge transfer resistance through the BPE

        IMPORTANT: I am using an equation from Bard and Faulkner with a fudge factor to get results similar to those
        provided by Alex Eden in his figure. The fudge factor is applied to the exchange current (i.e. not here).

        units: Ohm*m2
        Notes:
            See Olsen paper for some details and Bard and Faulkner for full details.
        """
        R_charge_transfer_resistance_calc = self.dataset.R * self.dataset.T / (self.dataset.F *
                                                                               self.current["j_0_exchange_current_bpe"])

        return R_charge_transfer_resistance_calc

    def calc_total_resistance_bpe(self):
        """
        Calculate bulk electrolyte and charge transfer resistance.
        """
        R_bulk_electrolyte, R_bulk_electrolyte_confinement = self.calc_bpe_bulk_electrolyte_resistance()
        R_charge_transfer = self.calc_bpe_charge_transfer_resistance()

        total_resistance_bpe = {
            "R_bpe_bulk_electrolyte": R_bulk_electrolyte,
            "R_bpe_bulk_electrolyte_confinement": R_bulk_electrolyte_confinement,
            "R_bpe_charge_transfer_resistance": R_charge_transfer,
        }

        return total_resistance_bpe


    # ------------ TIME SCALES  ------------------

    def calc_timescale_ac_field(self):
        """
        The characteristic time the applied AC field is "on" for.
        """
        tau_ac_field = 1 / (4 * (self.w / (2 * np.pi)))

        return tau_ac_field



    def calc_timescale_Ohmic_relaxation(self, capacitance_bpe):
        """
        The RC time constant for an equivalent resistor-capacitor circuit model.
        units: (s)
        Reference: Olsen
        Inputs:
            Bulk electrolyte resistance through the BPE region.
            Total capacitance of the BPE.
            Zeta potential (V):     chosen here as twice the thermal voltage if no input is given.
        Outputs:
            Olsen - RC time to charge/discharge the Debye layer
            Squires - RC charging time through the bulk electrolyte
            Bazant - nonlinear RC charging time through the bulk electrolte
        """
        tau_RC_Olsen = self.resistance["R_bpe_bulk_electrolyte"] * capacitance_bpe
        tau_RC_Olsen_confinement = self.resistance["R_bpe_bulk_electrolyte_confinement"] * capacitance_bpe
        tau_RC_Squires = capacitance_bpe * self.dataset.L_bpe / self.dataset.sigma
        tau_RC_nonlinear_Bazant = self.lamb * self.dataset.L_bpe / self.dataset.D * np.cosh(self.dataset.z *
                                                        self.dataset.e * self.zeta / (2 * self.dataset.kb * self.dataset.T))

        return tau_RC_Olsen, tau_RC_Olsen_confinement, tau_RC_Squires, tau_RC_nonlinear_Bazant

    def calc_timescale_Debye_relaxation(self):
        """
        The Debye relaxation time is the time required to charge the Debye layer
        units: (s)
        References:
            Olsen - tau_debye_olsen         =   permittivity of the fluid / conductivity of the fluid
            Bazant - tau_debye_bazant       =   Debye length ** 2 / mass diffusivity (a "material property of the electrolyte")
            Soni/Bazant - tau_debye_soni    =   Debye length * characterist length scale / mass diffusivity ("EDL charging time")
        """
        tau_Debye_Olsen = self.dataset.eps_fluid * self.dataset.eps_0 / self.dataset.sigma
        tau_Debye_Bazant = self.lamb ** 2 / self.dataset.D
        tau_Debye_Soni = self.lamb * self.dataset.L_bpe / self.dataset.D

        return tau_Debye_Olsen, tau_Debye_Bazant, tau_Debye_Soni

    def calc_timescale_faradaic_relaxation(self):
        """
        The Fardaic charging time is essentially the time to initiate Faradaic reactions. If an electric field is
        applied for longer than the Faradaic relaxation/charging time, then you can expect that Faradaic reactions will
        occur. Conversely, if an electric field is applied for less than the Faradaic relaxation time, you would not
        expect Faradaic reactions to occur.
        units: (s)
        Reference: Olsen
        Inputs:
            R_charge_transfer (Ohm*m^2):    the charge transfer resistance (area-specific)
            C (F/m^2):                      the BPE capacitance per unit area
        """
        tau_faradaic_linear = self.resistance["R_bpe_charge_transfer_resistance"] * self.capacitance['Ctotal_linear']
        tau_faradaic_nonlinear = self.resistance["R_bpe_charge_transfer_resistance"] * self.capacitance['Ctotal_nonlinear']

        return tau_faradaic_linear, tau_faradaic_nonlinear

    def calc_total_timescale_bpe(self):
        """
        Calculate all of the relevant timescales.
        units: (s)
        Reference: Olsen, Squires, Bazant
        Inputs:
            linearity: "linear" or "nonlinear" calculates the relevant timescales for a linear or nonlinear EDL.
        """
        # characteristic timescale the AC field is "on" for
        tau_ac_field = self.calc_timescale_ac_field()

        # Debye relaxation timescales (don't include capacitance; these are "material properites of the electrolyte")
        tau_Debye, tau_bulk_diffusion_Debye, tau_Debye_mixed_lambda_bpe = self.calc_timescale_Debye_relaxation()

        # linear zeta timescales to charge the BPE including: Debye layer, dielectric layer, buffer
        tau_RC_linear_Olsen, tau_RC_linear_Olsen_confinement, tau_RC_linear_Squires, tau_RC_nonlinear_Bazant = \
            self.calc_timescale_Ohmic_relaxation(capacitance_bpe=self.capacitance['Ctotal_linear'])

        # nonlinear zeta timescales to charge the BPE including: Debye layer, dielectric layer, buffer
        tau_RC_nonlinear_Olsen, tau_RC_nonlinear_Olsen_confinement, tau_RC_nonlinear_Squires, _ = \
            self.calc_timescale_Ohmic_relaxation(capacitance_bpe=self.capacitance['Ctotal_nonlinear'])

        # linear and nonlinear timescales for Faradaic reactions to occur
        tau_Faradaic_linear, tau_Faradaic_nonlinear = self.calc_timescale_faradaic_relaxation()

        total_timescale_bpe = {
            "tau_bulk_diffusion_Debye": tau_bulk_diffusion_Debye,
            "tau_AC_field": tau_ac_field,
            "tau_Debye": tau_Debye,
            "tau_Debye_mixed_lambda_bpe": tau_Debye_mixed_lambda_bpe,
            "tau_RC_linear_Olsen": tau_RC_linear_Olsen,
            "tau_RC_linear_Olsen_confinement": tau_RC_linear_Olsen_confinement,
            "tau_RC_linear_Squires": tau_RC_linear_Squires,
            "tau_Faradaic_linear": tau_Faradaic_linear,
            "tau_RC_nonlinear_Olsen": tau_RC_nonlinear_Olsen,
            "tau_RC_nonlinear_Olsen_confinement": tau_RC_nonlinear_Olsen_confinement,
            "tau_RC_nonlinear_Squires": tau_RC_nonlinear_Squires,
            "tau_RC_nonlinear_Bazant": tau_RC_nonlinear_Bazant,
            "tau_Faradaic_nonlinear": tau_Faradaic_nonlinear
        }

        return total_timescale_bpe


    # ---------- INDUCED ZETA POTENTIALS ----------------

    def calc_zeta_induced(self):
        """
        Calculate the induced zeta potential proportional to the applied electric field strength. The induced zeta
        potential is explore in detail later.
        """
        zeta = self.electric_field_strength * self.dataset.L_bpe

        return zeta

    def calc_zeta_induced_bpe(self):
        """
        Induced zeta potential for applied electric field (in the quasi-steady limit)
        Reference: Squires 2010
        """
        zeta_induced = self.electric_field_strength * self.dataset.x_bpe

        return zeta_induced

    def calc_zeta_induced_unsteady(self, tau_RC=None):
        """
        Induced zeta potential (unsteady) - the electric double layer doesn't fully charge during an AC field half-cycle
        """
        if tau_RC is None:
            zeta_induced_unsteady = self.transport['Re'] * self.electric_field_strength * self.dataset.x_bpe * \
                                    np.exp(self.w * self.timescale['tau_AC_field']) / (1 +
                                                                    self.timescale['tau_RC_linear_Squires'] * self.w)
        else:
            zeta_induced_unsteady = self.transport['Re'] * self.electric_field_strength * self.dataset.x_bpe * \
                                    np.exp(self.w * self.timescale['tau_AC_field']) / (1 + tau_RC * self.w)

        return zeta_induced_unsteady

    def calc_total_zeta_induced(self):
        zeta_induced_linear = self.calc_zeta_induced_bpe()
        zeta_induced_linear_capacitance = self.calc_zeta_induced_bpe() / (1 + self.capacitance["delta_linear"] +
                                                                          self.capacitance["beta"])
        zeta_induced_nonlinear_capacitance = self.calc_zeta_induced_bpe() / (1 + self.capacitance["delta_nonlinear"] +
                                                                             self.capacitance["beta"])
        zeta_induced_nonlinear_betacalc_capacitance = self.calc_zeta_induced_bpe() / (1 +
                                                    self.capacitance["delta_nonlinear"] + self.capacitance["beta_calc"])
        zeta_induced_unsteady_linear = self.calc_zeta_induced_unsteady()
        zeta_induced_unsteady_nonlinear = self.calc_zeta_induced_unsteady(tau_RC=self.timescale["tau_RC_nonlinear_Squires"])

        zetas = {
            "zeta_induced_linear": zeta_induced_linear,
            "zeta_induced_linear_capacitance": zeta_induced_linear_capacitance,
            "zeta_induced_linear_unsteady": zeta_induced_unsteady_linear,
            "zeta_induced_nonlinear_capacitance": zeta_induced_nonlinear_capacitance,
            "zeta_induced_nonlinear_betacalc_capacitance": zeta_induced_nonlinear_betacalc_capacitance,
            "zeta_induced_nonlinear_unsteady": zeta_induced_unsteady_nonlinear
        }

        return zetas


    # ---------- INDUCED CHARGE SLIP VELOCITIES ----------------

    def calc_u_slip_quasisteady(self):
        """
        Slip velocity (quasi-steady limit)
        Reference: Squires 2010
        """
        u_slip_quasisteady = self.dataset.eps_fluid * self.dataset.eps_0 * self.electric_field_strength ** 2 * \
                             self.dataset.x_bpe / (2 * self.dataset.mu)

        return u_slip_quasisteady

    def calc_u_slip_unsteady(self, tau_RC=None):
        """
        Slip velocity (unsteady limit)
        Reference: Squires 2010
        """
        if tau_RC is None:
            u_slip_unsteady = -self.dataset.eps_fluid * self.dataset.eps_0 * self.electric_field_strength ** 2 * \
                              self.dataset.x_bpe / (2 * self.dataset.mu * (1 + self.timescale['tau_RC_linear_Squires']
                                                                           ** 2 * self.w ** 2))
        else:
            u_slip_unsteady = -self.dataset.eps_fluid * self.dataset.eps_0 * self.electric_field_strength ** 2 * \
                              self.dataset.x_bpe / (2 * self.dataset.mu * (1 + tau_RC ** 2 * self.w ** 2))

        return u_slip_unsteady

    def calc_total_induced_charge_u_slip(self):
        """
        A wrapper for calculating all the induced charge slip velocities
        """
        # spatial terms (depend on x-location across BPE)
        u_slip_quasisteady = self.calc_u_slip_quasisteady()
        u_slip_linear_unsteady = self.calc_u_slip_unsteady()
        u_slip_nonlinear_unsteady = self.calc_u_slip_unsteady(tau_RC=self.timescale['tau_RC_nonlinear_Squires'])

        # scalar terms (RC max of ic_u_slip terms) (NOTE: RC time constant is 63.2%)
        rcmax_u_slip_quasisteady = np.max(u_slip_quasisteady) * 0.632
        rcmax_u_slip_linear_quasisteady = np.max(u_slip_linear_unsteady) * 0.632
        rcmax_u_slip_nonlinear_quasisteady = np.max(u_slip_nonlinear_unsteady) * 0.632

        induced_charge_u_slip = {
            "ic_u_slip_quasisteady": u_slip_quasisteady,
            "ic_u_slip_linear_unsteady": u_slip_linear_unsteady,
            "ic_u_slip_nonlinear_unsteady": u_slip_nonlinear_unsteady,
            "rcmax_ic_u_slip_quasisteady": rcmax_u_slip_quasisteady,
            "rcmax_ic_u_slip_linear_unsteady": rcmax_u_slip_linear_quasisteady,
            "rcmax_ic_u_slip_nonlinear_unsteady": rcmax_u_slip_nonlinear_quasisteady,
        }

        return induced_charge_u_slip

    # ---------- INDIVIDUAL QUANTITIES ----------------
    # in some cases, we leave the quantities to be defined directly as attributes for easier access.

    def calc_angular_frequency(self, frequency=None):
        """
        angular frequency (w)
        """
        if frequency is None:
            w = 2 * np.pi * self.frequency
        else:
            w = 2 * np.pi * frequency
        return w