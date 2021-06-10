# calculate_ICEO.py
"""
Notes
"""

# import modules
import numpy as np
import curlypiv.metrics.calculate_ICEO as iceo


# write script to calculate and output all of the below terms using the testSetup class

"""
Required Inputs:

# physical constants
eps_fluid:      permittivity of water           (F/m2)          CurlypivTestSetup.chip.material_fluid.permittivity
eps_dielectric: permittivity of sio2            ()              CurlypivTestSetup.chip.bpe.dielectric_coating.permittivity
T:              temperature                     (K)             CurlypivTestSetup.chip.material_fluid.temperature

# material properties
rho:            density                         (kg/m3)         depends on the instance
mu:             dynamic viscosity               (m2/s)          CurlypivTestSetup.chip.material_fluid.viscosity
sigma:          electrical conductivity         (S/m)           CurlypivTestSetup.chip.material_fluid.conductivity
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

# outputs
lamb:           debye length                    (m)
Cd:             capacitance of dielectric       (F/m2)          # needs to be scaled by BPE surface area
Cdl_linear:     linear double layer capacitance (F/m2)          # needs to be scaled by BPE surface area
Cdl_nonlinear:  nonlinear double layer cap.     (F/m2)          # needs to be scaled by BPE surface area
Cbuff:          buffer capacitance              (F/m2)          * = 0.024 from Squires * # needs to be scaled by BPE surface area
Ctotal:         total capacitance               (F/m2)          # needs to be scaled by BPE surface area

U:              characteristic velocity         (m/s)
Re:             Reynolds number                 ()
U_HS:           Helmholtz Smoluchowski velocity (m/s)
U_slip:         slip velocity                   (m/s)
tau:            Double layer charging time      (s)
zeta_qu_steady: quasi-steady induced zeta       (V)
U_quasi_steady: quasi-steady slip velocity      (m/s)
zeta_highfreq:  high-frequency induced zeta     (V)
U_highfreq:     high-frequency slip velocity    (m/s)
"""

# define independent variables


# define physical constants
e = 1.602e-19                   # C         elementary charge
k_b = 1.3806e-23                # m2kg/s2K1 Boltzmann constant
Na = 6.022e23                   # 1/mol     Avogadro's number
eps0 = 8.854e-12                # F/m       vacuum
T = 295                         # K         room temperature

# identities of components
dielectric_material = 'SiO2'
electrolyte_material = 'KCl'

# mechanical
mu = 0.001                      # m2/s      water
rho = 1000                      # kg/m3     water

# electro/chemical
eps_fluid = 80 * eps0           # F/m       water
eps_dielectric = 4.6 * eps0     # F/m       SiO2
reaction_site_density = 5       # num/m2    SiO2
Ka = 6                          #           Squires
Kb = -2                         #           Squires
pH = 6.2                        #           100 uM KCl
zeta_wall = -0.08               # V         SiO2 in 100 uM KCl
c = 0.1                         # mmol      100 uM KCl
c_n = Na*c                      # num/m3    number density of 100 uM KCl
z_i = 1                         #           valence of 100 uM KCl
sigma = 18.6e-4                 # S/m       100 uM KCl
thermal_voltage=k_b*T/z_i/e     # V         thermal voltage

# geometry
channel_height = 39e-6          # m         channel height
channel_width = 525e-6          # m         channel width
A = channel_width*channel_width # m2        channel cross sectional area
L = 20e-3                       # m         channel length
L_bpe = 200e-6                  # m         BPE length
x = np.linspace(-L_bpe/2, L_bpe/2, num=100)     # m     discretized BPE length
x_channel = np.linspace(0, L, num=100)          # m     discretized channel length
dielectric_thickness = 1e-12     # m         ALD SiO2 thickness

# experimental
E = 10e3                        # V/m       Electric field strength
f = 100                         # Hz        frequency
w = 2*np.pi*f                   # rad/s     angular frequency

# ------------------------------------------------------------------------

# ------------------ CALCULATE TERMS OF INTEREST -------------------------

# I want to know how long it takes the BPE to charge

# debye length
lamb_wall = iceo.calc_lamb(eps_fluid, T, c)

# --- capacitances ---
# Debye layer capacitance (per unit area) at low voltage
Cd_wall = iceo.calc_dielectric_capacitance(eps_fluid, dielectric_thickness)
# Dielectric layer capacitance (linear)
Cdl_linear_wall = iceo.calc_linear_doublelayer_capacitance(eps_fluid, lamb_wall)
# Dielectric layer capacitance (nonlinear)
Cd_nonlinear_wall = iceo.calc_nonlinear_doublelayer_capacitance(eps_fluid, lamb_wall, zeta_wall)
# Overall capacitance (Adjari)
C_total_Adjari = iceo.calc_total_capacitance_Adjari(eps_fluid, lamb_wall, Cdl_linear_wall)

# --- resistances ---

# exchange current through BPE
R_0 = iceo.calc_bpe_bulk_electrolyte_resistance(characteristic_length=L_bpe, sigma=sigma)       # output units: Ohm*m^2
j_0_bpe = iceo.calc_bpe_exchange_current(K_standard_rate_constant=Ka, c_bulk_oxidized=c, c_bulk_reduced=c, alpha_transfer_coefficient=0.5)
R_ct_bpe = iceo.calc_bpe_charge_transfer_resistance(j_0_bpe=j_0_bpe, T=T)

# time scales
# Debye frequency
debye_frequency_wall = iceo.calculate_Debye_frequency(sigma, eps_fluid)     # (Adjari, 2006)
# Debye charging time
tau_debye = iceo.calc_Debye_charging_time(eps_fluid, sigma)                 # (Adjari, 2006)
# RC time scale to charge Debye layer
tau_RC_debye = iceo.calc_RC_time(Cdl_linear_wall, L_bpe, sigma)             # (Squires, 2010)
# RC time scale to charge Debye layer through bulk electrolyte
tau_RC_electrolyte_Adjari = R_0 * C_total_Adjari                            # (Adjari, 2006)
# RC time scale to charge Debye layer through bulk electrolyte at high voltages
tau_RC_electrolyte_hv_Adjari = R_0 * Cdl_linear_wall                        # (Adjari, 2006)
# Characteristic time for discharging electrode through Faradaic reactions
tau_faradaic_discharge = R_ct_bpe * C_total_Adjari

j = eps_fluid/lamb_wall/Cdl_linear_wall


print("Geometric length: {}".format(L_bpe))
print("Debye length: {}".format(lamb_wall))
print("Debye charging time: {}".format(tau_debye))
print("Debye RC time: {}".format(tau_RC_debye))
print("Ohmic relaxation time: {}".format(tau_RC_electrolyte_Adjari))
print("Ohmic relaxation time (high-voltage): {}".format(tau_RC_electrolyte_hv_Adjari))
print("Faradaic charging time: {}".format(tau_faradaic_discharge))






"""
# ------ Channel variables ------
# Channel current
I = iceo.calc_channel_current(E, sigma, A)
# Electric potential across the channel
phi_channel = iceo.calc_channel_fluid_potential(E, L, L_bpe)
phi_bpe = iceo.calc_zeta_induced(E, x)


# ----- Double layers for the wall -----

# charge accumulated in the Debye layer (low voltages)
q_debye_linear_wall = iceo.calculate_q_debye_linear(eps_fluid, lamb_wall, zeta_wall)      # (Adjari, 2006)
q_debye_nonlinear_wall = iceo.calculate_q_debye_nonlinear(eps_fluid, zeta_wall, c, T)      # (Adjari, 2006)
# charge accumulated in the Debye layer (larger voltages)
q_debye_nonlinear_hv_wall = iceo.calculate_q_debye_nonlinear_hv(eps_fluid, lamb_wall, zeta_wall, T)


# --- voltage drops ---
# voltage drop across the double layer (Debye and Stern/dielectric layers)
# WRONG V_drop_lamb_dielectric = iceo.calc_V_drop_lamb_dielectric(zeta_wall, q_debye_nonlinear_hv_wall, Cd_nonlinear_wall, phi_channel)




"""