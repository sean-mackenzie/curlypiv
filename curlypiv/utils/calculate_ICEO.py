# calculate_ICEO.py
"""
Notes
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt

def calculate_ICEO(testSetup, testCol):
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

    # define variables here to simplify
    # mechanical
    mu = testSetup.chip.channel
    rho = testSetup.chip.channel.material_fluid.density
    T = testSetup.chip.channel.material_fluid.temperature

    # electro/chemical
    eps_fluid = testSetup.chip.channel.material_fluid.permittivity
    eps_dielectric = testSetup.chip.bpe.dielectric_coating.permittivity
    reaction_site_density = testSetup.chip.bpe.dielectric_coating.reaction_site_density
    Ka = testSetup.chip.bpe.dielectric_coating.Ka
    pH = testSetup.chip.channel.material_fluid.pH
    zeta_wall = testSetup.chip.channel.material_fluid.zeta
    c = testSetup.chip.channel.material_fluid.concentration
    sigma = testSetup.chip.channel.material_fluid.conductivity

    # geometry
    x = testSetup.chip.bpe.linspace_x
    channel_height = testSetup.chip.channel.height
    dielectric_thickness = testSetup.chip.bpe.dielectric_coating.thickness

    # test inputs
    E = testCol.locs.tests._E
    f = testCol.locs.tests._f
    t = np.linspace(0, 1/f, num=100)

    # calculate intermediaries
    w = calc_w(f=f)
    lamb = calc_lamb(c=c, T=T)
    Cd = calc_dielectric_capacitance(eps=eps_dielectric, d=dielectric_thickness)
    Cdl_linear = calc_linear_doublelayer_capacitance(eps=eps_dielectric, lamb=lamb)
    Cbuff = calc_buffer_capacitance(Cbuff_input=0.024)
    total_capacitance = calc_total_capacitance(eps_fluid=eps_fluid, lamb=lamb, Cdl=Cdl_linear, Cd=Cd, Cbuff=Cbuff)
    tau_max, tau_x = calc_RC_time(capacitance_bpe=total_capacitance, x=x, sigma=sigma)

    # calculate background flow
    u_HS = calc_U_HS(eps=eps_fluid, zeta=zeta_wall, E=E, mu=mu)
    Re = calc_Re(rho=rho, U=u_HS, l=channel_height, mu=mu)

    # calculate slip flow (DC)
    u_slip = calc_U_slip(eps=eps_fluid, E=E, x=x, mu=mu)

    # calculate slip flow (quasi-steady)
    zeta_induced_quasisteady = calc_zeta_induced_quasisteady(E=E, x=x)
    u_slip_quasisteady = calc_U_slip_quasisteady(eps=eps_fluid, E=E, x=s, mu=mu)

    # calculate slip flow (high-frequency)
    zeta_induced_highfreq = calc_zeta_induced_highfreq(Re=Re, E=E, x=x, w=w, t=t, tau=tau_x)
    u_slip_highfreq = calc_U_slip_highfreq(eps=eps_fluid, E=E, x=x, mu=mu, lamb=lamb, sigma=sigma, f=f)

    # calculate induced zeta with linear zeta and dielectric coating
    zeta_induced_Clamb_Cd_linear = calc_zeta_induced_Clamb_Cd(E=E, x=x, Cdl=Cdl_linear, Cd=Cd)

    # calculate induced zeta with nonlinear zeta and dielectric coating
    Cdl_nonlinear = calc_nonlinear_doublelayer_capacitance(eps_fluid, lamb=lamb, zeta=zeta_induced_Clamb_Cd_linear)
    zeta_induced_Clamb_Cd_nonlinear = calc_zeta_induced_Clamb_Cd(E, x, Cdl=Cdl_nonlinear, Cd=Cd)

    # calculate induced zeta with total capacitance
    zeta_induced_total_capacitance = calc_zeta_induced_total_capacitance(E=E, x=x, Cdl=Cdl_linear, Cd=Cd, Cbuff=Cbuff)
    u_ratio_slip_to_HS = U_ratio_slip_to_HS(Cdl=Cdl_linear, Cd=Cd, Cbuff=Cbuff)

    # compile into dictionary
    iceo_stats = {
        'electric_field_strength': E,
        'frequency': f,
        'fluid_viscosity': mu,
        'fluid_density': rho,
        'fluid_temperature': T,
        'fluid_pH': pH,
        'fluid_concentration': c,
        'fluid_conductivity': sigma,
        'fluid_permittivity': eps_fluid,
        'solid_permittivity': eps_dielectric,
        'solid_reaction_site_density': reaction_site_density,
        'solid_Ka': Ka,
        'solid_zeta': zeta_wall,
        'channel_height': channel_height,
        'u_HS': u_HS,
        'flow_Re': Re,
        'dielectric_thickness': dielectric_thickness,
        'capacitance_dielectric': Cd,
        'capacitance_Cdl_linear': Cdl_linear,
        'capacitance_Cdl_nonlinear': Cdl_nonlinear,
        'capacitance_Cbuff': Cbuff,
        'capacitance_total': total_capacitance,
        'tau': tau_max,
        'debye_length': lamb,
        'zeta_induced_quasisteady': zeta_induced_quasisteady,
        'zeta_induced_highfreq': zeta_induced_highfreq,
        'zeta_induced_Clamb_Cd_linear': zeta_induced_Clamb_Cd_linear,
        'zeta_induced_Clamb_Cd_nonlinear': zeta_induced_Clamb_Cd_nonlinear,
        'zeta_induced_total_capacitance': zeta_induced_total_capacitance,
        'u_slip': u_slip,
        'u_slip_quasisteady': u_slip_quasisteady,
        'u_slip_highfreq': u_slip_highfreq,
        'u_ratio_slip_to_HS': u_ratio_slip_to_HS
    }

def calc_U_HS(eps, zeta, E, mu):
    """
    Helmholtz-Smoluchowski
    """
    u_eof = -eps*zeta*E/mu
    return u_eof

def calc_Re(rho, U, l, mu):
    Re = rho*U*l/mu
    return Re

def calc_w(f):
    """
    angular frequency (w)
    """
    w = 2*np.pi*f
    return w

def calc_lamb(eps_fluid, T, c):
    """
    Debye length (m) for symmetric monovalent electrolyte
    """
    e = -1.602e-19      # (C) charge of an electron
    kb = 1.3806e-23     # (J/K) Boltzmann constant
    Na = 6.022e23       # (1/mol) Avogadro's number
    z = 1               # () valence of electrolyte
    lamb = np.sqrt(eps_fluid*kb*T/(e**2*2*(z**2*Na*c*1e-3)))
    return lamb

def calc_U_slip(eps, E, x, mu):
    """
    Slip velocity (DC field)
    """
    u_slip = -eps*E**2*x/mu
    return u_slip

def calc_RC_time(capacitance_bpe, x, sigma):
    """
    The RC time constant for charging an electric double layer.
    """
    tau_x = capacitance_bpe*x/sigma
    tau_max = np.max(tau_x)
    return tau_max, tau_x

def calc_zeta_induced_quasisteady(E, x):
    """
    Induced zeta potential (quasi-steady limit)
    """
    zeta_induced_quasisteady = E*x
    return zeta_induced_quasisteady

def calc_U_slip_quasisteady(eps, E, x, mu):
    """
    Slip velocity (quasi-steady limit)
    """
    u_slip_quasisteady = -eps*E**2*x/(2*mu)
    return u_slip_quasisteady

def calc_zeta_induced_highfreq(Re, E, x, w, t, tau):
    """
    Induced zeta potential (high frequency)
    """
    zeta_induced_highfreq = Re*E*x*np.exp(w*t)/(1+tau*w)
    return zeta_induced_highfreq

def calc_U_slip_highfreq(eps, E, x, mu, lamb, sigma, f):
    """
    Slip velocity (high frequency)
    """
    w = calc_w(f)
    tau = calc_RC_time(eps, x, lamb, sigma)
    u_slip_highfreq = -eps*E**2*x/(2*mu*(1+tau**2*w**2))
    return u_slip_highfreq

def calc_dielectric_capacitance(eps, d):
    Cd = eps/d
    return Cd

def calc_linear_doublelayer_capacitance(eps, lamb):
    Cdl_linear = eps/lamb
    return Cdl_linear

def calc_nonlinear_doublelayer_capacitance(eps_fluid, lamb, zeta):
    Cdl_nonlinear = (eps_fluid/lamb)*np.cosh(zeta/2)
    return Cdl_nonlinear

def calc_zeta_induced_Clamb_Cd(E, x, Cdl, Cd):
    delta = Cdl/Cd
    zeta_induced_Clamb_Cd = E*x/(1+delta)
    return zeta_induced_Clamb_Cd

def calc_buffer_capacitance(Cbuff_input=0.024, Ns=None, T=None, Ka=None, a_h=None, zeta=None):
    e = -1.602e-19          # (C) charge of an electron
    kb = 1.3806e-23         # (J/K) Boltzmann constant
    if Cbuff_input is None:
        Cbuff = (e**2*Ns/(kb*T))*(Ka*a_h*np.exp(-e*zeta/(kb*T))/(Ka+a_h*np.exp(-e*zeta/(kb*T))))
    else:
        Cbuff = Cbuff_input     # (F/m2) taken from Squires but should be fit to data.
    return Cbuff

def calc_zeta_induced_total_capacitance(E, x, Cdl, Cd, Cbuff):
    beta = Cbuff/Cd
    delta = Cdl/Cd
    zeta_induced_total_capacitance = E*x/(1+delta+beta)
    return zeta_induced_total_capacitance

def calc_total_capacitance(eps_fluid, lamb, Cdl, Cd, Cbuff):
    beta = Cbuff/Cd
    delta = Cdl/Cd
    total_capacitance = (eps_fluid/lamb)*((1+beta/delta)/(1+delta+beta))
    return total_capacitance

def U_ratio_slip_to_HS(Cdl, Cd, Cbuff):
    beta = Cbuff/Cd
    delta = Cdl/Cd
    u_ratio_slip_to_HS = 1+delta+beta
    return u_ratio_slip_to_HS