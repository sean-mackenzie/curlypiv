# calculate_ICEO.py
"""
Notes
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt

def calculate_ICEO(testSetup, testCol, plot_figs=False, savePath=None):
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

    # identities of components
    dielectric_material = testSetup.chip.bpe.dielectric_coating.name
    electrolyte_material = testSetup.chip.channel.material_fluid.name

    # mechanical
    mu = testSetup.chip.channel.material_fluid.viscosity
    rho = testSetup.chip.channel.material_fluid.density
    T = testSetup.chip.channel.material_fluid.temperature

    # electro/chemical
    eps_fluid = testSetup.chip.channel.material_fluid.permittivity
    eps_dielectric = testSetup.chip.bpe.dielectric_coating.permittivity
    reaction_site_density = testSetup.chip.bpe.dielectric_coating.reaction_site_density
    Ka = testSetup.chip.bpe.dielectric_coating.Ka
    pH = testSetup.chip.channel.material_fluid.pH
    zeta_wall = testSetup.chip.channel.material_wall_surface.zeta
    c = testSetup.chip.channel.material_fluid.concentration
    sigma = testSetup.chip.channel.material_fluid.conductivity

    # geometry
    L = testSetup.chip.channel.length
    L_bpe = testSetup.chip.bpe.length
    x = testSetup.chip.bpe.linspace_x
    channel_height = testSetup.chip.channel.height
    dielectric_thickness = testSetup.chip.bpe.dielectric_coating.thickness

    # PIV
    dt = 1 / testSetup.optics.microscope.ccd.img_acq_rate   # (s) time between images
    p_d = testSetup.optics.fluorescent_particles.diameter
    microns_to_pixels = 1/testSetup.optics.microscope.objective.pixel_to_micron
    img_acq_rate = testSetup.optics.microscope.ccd.img_acq_rate
    u_slip_error_scale = 0.3

    # print PIV stats
    dx_brownian = calc_brownian_displacement(dt, mu, p_d, T)
    print("Brownian displacement: {} for {} particle diameter and {} time step".format(dx_brownian, p_d, dt))
    print("Squires recommended: U_min_acceptable > {} um/s ({} pix/frame) or 20% of Brownian motion".format(np.round(dx_brownian*1e6/dt*0.2,2), np.round(microns_to_pixels*dx_brownian*1e6/(dt*img_acq_rate)*0.2,2)))

    # extract the test collection test parameters
    test_params = []
    for key in testCol.locs:
        loc = testCol.locs[key]
        loc_tests = loc.tests
        for ky in loc_tests:
            test_keys = loc_tests[ky]
            test_params.append((test_keys._E, test_keys._f))

    # initialize output data arrays
    electric_fields = []
    frequencys = []
    dielectrics = []
    buffers = []
    UbyUo = []
    raw_uvel_max = []
    raw_slope = []
    betas = []
    deltas = []
    taus = []
    d_eps = []
    d_pKa = []
    d_Ns = []
    d_thick = []
    b_conc = []
    b_conduct = []
    b_pH = []
    b_viscosity = []
    b_eps = []
    b_debye = []
    voltages = []
    electrode_spacings = []

    # Non-Squires terms
    uvel_brownian_error_steady = []
    uvel_brownian_error_quasisteady = []
    uvel_brownian_error_highfreq = []

    # iterate through test parameters
    for i in range(len(test_params)):


        # iterables
        V_channel = test_params[i][0]
        f = test_params[i][1]

        # calculate intermediaries
        E = V_channel/L
        t = np.linspace(0, 1/f, num=100)
        w = calc_w(f=f)
        lamb = calc_lamb(eps_fluid=eps_fluid, c=c, T=T)
        Cd = calc_dielectric_capacitance(eps=eps_dielectric, d=dielectric_thickness)
        Cdl_linear = calc_linear_doublelayer_capacitance(eps=eps_fluid, lamb=lamb)
        Cbuff = calc_buffer_capacitance(Cbuff_input=0.024)
        C_bare_metal = Cdl_linear + Cbuff
        total_capacitance, beta, delta = calc_total_capacitance(eps_fluid=eps_fluid, lamb=lamb, Cdl=Cdl_linear, Cd=Cd, Cbuff=Cbuff)
        tau = calc_RC_via_bulk_time(capacitance=Cdl_linear, L=L_bpe, sigma=sigma)

        # calculate background flow
        u_HS = calc_U_HS(eps=eps_fluid, zeta=zeta_wall, E=E, mu=mu)
        Re = calc_Re(rho=rho, U=u_HS, l=channel_height, mu=mu)

        # calculate slip flow (DC)
        zeta_induced = calc_zeta_induced(E=E, x=x)
        u_slip = calc_U_slip(eps=eps_fluid, E=E, x=x, mu=mu)
        slope_x = 40
        u_slip_slope = u_slip[slope_x:len(u_slip)-slope_x]

        # calculate the Brownian error for quasi-steady slip flow
        error_brownian_steady = calc_brownian_error(U_estimated=u_slip, u_scale=u_slip_error_scale, dt=dt, viscosity=mu, particle_diameter=p_d, temperature=T)

        # calculate slip flow (quasi-steady)
        zeta_induced_quasisteady = calc_zeta_induced_quasisteady(E=E, x=x)
        u_slip_quasisteady = calc_U_slip_quasisteady(eps=eps_fluid, E=E, x=x, mu=mu)

        # calculate the Brownian error for quasi-steady slip flow
        error_brownian_quasisteady = calc_brownian_error(U_estimated=u_slip_quasisteady, u_scale=u_slip_error_scale, dt=dt, viscosity=mu, particle_diameter=p_d, temperature=T)

        # calculate slip flow (high-frequency)
        zeta_induced_highfreq = calc_zeta_induced_highfreq(Re=Re, E=E, x=x, w=w, t=t, tau=tau)
        u_slip_highfreq = calc_U_slip_highfreq(eps=eps_fluid, E=E, x=x, mu=mu, tau=tau, f=f)

        # calculate the Brownian error for quasi-steady slip flow
        error_brownian_highfreq = calc_brownian_error(U_estimated=u_slip_quasisteady, u_scale=0.1, dt=dt, viscosity=mu, particle_diameter=p_d, temperature=T)

        # calculate induced zeta with linear zeta and dielectric coating
        zeta_induced_Clamb_Cd_linear = calc_zeta_induced_Clamb_Cd(E=E, x=x, Cdl=Cdl_linear, Cd=Cd)

        # calculate induced zeta with nonlinear zeta and dielectric coating
        Cdl_nonlinear = calc_nonlinear_doublelayer_capacitance(eps_fluid, lamb=lamb, zeta=zeta_induced_Clamb_Cd_linear)
        zeta_induced_Clamb_Cd_nonlinear = calc_zeta_induced_Clamb_Cd(E, x, Cdl=Cdl_nonlinear, Cd=Cd)

        # calculate induced zeta with total capacitance
        zeta_induced_total_capacitance = calc_zeta_induced_total_capacitance(E=E, x=x, Cdl=Cdl_linear, Cd=Cd, Cbuff=Cbuff)
        u_ratio_slip_to_HS = U_ratio_slip_to_HS(Cdl=Cdl_linear, Cd=Cd, Cbuff=Cbuff)

        # calculate some Squires specific data
        slope_x = 40
        u_slope = (u_slip_quasisteady[-slope_x]-u_slip_quasisteady[slope_x]) / (x[-slope_x]-x[slope_x])
        u_UbyUo = -u_slope / np.max(u_slip_slope)

        # plot important metrics
        if plot_figs is True:
            import matplotlib as mpl
            from cycler import cycler
            mpl.rc('lines', linewidth=4, linestyle='-')
            mpl.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])

            fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(13,10))
            ax = axes.ravel()

            ax[0].plot(x*1e6, zeta_induced*1e3, label=r'$steady$')
            ax[0].plot(x*1e6, zeta_induced_quasisteady*1e3, label=r'$quasi-steady$')
            ax[0].plot(x * 1e6, zeta_induced_highfreq * 1e3, label=r'$high-frequency$')
            ax[0].plot(x * 1e6, zeta_induced_Clamb_Cd_linear * 1e3, label=r'$C_{\lambda}+C_d (linear)$')
            ax[0].plot(x * 1e6, zeta_induced_Clamb_Cd_nonlinear * 1e3, label=r'$C_{\lambda}+C_d (non linear)$')
            ax[0].plot(x * 1e6, zeta_induced_total_capacitance * 1e3, label=r'$C_{total}$')

            ax[1].plot(x*1e6, u_slip*1e6, label=r'$steady$')
            ax[1].plot(x*1e6, u_slip_quasisteady*1e6, label=r'$quasi-steady$')
            ax[1].plot(x * 1e6, u_slip_highfreq * 1e6, label=r'$high frequency$')

            ax[2].plot(x*1e6, error_brownian_steady, label=r'$error_{steady}$')
            ax[2].plot(x*1e6, error_brownian_quasisteady, label=r'$error_{quasi-steady}$')
            ax[2].plot(x*1e6, error_brownian_highfreq, label=r'$error_{high-frequency}$')
            ax[2].axhline(y=-0.2, xmin=x[0]*1e6, xmax=x[-1]*1e6, color='gray', linestyle='dashed', linewidth=2, alpha=0.65, label=r'$error_{max-acceptable}$')
            ax[2].axhline(y=0.2, xmin=x[0] * 1e6, xmax=x[-1] * 1e6, color='gray', linestyle='dashed', linewidth=2, alpha=0.65,)

            ax[0].set_ylabel(r'$\zeta_{induced} (mV)$')
            ax[0].legend(fancybox=True, loc="upper left", bbox_to_anchor=[1.01, 1])

            ax[1].set_ylabel(r'$U_{slip, induced} (\mu m/s)$')
            ax[1].legend(fancybox=True, loc="upper left", bbox_to_anchor=[1.01, 1])

            ax[2].set_ylim(bottom=-0.5, top=0.5)
            ax[2].set_ylabel(r'$\epsilon_{x} (\frac{\sigma_{x}}{\Delta x})$')
            ax[2].set_xlabel(r'$x (\mu m)$')
            ax[2].set_title((r'Relative Error $(\Delta x = $')+str(u_slip_error_scale*100)+(r'% of $\frac{U_{slip}}{\Delta t})$'))
            ax[2].legend(fancybox=True, loc="upper left", bbox_to_anchor=[1.01, 1])

            plt.suptitle('BPE-ICEO: E={} V/mm, f={} Hz'.format(E*1e-3, int(f)))
            plt.tight_layout()
            plt.show()


        # compile into dictionary
        iceo_stats_dict = {
            'electric_field_strength': E,
            'frequency': f,
            'fluid': electrolyte_material,
            'fluid_viscosity': mu,
            'fluid_density': rho,
            'fluid_temperature': T,
            'fluid_pH': pH,
            'fluid_concentration': c,
            'fluid_conductivity': sigma,
            'fluid_permittivity': eps_fluid,
            'l_bpe': L_bpe,
            'dielectric': dielectric_material,
            'dielectric_thickness': dielectric_thickness,
            'solid_permittivity': eps_dielectric,
            'solid_reaction_site_density': reaction_site_density,
            'solid_Ka': Ka,
            'solid_zeta': zeta_wall,
            'channel_height': channel_height,
            'u_HS': u_HS,
            'flow_Re': Re,
            'capacitance_dielectric': Cd,
            'capacitance_Cdl_linear': Cdl_linear,
            #'capacitance_Cdl_nonlinear': Cdl_nonlinear,                            # should be plotted
            'capacitance_Cbuff': Cbuff,
            'capacitance_total': total_capacitance,
            'beta': beta,
            'delta': delta,
            'tau': tau,
            'debye_length': lamb,
            'max_zeta_induced': np.max(zeta_induced),
            'max_zeta_induced_quasisteady': np.max(zeta_induced_quasisteady),                  # should be plotted
            'max_zeta_induced_highfreq': np.max(zeta_induced_highfreq),                        # should be plotted
            'max_zeta_induced_Clamb_Cd_linear': np.max(zeta_induced_Clamb_Cd_linear),          # should be plotted
            'max_zeta_induced_Clamb_Cd_nonlinear': np.max(zeta_induced_Clamb_Cd_nonlinear),    # should be plotted
            'max_zeta_induced_total_capacitance': np.max(zeta_induced_total_capacitance),      # should be plotted
            'max_u_slip': np.max(u_slip),                                                      # should be plotted
            'u_UbyUo': u_UbyUo,
            'max_u_slip_quasisteady': np.max(u_slip_quasisteady),                              # should be plotted
            'max_u_slip_highfreq': np.max(u_slip_highfreq),                                    # should be plotted
            'u_ratio_slip_to_HS': u_ratio_slip_to_HS
        }

        # append to storage list
        electric_fields.append(E)
        frequencys.append(f)
        dielectrics.append(dielectric_material)
        buffers.append(electrolyte_material)
        UbyUo.append(u_UbyUo)
        raw_uvel_max.append(np.max(u_slip))
        uvel_brownian_error_quasisteady.append(error_brownian_quasisteady)
        uvel_brownian_error_highfreq.append(error_brownian_highfreq)
        raw_slope.append(u_slope)
        betas.append(beta)
        deltas.append(delta)
        taus.append(tau)
        d_eps.append(eps_dielectric)
        d_pKa.append(Ka)
        d_Ns.append(reaction_site_density)
        d_thick.append(dielectric_thickness)
        b_conc.append(c)
        b_conduct.append(sigma)
        b_pH.append(pH)
        b_viscosity.append(mu)
        b_eps.append(eps_fluid)
        b_debye.append(lamb)
        voltages.append(V_channel)
        electrode_spacings.append(L)

    # make numpy arrays of correct datatype
    # append to storage list
    electric_fields = np.array(electric_fields, dtype=float)
    frequencys = np.array(frequencys, dtype=float)
    dielectrics = np.array(dielectrics, dtype=str)
    buffers = np.array(buffers, dtype=str)
    UbyUo = np.array(UbyUo, dtype=float)
    raw_uvel_max = np.array(raw_uvel_max, dtype=float)
    uvel_brownian_error_quasisteady = np.array(uvel_brownian_error_quasisteady, dtype=float)
    uvel_brownian_error_highfreq = np.array(uvel_brownian_error_highfreq, dtype=float)
    raw_slope = np.array(raw_slope, dtype=float)
    betas = np.array(betas, dtype=float)
    deltas = np.array(deltas, dtype=float)
    taus = np.array(taus, dtype=float)
    d_eps = np.array(d_eps, dtype=float)
    d_pKa = np.array(d_pKa, dtype=float)
    d_Ns = np.array(d_Ns, dtype=float)
    d_thick = np.array(d_thick, dtype=float)
    b_conc = np.array(b_conc, dtype=float)
    b_conduct = np.array(b_conduct, dtype=float)
    b_pH = np.array(b_pH, dtype=float)
    b_viscosity = np.array(b_viscosity, dtype=float)
    b_eps = np.array(b_eps, dtype=float)
    b_debye = np.array(b_debye, dtype=float)
    voltages = np.array(voltages, dtype=float)
    electrode_spacings = np.array(electrode_spacings, dtype=float)

    iceo_stats = np.vstack((electric_fields, frequencys, dielectrics, buffers,
                          UbyUo, raw_uvel_max, raw_slope, betas, deltas, taus,
                          d_eps, d_pKa, d_Ns, d_thick,
                          b_conc, b_conduct, b_pH, b_viscosity, b_eps, b_debye,
                          voltages, electrode_spacings)).T

    header = "electric_fields,frequencys,dielectrics,buffers,UbyUo,raw_uvel_max,raw_slope,beta,delta,tau,d_eps,d_pKa,d_Ns,d_thick,b_conc,b_conduct,b_pH,b_viscosity,b_eps,b_debye,voltages,electrode_spacings"
    if savePath:
        # Write to .csv file
        np.savetxt(savePath, iceo_stats, fmt='%s', delimiter=',', header=header)

    return iceo_stats, header


# ---------- INDUCED ZETA AND SLIP VELOCITIES ----------------


def calc_lamb(eps_fluid, T, c):
    """
    Debye length (m) for symmetric monovalent electrolyte
    """
    e = -1.602e-19      # (C) charge of an electron
    kb = 1.3806e-23     # (J/K) Boltzmann constant
    Na = 6.022e23       # (1/mol) Avogadro's number
    z = 1               # () valence of electrolyte
    lamb = np.sqrt(eps_fluid*kb*T/(2*(z**2*Na*c)*e**2))
    return lamb

def calc_zeta_induced(E, x):
    """
    Induced zeta potential for ~DC field
    """
    zeta_induced = E*x
    return zeta_induced

def calc_zeta_induced_Clamb_Cd(E, x, Cdl, Cd):
    delta = Cdl/Cd
    zeta_induced_Clamb_Cd = E*x/(1+delta)
    return zeta_induced_Clamb_Cd

def calc_zeta_induced_quasisteady(E, x):
    """
    Induced zeta potential (quasi-steady limit)
    """
    zeta_induced_quasisteady = E*x
    return zeta_induced_quasisteady

def calc_zeta_induced_total_capacitance(E, x, Cdl, Cd, Cbuff):
    beta = Cbuff/Cd
    delta = Cdl/Cd
    zeta_induced_total_capacitance = E*x/(1+delta+beta)
    return zeta_induced_total_capacitance

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
    t = 1/(4*(w/(2*np.pi)))
    zeta_induced_highfreq = Re*E*x*np.exp(w*t)/(1+tau*w)
    return zeta_induced_highfreq

def calc_U_HS(eps, zeta, E, mu):
    """
    Helmholtz-Smoluchowski
    """
    u_eof = -eps*zeta*E/mu
    return u_eof

def calc_U_slip(eps, E, x, mu):
    """
    Slip velocity (DC field)
    """
    u_slip = -eps*E**2*x/mu
    return u_slip

def calc_U_slip_highfreq(eps, E, x, mu, tau, f):
    """
    Slip velocity (high frequency)
    """
    w = calc_w(f)
    u_slip_highfreq = -eps*E**2*x/(2*mu*(1+tau**2*w**2))
    return u_slip_highfreq

def U_ratio_slip_to_HS(Cdl, Cd, Cbuff):
    beta = Cbuff/Cd
    delta = Cdl/Cd
    u_ratio_slip_to_HS = 1+delta+beta
    return u_ratio_slip_to_HS



# ---------- CAPACITANCES ----------------

def calc_linear_doublelayer_capacitance(eps, lamb):
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
    Cdl_linear = eps/lamb
    return Cdl_linear

def calc_nonlinear_doublelayer_capacitance(eps_fluid, lamb, zeta):
    Cdl_nonlinear = (eps_fluid/lamb)*np.cosh(zeta/2)
    return Cdl_nonlinear

def calc_dielectric_capacitance(eps, d):
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
    Cd = eps/d
    return Cd

def calc_buffer_capacitance(Cbuff_input=0.024, Ns=None, T=None, Ka=None, a_h=None, zeta=None):
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
    e = -1.602e-19          # (C) charge of an electron
    kb = 1.3806e-23         # (J/K) Boltzmann constant
    if Cbuff_input is None:
        Cbuff = (e**2*Ns/(kb*T))*(Ka*a_h*np.exp(-e*zeta/(kb*T))/(Ka+a_h*np.exp(-e*zeta/(kb*T))))
    else:
        Cbuff = Cbuff_input     # (F/m2) taken from Squires but should be fit to data.
    return Cbuff

def calc_doublelayer_dielectric_capacitance(eps_fluid, lamb, Cdl):
    """
    Total capacitance due to Debye layer and surface capacitance (Stern layer or dielectric coating)
    units: F/m^2
    Notes:
        Adjari, 2006    -   "The overall capacitance per unit area in the Debye-Huckel limit"
    Inputs:
        eps_fluid   =   permittivity of the fluid                   (F/m)
        lamb        =   Debye length                                (m)
        Cdl         =   Capacitance due to Stern/dielectric layer   (F/m^2)
    """
    C_total_Adjari = (1/(1+(eps_fluid/lamb/Cdl)))*(eps_fluid/lamb)
    return C_total_Adjari

def calc_total_capacitance(eps_fluid, lamb, Cdl, Cd, Cbuff):
    """
    Total capacitance from Debye layer, dielectric layer, and buffer
    """
    beta = Cbuff/Cd
    delta = Cdl/Cd
    total_capacitance = (eps_fluid/lamb)*((1+beta/delta)/(1+delta+beta))
    return total_capacitance, beta, delta

def calc_delta_capacitance_ratio(capacitance_doublelayer, capacitance_dielectric):
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

def calc_beta_capacitance_ratio(capacitance_buffer, capacitance_dielectric):
    """
    Ratio of the buffer capacitance to the dielectric capacitance
    units: ~
    Notes:
        Squires, 2010   -   "   "
    """
    beta = capacitance_buffer / capacitance_dielectric
    return beta





# ------------ TIME SCALES  ------------------

def calc_Debye_charging_time(eps_fluid, sigma):
    """
    The Debye charging time is the time required to charge the Debye layer
    units: s
    Notes:
        Adjari, 2006    -   "Debye time scale"
    """
    tau_debye = eps_fluid / sigma
    return tau_debye

def calc_RC_via_bulk_time(capacitance, L, sigma):
    """
    Characteristic time for induced double layer to form considering a capacitance.
    units: s
    Notes:
        Squires, 2010   -   "characteristic time for induced double layer to form"
    Inputs:
        capacitance:    F/m^2    C^2*s^2/kg/m^2/m^2       (total capacitance of double layer system)
        L_bpe:          m        m       (characteristic length of BPE)
        sigma:          S/m      C^2*s/kg/m^2/m       (conductivity of electrolyte/buffer)
    Outputs:
        tau:            s
    """
    tau = capacitance*L/sigma
    return tau

def calc_RC_via_bulk_HV_time(capacitance, L, sigma):
    """
    Characteristic charging and relaxation time of the electric double layer through the bulk electrolyte
    units: s
    Notes:
        Adjari, 2006    -   "Relaxation time at high voltages"
            --> At high voltages, the Stern/dielectric layer dominates the capacitance of the
            --> double layer and the relaxation time changes
    """

    tau_debye_highvoltage = capacitance * L / sigma
    return tau_debye_highvoltage

def calc_Debye_charging_via_Faradaic_time(charge_transfer_resistance, capacitance_dielectric):
    """
    Characteristic time for (de)charging the Debye layer through Faradaic reactions
    units: s
    Notes:
        Adjari, 2006    -   "Characteristic time for (de)charging the Debye layer through Faradaic reactions"
            --> When Rct << R0, this can be significantly faster than the Ohmic charging
            --> acting effectively as a "short circuit" on the Debye layer
    """
    tau_charge_transfer = charge_transfer_resistance * capacitance_dielectric
    return tau_charge_transfer

def calculate_Debye_frequency(sigma, eps_fluid):
    """
    The Debye frequency is the inverse of the Debye layer charging time (Adjari, 2006).
    units: Hz
    Notes:
        Adjari, 2006    -   minimum frequency the Debye layer can fully charge
            --> Any driving frequency should be well below this.
    Inputs:
        sigma       =   conductivity of buffer/electrolyte      (S/m)
        eps_fluid   =   permittivity of buffer/electrolyte      (F/m)
    """
    w_D = sigma / eps_fluid
    return w_D


# ----------- CURRENT AND CHARGE TRANSFER -----------------
def calc_channel_current(E, sigma, A):
    """
    Calculate channel current
    """
    I = E * sigma * A
    return I

def calculate_q_debye_linear(eps_fluid, lambda_d, zeta):
    """
    Calculate the charge accumulated in the Debye layer
    (Adjari, 2006)
    units: Coulombs
    Notes:

    """
    q = -eps_fluid * zeta / lambda_d
    return q

def calculate_q_debye_nonlinear(eps_fluid, zeta, c, T):
    """
    Calculate the charge accumulated in the Debye layer
    (Adjari, 2006)
    units: Coulombs
    Notes:
        For voltages below the thermal voltage (Debye Huckel limit)
    """
    kb = 1.3806e-23     # (J/K) Boltzmann constant
    e = -1.602e-19      # (C) charge of an electron
    Na = 6.022e23       # (1/mol) Avogadro's number
    z = 1               # () valence of electrolyte

    q = -1*np.sign(zeta)*np.sqrt(2*eps_fluid*kb*T*((c*Na*(np.exp(-z*e*zeta/kb/T)-1))+(c*Na*(np.exp(-z*-e*zeta/kb/T)-1))))
    return q

def calculate_q_debye_nonlinear_hv(eps_fluid, lambda_d, zeta, T):
    """
    Calculate the charge accumulated in the Debye layer for larger voltages
    (Adjari, 2006)
    units: Coulombs
    Notes:
        For larger voltages
    """
    kb = 1.3806e-23     # (J/K) Boltzmann constant
    e = -1.602e-19      # (C) charge of an electron
    Na = 6.022e23       # (1/mol) Avogadro's number
    z = 1               # () valence of electrolyte

    q = (-eps_fluid/lambda_d)*(2*kb*T/e)*np.sinh(e*zeta/(2*kb*T))
    return q



# ----------- CHANNEL-WISE VARIABLES -----------------

# Bulk fluid potential due to externally applied electric field
def calc_channel_fluid_potential(E, L_channel, L_bpe):
    """
    Calculate the fluid potential (phi) as a function of channel location
    units: V
    """
    channel_start = 0
    channel_stop = L_channel

    bpe_start = L_channel/2 - L_bpe/2
    bpe_stop = L_channel/2 + L_bpe/2

    bpe_step = L_bpe / 100
    channel_step = (L_channel - bpe_stop) / 100


    L_pre_bpe = np.linspace(channel_start, bpe_start, num=100, endpoint=True)
    L_bpe = np.linspace(bpe_start, bpe_stop, num=100, endpoint=True)
    L_post_bpe = np.linspace(bpe_stop+channel_step, channel_stop, endpoint=False)

    L = np.concatenate((L_pre_bpe, L_bpe, L_post_bpe), axis=1)

    phi_channel = E * (1 - L/L_channel)
    return phi_channel

def calc_V_drop_lamb_dielectric(zeta, q_debye, C_dl, phi_channel):
    """
    THIS IS WRONG - THIS IS WRONG - THIS IS WRONG
    Calculate the voltage drop across the Stern/dielectric and Debye layers throughout the channel
    units: V
    """
    V_drop_lamb_dielectric = zeta - q_debye/C_dl + phi_channel
    return V_drop_lamb_dielectric




# ----------- ELECTROCHEMISTRY ----------------
# Exchange current denisty through the BPE
def calc_bpe_exchange_current(K_standard_rate_constant, c_bulk_oxidized, c_bulk_reduced, alpha_transfer_coefficient=0.5):
    """
    The exchange current density at a specific BPE location
    units: Coulombs/s*m2
    Notes:
        K_standard_rate_constant: usually between 2 and 6
        c_bulk (oxidized and reduced) can be taken as just the concentration of the bulk species
        alpha is set to 1/2 in Adjari, 2006
    """
    e = -1.602e-19  # (C) charge of an electron

    j_0 = e * K_standard_rate_constant * c_bulk_reduced**alpha_transfer_coefficient * c_bulk_oxidized**(1-alpha_transfer_coefficient)
    return j_0

# Charge transfer resistance through the BPE
def calc_bpe_charge_transfer_resistance(j_0_bpe, T):
    """
    The area specific charge transfer resistance through the BPE
    units: Ohm*m2
    Notes:

    """
    kb = 1.3806e-23     # (J/K) Boltzmann constant
    e = -1.602e-19      # (C) charge of an electron

    R_ct = kb * T / j_0_bpe / e
    return R_ct

# Charge transfer resistance through bulk electrolyte
def calc_bpe_bulk_electrolyte_resistance(characteristic_length, sigma):
    """
    The area specific charge transfer resistance through the bulk electrolyte
    units: Ohm*m2
    Notes:
        Adjari, 2006    -   "(area specific) bulk electrolyte resistance"
        Squires, 2010   -   does not explicitly define this but uses the same equation
    Inputs:
        char_length:    (m)     length of  BPE
        sigma           (S/m)   conductivity of electrolyte/buffer
    Output:
        Resistance:     Ohm*m^2
    """
    R_0 = characteristic_length / sigma
    return R_0

# Faradaic conductance
def calc_faradaic_conductance(R_0, R_ct, tau_0, tau_ct):
    """
    Faradaic conductance - a measure of the facility of the electrode reaction
    units: ~
    Notes:
        Adjari, 2006    -   "a measure of the facility of the electrode reaction"
    """
    K_resistance = R_0 / R_ct
    K_tau = tau_0 / tau_ct
    # NOTE - both K's should be equal
    return K_resistance, K_tau




# ----- CALCULATE PIV SPECIFIC QUANTITIES -----

def calc_brownian_displacement(dt, viscosity, particle_diameter, temperature):
    """
    Calculate brownian motion characteristic displacement per dt
    """
    kb = 1.3806e-23  # (J/K) Boltzmann constant
    dx = np.sqrt(2*kb*temperature*dt/(3*np.pi*viscosity*particle_diameter))
    return dx

def calc_Re(rho, U, l, mu):
    Re = rho*U*l/mu
    return Re

def calc_w(f):
    """
    angular frequency (w)
    """
    w = 2*np.pi*f
    return w

def calc_particle_image_diameter(magnification, particle_diameter, wavelength, numerical_aperture,
                                 index_of_refraction):
    """
    Calculate the particle image diameter on the camera sensor
    Recommended to be ~2-3 pixels (Westerweel et al., 2009)
    """
    particle_image_diameter = np.sqrt(magnification**2*particle_diameter**2+5.95*(magnification+1)**2*wavelength**2*(index_of_refraction/(2*numerical_aperture))**2)
    return particle_image_diameter

def calc_brownian_error(U_estimated, u_scale, dt, viscosity, particle_diameter, temperature):
    """
    Calculate the error due to Brownian motion relative to the mean squared displacement
    (Santiago & Devansatipathy, 2001)
    """
    kb = 1.3806e-23  # (J/K) Boltzmann constant
    diffusivity_particle = kb*temperature/(3*np.pi*viscosity*particle_diameter)
    error = (1/(U_estimated*u_scale))*np.sqrt(2*diffusivity_particle/(dt))
    return error

def calc_random_piv_error(particle_image_diameter):
    """
    Caclulate the random error amplitude which is proportional to the diameter of the displacement correlation peak.
    (Westerweel et al., 2009)
    """
    c = 0.1
    error = c*np.sqrt(2)*particle_image_diameter/np.sqrt(2)
    return error