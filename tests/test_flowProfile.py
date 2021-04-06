# test_flowProfile
"""
This programs plots the generated flow profiles
"""

# import modules


# import curlypiv
from curlypiv.utils.flowProfile import generate_flowProfile, plot_flowProfile_2D, plot_flowProfile_3D
from curlypiv.CurlypivTestSetup import CurlypivTestSetup, chip, material_liquid, material_solid, fluorescent_particles
from curlypiv.CurlypivTestSetup import electrode_configuration, channel, bpe
from curlypiv.CurlypivTestSetup import reservoir

# step 0 - initialize setup
# low level materials
sio2_channel = material_solid(zeta=-0.085)
sio2_chip = material_solid(transparency=0.99, index_of_refraction=1.45)
gold_bpe = material_solid(transparency=0.5)
polystyrene = material_solid(transparency=0.9, zeta=-0.01, index_of_refraction=1.5)
kcl = material_liquid(species='KCl', conductivity=25e-4, concentration=0.1, pH=5.5, density=1000, viscosity=0.00089)
# fluidic
bpe_iceo_reservoir = reservoir(diameter=2e-3, height=2e-3, height_of_reservoir=0, material=kcl)
# physical
fluoro_particles = fluorescent_particles(diameter=500e-9, concentration=0.02, materials=polystyrene, electrophoretic_mobility=-20)
bpe_iceo_channel = channel(length=25e-3, width=500e-6, height=20e-6, material_wall_surface=sio2_channel, material_fluid=kcl)
bpe_iceo_bpe = bpe(length=500e-6, width=500e-6, height=30e-9, material=gold_bpe)
bpe_iceo_electrode_config = electrode_configuration(material='Stainless Steel', length=bpe_iceo_channel.length, entrance_length=1e-3)
# higher-level
bpe_iceo_chip = chip(channel=bpe_iceo_channel, material=sio2_chip, bpe=bpe_iceo_bpe, reservoir=bpe_iceo_reservoir, electrodes=bpe_iceo_electrode_config, material_in_optical_path=sio2_chip, thickness_in_optical_path=1e-3)
# test Setup Class
testSetup = CurlypivTestSetup(name='bpe-iceo', chip=bpe_iceo_chip, optics=None, fluid_handling_system=None)


# step 0 - initialize the flowProfile variables
slip_near = 4
slip_far = 0
pspf_pressure = -1
E = 1
ep_mobility = 1
(u_xyz_pdf, u_xyz_slip, u_xyz_ep) = (0,0,0)
fT=['pdf', 'slip']
cmap = 'viridis'
interp = 'bilinear'
save_plot = False
savePath = ''

# ----- PRESSURE DRIVEN FLOW -----
if len(fT) > 1:
    u_xyz = generate_flowProfile(testSetup=testSetup, Umax_pdf=pspf_pressure, slip_near=slip_near, slip_far=slip_far,
                                     ep_mobility=ep_mobility, E=E, flowType=fT, z_resolution=10, y_mod=1.0005, z_mod=10)

else:
    if fT == ['pdf']:
        # step 1 - generate flowProfile
        u_xyz_pdf = generate_flowProfile(testSetup=testSetup, Umax_pdf=pspf_pressure, flowType=['pdf'], y_mod=1.0005, z_resolution=10)

    # ----- SLIP FLOW -----
    elif fT == ['slip']:
        # step 1 - generate flowProfile
        u_xyz_slip = generate_flowProfile(testSetup=testSetup, slip_near=slip_near, slip_far=0, flowType=['slip'], z_resolution=10, y_mod=1.05, z_mod=10)

    # ----- ELECTROPHORETIC FLOW -----
    elif fT == ['ep']:
        # step 1 - generate flowProfile
        u_xyz_ep = generate_flowProfile(testSetup=testSetup, E=E, ep_mobility=ep_mobility, flowType=['ep'], z_resolution=10, y_mod=1.0005)

    u_xyz = u_xyz_pdf + u_xyz_slip + u_xyz_ep

# step 2 - plot flow profile in 2D
plot_flowProfile_2D(u_xyz, show_plot=False, save_plot=False, savePath=None, saveName=None, saveType='.jpg',
                        cmap='viridis', interp='bilinear')

# step 2 - plot flow profile in 2D
plot_flowProfile_3D(u_xyz, show_plot=True, save_plot=False, savePath=None, saveName=None, saveType='.jpg',
                        cmap='viridis')