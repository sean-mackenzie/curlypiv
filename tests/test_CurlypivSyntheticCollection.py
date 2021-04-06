# test CurlypivSyntheticCollection
"""
Notes about this test program:

"""
from os.path import isdir, join, dirname
from os import mkdir
from pathlib import Path
from math import ceil

# numerics
import numpy as np
import random
from numpy.random import default_rng
import pandas as pd
# scientific
from scipy import signal, misc
from skimage.exposure import rescale_intensity


# matplotlib
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors, cm
import cycler
import time

# curlypiv
from curlypiv.CurlypivSyntheticCollection import CurlypivSyntheticCollection
from curlypiv.CurlypivTestSetup import CurlypivTestSetup, chip, material_liquid, material_solid, fluorescent_particles
from curlypiv.CurlypivTestSetup import illumination, electrode_configuration, microscope, objective, channel, bpe
from curlypiv.CurlypivTestSetup import reservoir, optics, fluid_handling_system, tubing, ccd, microgrid
from curlypiv.utils.flowProfile import generate_flowProfile
from curlypiv.CurlypivPlotting import plot_linear_cube, plot_square
from curlypiv.utils.calibrateCamera import particle_illumination_distribution
from curlypiv.utils.generate_synthetic_imageset import generate_sig_settings, generate_random_coordinates
from curlypiv.utils.particle_intensity_weighting import illuminator

# --- assign some paths ---
gridPath = '/Users/mackenzie/Desktop/testSynthetics/microgrid/grid1_0.tif'
samplePath = '/Users/mackenzie/Desktop/testSynthetics/raw'

illumPath = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test/tests/loc1/E2.5Vmm/run3num/1'
illumXY = '/Users/mackenzie/Desktop/testSynthetics/save/illumXY/iXY.tif'
illumSavePath = '/Users/mackenzie/Desktop/testSynthetics/save/illumXY'
illumSaveName = 'iXY'
showIllumPlot = False

showScatter = False
showIntensityTrajectory = False
saveTxt = True
onlyBrightParticles = True
saveTxtPath = '/Users/mackenzie/PythonProjects/microsig/examples/example_sean/txt'

# --- initialize some necessities ---

# low level materials
sio2_channel = material_solid(zeta=-0.085)
sio2_chip = material_solid(transparency=0.99, index_of_refraction=1.45)
gold_bpe = material_solid(transparency=0.5)
polystyrene = material_solid(transparency=0.9, zeta=-0.01, index_of_refraction=1.5)
tygon = material_solid(zeta=-0.001)
kcl = material_liquid(species='KCl', conductivity=25e-4, concentration=0.1, pH=5.5, density=1000, viscosity=0.00089)
# fluidic
bpe_iceo_reservoir = reservoir(diameter=2e-3, height=2e-3, height_of_reservoir=0, material=kcl)
fhs_reservoir = reservoir(diameter=20e-3, height=20e-3, height_of_reservoir=50e-3, material=kcl)
tubing = tubing(inner_diameter=0.5e-3, length=100e-3, material=tygon)
fhs = fluid_handling_system(fluid_reservoir=bpe_iceo_reservoir, all_tubing=tubing, onchip_reservoir=bpe_iceo_reservoir)
# physical
fluoro_particles = fluorescent_particles(diameter=500e-9, concentration=0.02, materials=polystyrene, electrophoretic_mobility=-20)
bpe_iceo_channel = channel(length=25e-3, width=500e-6, height=20e-6, material_wall_surface=sio2_channel, material_fluid=kcl)
bpe_iceo_bpe = bpe(length=500e-6, width=500e-6, height=30e-9, material=gold_bpe)
bpe_iceo_electrode_config = electrode_configuration(material='Stainless Steel', length=bpe_iceo_channel.length, entrance_length=1e-3)
# optics
microgrid_100um = microgrid(gridPath=gridPath, center_to_center_spacing=100e-6, feature_width=10e-6, grid_type='circle_grid')
emccd = ccd(exposure_time=.03972, img_acq_rate=25, EM_gain=300)
synth_illumination = illumination(source='Hg', excitation=560e-9, emission=595e-9, illumination_distribution=illumXY, calculate_illumination_distribution=True, illumPath=illumPath, illumSavePath=illumSavePath, illumSaveName=illumSaveName, showIllumPlot=showIllumPlot)
x50_objective = objective(numerical_aperture=0.7, magnification=50, field_number=26.5e-3, fluoro_particle=fluoro_particles, illumination=synth_illumination, pixel_to_micron=1/154/100e-6/1e6)
microscope = microscope(type='Olympus iX73', objective=x50_objective, illumination=synth_illumination, ccd=emccd)
# higher-level
bpe_iceo_optics = optics(microscope=microscope, illumination=synth_illumination, fluorescent_particles=fluoro_particles, pixel_to_micron_scaling=None)
bpe_iceo_chip = chip(channel=bpe_iceo_channel, material=sio2_chip, bpe=bpe_iceo_bpe, reservoir=bpe_iceo_reservoir, electrodes=bpe_iceo_electrode_config, material_in_optical_path=sio2_chip, thickness_in_optical_path=1e-3)

# test Setup Class
testSetup = CurlypivTestSetup(name='bpe-iceo', chip=bpe_iceo_chip, optics=bpe_iceo_optics, fluid_handling_system=fhs)



# ----- test 0: CurlypivSyntheticCollection ------

# step 1 - load the synthetic collection class
synCol = CurlypivSyntheticCollection(testSetup=testSetup, imgSamplingPath=samplePath, imgIlluminationXYPath=samplePath,
                                     num_images=50, img_type='.tif')
# step 2 - compute image sampling stats
synCol.sampleImage(p_bkg=90, p_sig=90)

# step 3 - initialize some parameters
n_images = 25               # number of images
buffer_images = 4
z_focal_plane = 4e-6        # height of focal plane
z_resolution = 10           # Increase z-space resolution by 10X
density = 50e-5              # 1
cmap='cool'
num_plot_particles = 20     # number of particles to plot intensity trajectory

# step 4 - generate 2D flow profile
pdf = 0
slip_near = 3
slip_far = 0
E = 0
ep_mobility = 1
U_mag = pdf + slip_far + slip_near + E*ep_mobility + 1
fT = ['slip']          # 'slip', 'ep'

u_xyz = generate_flowProfile(testSetup=testSetup, z_resolution=z_resolution, Umax_pdf=pdf, slip_near=slip_near, slip_far=slip_far,
                                     ep_mobility=ep_mobility, E=E, flowType=fT, y_mod=1.0005, z_mod=10)

# step 5 - generate random coordinates of particles in a domain
x = synCol.microsigSetup['pixel_dim_x'] * 2         # give x-domain more space to accompany particle transport
y = synCol.microsigSetup['pixel_dim_y']             # y-domain constrained by walls
z = synCol.microsigSetup['pixel_dim_z']             # z-domain constrained by walls

x_points = x + 1                                #
y_points = y + 1                                #
z_points = int(z * z_resolution * 1e6 + 1)      # increase the depth-wise resolution

z_pixels = z * 1e6                  # scale units to pixels
z_scale = (z_points - 1) / z        # scaling between integer coordinates and real coordinates

x_space = np.linspace(0, x - 1, x_points) - x / 2       # center the interrogation region in the full x-domain
y_space = np.linspace(0, y - 1, y_points)               #
z_space = np.linspace(0, z_points - 1, z_points)        #

volume = x_points * y_points * z_pixels                 # calculate full domain volume
num_particles = int(density * volume)                   # calculate initial particle density

rng = default_rng()                                     # setup random integer generator
x_rand = rng.integers(low=x_space[0], high=x_space[-1], size=num_particles)
y_rand = rng.integers(low=y_space[0], high=y_space[-1], size=num_particles)
z_rand = rng.integers(low=z_space[0], high=z_space[-1], size=num_particles)

# scale z_rand to microns
z_rand = z_rand / ((z_points - 1) / z_pixels)

# stack random coordinate 1D arrays to form random 3D particle positions
coords = np.column_stack((x_rand, y_rand, z_rand))

# import illumination data to get the interrogation window size
(illum_y, illum_x) = synCol.setup.optics.illumination.illumination_distribution.shape
illum_z = synCol.setup.optics.microscope.objective.depth_of_correlation * 1e6

# center the interrogation window in the x, y, z space
x_center = x_space // 2
y_center = y_space // 2

# initial x window
illum_x_init = x_space[-1] // 2 - illum_x

# initial y window
illum_y_init = y_space[-1] // 2 - illum_y
if illum_y_init < 0:
    illum_y_init = 0

# initial z window
illum_z_init = z_focal_plane - illum_z
if illum_z_init < 0:
    illum_z_init = 0

# final z window
illum_z_final = z_focal_plane + illum_z     # scale to microns
if illum_z_final > z_space[-1] / ((z_points - 1) / z_pixels):
    illum_z_final = 0

# organize coordinates into dataframe
df = pd.DataFrame(data=coords, columns=['x', 'y', 'z'])

# add a 'particle id' column to the dataframe
pid = np.linspace(1, len(df), len(df), dtype=int)
df.insert(loc=0, column='id', value=np.transpose(pid), allow_duplicates=False)
df.set_index(keys='id', drop=True, inplace=True, verify_integrity=False)

# calculate particle intensity
n_c_int = []
for index, particle in df.iterrows():
    # particle intensity checker
    p_c_int = particle_illumination_distribution(particle.x, particle.y, particle.z,
                                                 illum_xy=synCol.setup.optics.illumination.illumination_distribution,
                                                 framex=illum_x, framey=illum_y, startx=illum_x_init, starty=illum_y_init,
                                                 scale_z=True, focal_z=z_focal_plane, testSetup=testSetup)
    n_c_int.append(p_c_int)

# add column for particle illumination
df['c_int'] = n_c_int


# ----- PLOTTING -----

if showScatter:
    # initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(3,1,(1,2), projection='3d')
    axy = fig.add_subplot(4,2,7)
    axz = fig.add_subplot(4,2,8)

    # plot interrogation window
    plot_linear_cube(illum_x_init, illum_y_init, illum_z_init, illum_x, illum_y, dz=illum_z,
                     ax=ax, ylim=y_space[-1], zlim=z_space[-1], color='red', alpha=0.5, label='Vis.')

    # initialize scatter - 3D projection
    sctt = ax.scatter(df.x, df.y, df.z, c=df.c_int, cmap=cmap, alpha=0.25)

    # colorbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), ax=ax, shrink=0.35, aspect=15)
    cbar.ax.set_title('Illumination', fontsize=12)

    # scatter plot - xy projection
    sctt1 = axy.scatter(df.x, df.y, c=df.c_int, alpha=0.25, cmap=cmap)
    plot_square(illum_x_init, illum_y_init, illum_x, illum_y, ax=axy, color='red', alpha=0.5)

    # scatter plot - xz projection
    sctt2 = axz.scatter(df.x, df.z, c=df.c_int, alpha=0.25, cmap=cmap)
    plot_square(illum_x_init, illum_z_init, illum_x, illum_z, ax=axz, color='red', alpha=0.5)

    # setup the figure lables
    ax.set_title('Depth of correlation: {} um'.format(np.round(synCol.setup.optics.microscope.objective.depth_of_correlation*1e6,1)))
    axy.set_title('X-Y projection', fontsize=10)
    axy.set_xlabel('x (pix)')
    axy.set_ylabel('y (pix)')
    axz.set_title('X-Z projection', fontsize=10)
    axz.set_xlabel('x (pix)')
    axz.set_ylabel('z (um)')

# initialize trackers for particle stepper
n = 0               # image number tracker
pid_c_int = []      # particle id and intensity tracker

# start the data loop
for n in range(n_images + 1):

    df.drop(columns='c_int')    # drop c_int here because the first particle location is already plotted above
    n_c_int = []                # list for intensity tracker
    pid_c_counter = []          # list for index and intensity tracker

    for index, particle in df.iterrows():

        # particle intensity checker
        p_c_int = particle_illumination_distribution(particle.x, particle.y, particle.z,
                                                    illum_xy=synCol.setup.optics.illumination.illumination_distribution,
                                                    framex=illum_x, framey=illum_y, startx=illum_x_init,
                                                    starty=illum_y_init, scale_z=True, focal_z=z_focal_plane,
                                                    testSetup=testSetup)
        n_c_int.append(p_c_int)
        pid_c_counter.append([index, n, p_c_int])

    pid_c_int += pid_c_counter
    df['c_int'] = n_c_int

    if showScatter:
        sctt = ax.scatter(df.x, df.y, df.z, c=df.c_int, cmap=cmap, alpha=0.25)  # scatter plot - 3D projection
        sctt1 = axy.scatter(df.x, df.y, c=df.c_int, alpha=0.25, cmap=cmap)      # scatter plot - xy projection
        sctt2 = axz.scatter(df.x, df.z, c=df.c_int, alpha=0.25, cmap=cmap)      # scatter plot - xz projection


    # organize data for export to text file
    if onlyBrightParticles:
        df_illum = df[df['c_int']>0.1]

    saveTxt = True
    if saveTxt:
        df_dp = np.ones_like(df_illum.c_int) * synCol.setup.optics.fluorescent_particles.diameter * 1e6
        df_illum['dp'] = df_dp
        dfx_pos = df_illum.x + x / 4
        coords = np.stack((dfx_pos, df_illum.y, df_illum.z, df_illum.dp, df_illum.c_int))
        coords = np.transpose(coords)
        np.savetxt(fname=saveTxtPath+'/'+str(n)+'.txt', X=coords, fmt='%10.3f')

    # particle motion stepper
    df.x = df.x + U_mag*u_xyz[df.z.astype(int) * z_resolution, df.y.astype(int)]

if showScatter:
    plt.show()


# create dataframe
dfc = pd.DataFrame(data=pid_c_int, index=None, columns=['pid', 'frame', 'c_int'])

if onlyBrightParticles:
    dfc = dfc[dfc['c_int'] > 0.01]

# random choice from list
pid_rand = random.choices(dfc.pid.unique(), k=num_plot_particles)

if showIntensityTrajectory:

    # setup colormap
    intervals = np.linspace(0, 1, num_plot_particles)
    colors = [cm.viridis(x) for x in intervals]

    fig, ax = plt.subplots()

    for i in range(len(pid_rand)):
        y = dfc[dfc['pid'] == pid_rand[i]]
        y = y.sort_values(by='frame', axis=0, ignore_index=True)
        ax.plot(y.frame, y.c_int, color=colors[i], linewidth=4, label=str(i))

    ax.set_xticks(ticks=np.arange(0, len(y)+5, 5, dtype=int), minor=False)
    ax.set_yticks(ticks=np.linspace(0, 1, 5, endpoint=True))
    ax.set_xlabel('Frame (#)')
    ax.set_ylabel('Correlation Intensity (%)')

    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    plt.tight_layout()

    plt.show()



