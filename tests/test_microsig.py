# test Curlypiv.utils.microsig
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
from curlypiv.utils.microsig import CurlypivMicrosigCollection

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

export_settings_file = True
export_settings_path = '/Users/mackenzie/PythonProjects/microsig/examples/example_sean'
export_settings_name = 'auto_settings.txt'

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

# Load testSetup Class
testSetup = CurlypivTestSetup(name='bpe-iceo', chip=bpe_iceo_chip, optics=bpe_iceo_optics, fluid_handling_system=fhs)

# Load syntheticCollection class
synCol = CurlypivSyntheticCollection(testSetup=testSetup, imgSamplingPath=samplePath, imgIlluminationXYPath=samplePath,
                                     num_images=50, img_type='.tif', export_settings_file=export_settings_file, export_settings_path=export_settings_path, export_settings_name=export_settings_name,)

# ----- test microsig -----
use_gui = False
setting_file = '/Users/mackenzie/PythonProjects/microsig/examples/example_sean/settings_sm.txt'
data_files = '/Users/mackenzie/PythonProjects/microsig/examples/example_sean/txt'
destination_folder = '/Users/mackenzie/PythonProjects/microsig/examples/example_sean/images'

use_internal_setting = False
use_internal_data = False
to_internal_sequence = False
output_dtype = 'np.uint16'


microsigCol = CurlypivMicrosigCollection(testSetup=testSetup, synCol=synCol, use_gui=use_gui,
                                         use_internal_setting=use_internal_setting, use_internal_data=use_internal_data,
                                         to_internal_sequence=to_internal_sequence, output_dtype=output_dtype,
                                         setting_file=setting_file, data_files=data_files, destination_folder=destination_folder,
                                         )





