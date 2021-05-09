"""
Notes about the program
"""


# ---------- ----------  STEP 1: LOAD TEST COLLECTION ----------  ---------- ---------- ---------- ---------- ----------

# import modules
from os.path import join
import numpy as np
import pandas as pd

from curlypiv.CurlypivTestSetup import CurlypivTestSetup, chip, material_liquid, material_solid, fluorescent_particles
from curlypiv.CurlypivTestSetup import illumination, electrode_configuration, microscope, objective, channel, bpe, darkfield
from curlypiv.CurlypivTestSetup import reservoir, optics, fluid_handling_system, tubing, ccd, microgrid
from curlypiv.CurlypivTestCollection import CurlypivRun, CurlypivTest, CurlypivTestCollection
from curlypiv.utils.calculate_ICEO import calculate_ICEO

# --- Load file paths ---
dirPath = '/Users/mackenzie/Desktop/04.23.21-iceo-test'
gridPath = join(dirPath, 'setup/calibration/microgrid')
illumPath = join(dirPath, 'setup/calibration/illumination/flatfieldPath.tif')
camNoisePath = None
iceoStatsPath = '/Users/mackenzie/Desktop/04.23.21-iceo-test/results/iceo-stats/iceo-stats-copySquires.csv'
iceoMergeStatsPath = '/Users/mackenzie/Desktop/04.23.21-iceo-test/results/iceo-stats/iceo-merge-stats-copySquires.csv'

# --- Test Setup ---
# grid
grid_show=False
# flatfieldPath
calculateIllumDistribution = False
illum_save_txt = False
illum_save_plot = False
illum_save_image = False
illum_show_plot = False
# darkfield
darkfield_show_image = False
darkfield_save_image = False
darkfield_save_plot = False
# flatfieldPath correction
apply_flatfield_correction = False
# objective
obj_show_depth_plot = False
obj_save_depth_plot = False
# examine testset
examine_testset = False
examine_testset_raw = False
img_animate = 'bgs'
# piv
img_piv = 'bgs'
piv_num_analysis_frames = 30
save_text=False
vectors_on_image=True
show_plot=True
save_plot=False
img_piv_plot = img_piv

# instantiate CurlypivTestSetup class

# low level materials
sio2_channel = material_solid(name='SiO2', zeta=-0.0826, permittivity=4.6, index_of_refraction=1.5, conductivity=1e-18, thickness=500e-6, youngs_modulus=71.7e9, poissons_ratio=0.17, density=2.203e3, dielectric_strength=470e6)         # Ref: 2/13/21, ZuPIV of SiO2 w/ 100 uM KCl
pdms_channel = material_solid(name='PDMS', zeta=-0.005, permittivity=2.5, index_of_refraction=1.4, conductivity=4e-13, thickness=3e-3, youngs_modulus=500e3, poissons_ratio=0.5, density=0.97, dielectric_strength=20e6)                  # Ref: <http://www.mit.edu/~6.777/matprops/pdms.htm>
sio2_chip = material_solid(name='SiO2', transparency=0.99, permittivity=4.6, index_of_refraction=1.5, conductivity=1e-18, thickness=500e-6, youngs_modulus=71.7e9, poissons_ratio=0.17, density=2.203e3, dielectric_strength=470e6)       # Ref: None
gold_bpe = material_solid(name='Au', transparency=0.5, conductivity=22e-9)        # Ref: 4/6/21, 30 nm Au - 75% @ 50X, 51% @ 20X
sio2_dielectric = material_solid(name='SiO2', zeta=-0.08, permittivity=4.6, index_of_refraction=1.5, conductivity=1e-18, thickness=33e-9, youngs_modulus=71.7e9, poissons_ratio=0.17, density=2.203e3, dielectric_strength=470e6, Ka=6, reaction_site_density=5)          # Ref: 2/13/21, ZuPIV of SiO2 w/ 100 uM KCl
alkane_thiol = material_solid(name='Alkane-thiol', transparency=0.99)    # Ref: None
polystyrene = material_solid(name='Polystyrene', transparency=0.9, zeta=-0.045, index_of_refraction=1.59, density=1.05e3, permittivity=2.6) # Ref: 3/4/21 BNL Zetasizer
tygon = material_solid(name='Tygon', zeta=-0.001)                 # Ref: None
kcl = material_liquid(name='KCl', species='KCl', conductivity=18.6e-4, concentration=0.1, pH=6.3, density=1000, viscosity=0.001, permittivity=80.1, temperature=298) # Ref: 3/4/21 BNL Conductivity Meter + Zetasizer
# fluidic
bpe_iceo_reservoir = reservoir(diameter=2e-3, height=2e-3, height_of_reservoir=0, material=kcl)
fhs_reservoir = reservoir(diameter=20e-3, height=20e-3, height_of_reservoir=50e-3, material=kcl)
tubing = tubing(inner_diameter=0.5e-3, length=100e-3, material=tygon)
fhs = fluid_handling_system(fluid_reservoir=bpe_iceo_reservoir, all_tubing=tubing, onchip_reservoir=bpe_iceo_reservoir)
# physical
fluoro_particles = fluorescent_particles(diameter=500e-9, concentration=2e-5, materials=polystyrene, electrophoretic_mobility=-3.531e-8)    # Ref: 3/4/21 BNL Zetasizer
bpe_iceo_channel = channel(length=25e-3, width=500e-6, height=15e-6, material_wall_surface=sio2_channel, material_fluid=kcl)                # Ref: 3/11/21, NCF Dektak of PDMS channel height
bpe_iceo_bpe = bpe(length=50e-6, width=500e-6, height=20e-9, material=gold_bpe, adhesion_material=alkane_thiol, dielectric_coating=sio2_dielectric)                            # Ref: 4/6/21, Brightfield BPE images + NCF Dektak
bpe_iceo_electrode_config = electrode_configuration(material='Stainless Steel', length=bpe_iceo_channel.length, entrance_length=1e-3)
# optics
microgrid_100um = microgrid(gridPath=gridPath, center_to_center_spacing=50e-6, feature_width=10e-6, grid_type='grid', show_grid=grid_show)
darkfield = darkfield(basePath=dirPath, show_image=darkfield_show_image, save_image=darkfield_save_image, save_plot=darkfield_save_plot)
emccd = ccd(exposure_time=.045, img_acq_rate=22.09, EM_gain=25, darkfield=darkfield)
illumination = illumination(basePath=dirPath, source='Hg', excitation=460e-9, emission=495e-9, calculate_illumination_distribution=calculateIllumDistribution, illumPath=illumPath, showIllumPlot=illum_show_plot, save_plot=illum_save_plot, save_image=illum_save_image, save_txt=illum_save_txt)
x50_objective = objective(numerical_aperture=0.7, magnification=50, field_number=26.5e-3, fluoro_particle=fluoro_particles, illumination=illumination, pixel_to_micron=0.604, basePath=dirPath, channel_height=bpe_iceo_channel.height, show_depth_plot=obj_show_depth_plot, save_depth_plot=obj_save_depth_plot) # Ref: 4/6/21, Brightfield grid @ 50X
microscope = microscope(type='Olympus iX73', objective=x50_objective, illumination=illumination, ccd=emccd)
# higher-level
bpe_iceo_optics = optics(microscope=microscope, fluorescent_particles=fluoro_particles, calibration_grid=microgrid_100um, pixel_to_micron_scaling=x50_objective.pixel_to_micron)
bpe_iceo_chip = chip(channel=bpe_iceo_channel, material=sio2_chip, bpe=bpe_iceo_bpe, reservoir=bpe_iceo_reservoir, electrodes=bpe_iceo_electrode_config, material_in_optical_path=sio2_chip, thickness_in_optical_path=500e-6)
# --- --- --- --- --- ---

# --- Test Collection ---
# load test files
name = 'testCol'

img_type = '.tif'
testid = ('V','channel', 'f', 'Hz')
runid = ('run', 'num')
seqid = ('test_', '_X')
frameid = '_X'

# declare filter settings
cropspecs = None
filterspecs = None
resizespecs = None
backsubspecs = None
# --- --- --- --- --- ---

# instantiate CurlypivTestCollection and CurlypivTestSetup classes
testCol = CurlypivTestCollection(collectionName=name, dirpath=dirPath, file_type=img_type, testid=testid, runid=runid, seqid=seqid, frameid=frameid, load_files=False)
testSetup = CurlypivTestSetup(name='bpe-iceo', chip=bpe_iceo_chip, optics=bpe_iceo_optics, fluid_handling_system=fhs)

# --- calculate ICEO ---
iceo_stats, header = calculate_ICEO(testSetup=testSetup, testCol=testCol, plot_figs=True, savePath=iceoStatsPath)
header = header.split(',')

# read Squires output data from .csv
squiresPath = '/Users/mackenzie/Desktop/04.23.21-iceo-test/results/benchmark/Pascall_and_Squires_2009_SMoutputs.csv'
dtypes = "f8,f8,U5,U5,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8"
names = 'electric_fields,frequencys,dielectrics,buffers,U_true_est,UbyUo,raw_uvel_max,raw_slope,beta,delta,tau,d_eps,d_pKa,d_pKb,d_Ns,d_thick,b_conc,b_conduct,b_pH,b_viscosity,b_eps,b_debye,voltages,electrode_spacings,pivcorr'
names = names.split(',')
squires_stats = np.genfromtxt(squiresPath, delimiter=',', dtype=dtypes, names=names)

# create dataframes
df_iceo = pd.DataFrame(data=iceo_stats, index=None, columns=header)
df_squires = pd.DataFrame(data=squires_stats, index=None, columns=names)

# change data types of each column
df_iceo = df_iceo.astype({'electric_fields': float,'frequencys': float,'dielectrics': str,'buffers': str, 'UbyUo': float,'raw_uvel_max': float,'raw_slope': float,'beta': float,'delta': float,'tau': float,'d_eps': float,'d_pKa': float,'d_Ns': float,'d_thick': float,'b_conc': float,'b_conduct': float,'b_pH': float,'b_viscosity': float,'b_eps': float,'b_debye': float,'voltages': float,'electrode_spacings': float})
df_squires = df_squires.astype({'electric_fields': float,'frequencys': float,'dielectrics': str,'buffers': str,'U_true_est': float,'UbyUo': float,'raw_uvel_max': float,'raw_slope': float,'beta': float,'delta': float,'tau': float,'d_eps': float,'d_pKa': float,'d_pKb': float,'d_Ns': float,'d_thick': float,'b_conc': float,'b_conduct': float,'b_pH': float,'b_viscosity': float,'b_eps': float,'b_debye': float,'voltages': float,'electrode_spacings': float,'pivcorr': float})

# add columns for "sean" and "squires"
sm = np.ones(len(df_iceo))
squires = np.zeros(len(df_squires))
df_iceo.insert(0, "id", sm, allow_duplicates=True)
df_squires.insert(0, "id", squires, allow_duplicates=True)

# adjust units in df_iceo to match df_squires
df_iceo.UbyUo = df_iceo.UbyUo / 1e6                 # (m/s) --> (um/s)
df_iceo.raw_uvel_max = df_iceo.raw_uvel_max   # (m/s) --> (um/s)
df_iceo.raw_slope = df_iceo.raw_slope         # (m/s) --> (um/s)
df_iceo.d_eps = df_iceo.d_eps / 8.854e-12           # (epsr*eps0) --> (epsr)
df_iceo.d_thick = df_iceo.d_thick * 1e9             # (m) --> (nm)
df_iceo.b_conc = df_iceo.b_conc * 1e3               # (mmol) --> (umol)
df_iceo.b_conduct = df_iceo.b_conduct * 1e4         # (S/m) --> (uS/cm)
df_iceo.b_eps = df_iceo.b_eps / 8.854e-12           # (epsr*eps0) --> (epsr)


# filter data frames
Emin = df_iceo.electric_fields.min()*0.25
Emax = df_iceo.electric_fields.max()*2
fmin = df_iceo.frequencys.min()*0.25
fmax = df_iceo.frequencys.max()*1.25
buff = df_iceo.buffers.unique().tolist()[0]
buff_min = df_iceo.b_conc.min()*0.5
buff_max = df_iceo.b_conc.max()*1.5
diel = df_iceo.dielectrics.unique().tolist()[0]
diel_thick_max = df_iceo.d_thick.max()*1.25


# filter electric fields
dff = df_squires.loc[(df_squires['electric_fields'] >= Emin) & (df_squires['electric_fields'] <= Emax)]
# filter frequencies
dff = dff.loc[(dff['frequencys'] >= fmin) & (dff['frequencys'] <= fmax)]
# filter buffer and buffer concentration
dff = dff.loc[(df_squires['buffers'] == buff) & (df_squires['b_conc'] >= buff_min) & (df_squires['b_conc'] <= buff_max)]
# filter dielectric thickness
dff = dff.loc[(dff['dielectrics'] == diel) & (df_squires['d_thick'] < diel_thick_max)]
# assign to proper variable
df_filter_squires = dff

# cocatenate Squires and SM dataframes
result = pd.concat([df_iceo, df_filter_squires], ignore_index=True, sort=False)

# write to csv
result.to_csv(iceoMergeStatsPath)



j=1





