"""
Notes about the program
"""

# -------- import modules -----------
# import modules
from os.path import join
from skimage.morphology import disk
import numpy as np

from curlypiv.CurlypivTestSetup import CurlypivTestSetup, chip, material_liquid, material_solid, fluorescent_particles
from curlypiv.CurlypivTestSetup import illumination, electrode_configuration, microscope, objective, channel, bpe, darkfield
from curlypiv.CurlypivTestSetup import reservoir, optics, fluid_handling_system, tubing, ccd, microgrid
from curlypiv.CurlypivTestCollection import CurlypivTestCollection

# ---------- ----------  STEP 1: LOAD TEST COLLECTION and TEST SETUP settings ----------  ---------- ---------- ---------- ---------- ----------

# --- Load file paths ---
dirPath = '/Users/mackenzie/Desktop/04.23.21-iceo-test'
gridPath = join(dirPath, 'setup/calibration/microgrid/grid_10umLines_50umSpacing/50X')
flatfieldPath = join(dirPath, 'setup/calibration/illumination/flatfield.tif')
darkfieldPath = join(dirPath, 'setup/calibration/cameraNoise/darkfield/darkfield.tif')
iceoStatsPath = '/Users/mackenzie/Desktop/04.23.21-iceo-test/results/iceo-stats/iceo-stats.csv'
iceoMergeStatsPath = '/Users/mackenzie/Desktop/04.23.21-iceo-test/results/iceo-stats/iceo-merge-stats.csv'
load_files = True

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
apply_darkfield_correction = True
# objective
obj_show_depth_plot = False
obj_save_depth_plot = False
# examine testset
examine_testset = True
examine_testset_raw = False
img_animate = 'filtered'
# piv
backSub_init_frames = 50

# low level materials
sio2_channel = material_solid(name='SiO2', zeta=-0.0826, permittivity=4.6, index_of_refraction=1.5, conductivity=1e-18, thickness=500e-6, youngs_modulus=71.7e9, poissons_ratio=0.17, density=2.203e3, dielectric_strength=470e6)         # Ref: 2/13/21, ZuPIV of SiO2 w/ 100 uM KCl
pdms_channel = material_solid(name='PDMS', zeta=-0.005, permittivity=2.5, index_of_refraction=1.4, conductivity=4e-13, thickness=3e-3, youngs_modulus=500e3, poissons_ratio=0.5, density=0.97, dielectric_strength=20e6)                  # Ref: <http://www.mit.edu/~6.777/matprops/pdms.htm>
sio2_chip = material_solid(name='SiO2', transparency=0.99, permittivity=4.6, index_of_refraction=1.5, conductivity=1e-18, thickness=500e-6, youngs_modulus=71.7e9, poissons_ratio=0.17, density=2.203e3, dielectric_strength=470e6)       # Ref: None
gold_bpe = material_solid(name='Au', transparency=0.5, conductivity=22e-9)        # Ref: 4/6/21, 30 nm Au - 75% @ 50X, 51% @ 20X
sio2_dielectric = material_solid(name='SiO2', zeta=-0.08, permittivity=4.6, index_of_refraction=1.5, conductivity=1e-18, thickness=5e-9, youngs_modulus=71.7e9, poissons_ratio=0.17, density=2.203e3, dielectric_strength=470e6, Ka=6, reaction_site_density=5)          # Ref: 2/13/21, ZuPIV of SiO2 w/ 100 uM KCl
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
bpe_iceo_channel = channel(length=25e-3, width=500e-6, height=15e-6, material_bottom_wall_surface=sio2_channel, material_fluid=kcl)                # Ref: 3/11/21, NCF Dektak of PDMS channel height
bpe_iceo_bpe = bpe(length=50e-6, width=500e-6, height=20e-9, material=gold_bpe, adhesion_material=alkane_thiol, dielectric_coating=sio2_dielectric)                            # Ref: 4/6/21, Brightfield BPE images + NCF Dektak
bpe_iceo_electrode_config = electrode_configuration(material='Stainless Steel', length=bpe_iceo_channel.length, entrance_length=1e-3)
# optics
microgrid_100um = microgrid(gridPath=gridPath, center_to_center_spacing=50e-6, feature_width=10e-6, grid_type='grid', show_grid=grid_show)
darkfield = darkfield(basePath=dirPath, show_image=darkfield_show_image, save_image=darkfield_save_image, save_plot=darkfield_save_plot)
emccd = ccd(exposure_time=.045, img_acq_rate=22.09, EM_gain=25, darkfield=darkfield)
illumination = illumination(basePath=dirPath, source='Hg', excitation=460e-9, emission=495e-9, calculate_illumination_distribution=calculateIllumDistribution, illumPath=flatfieldPath, showIllumPlot=illum_show_plot, save_plot=illum_save_plot, save_image=illum_save_image, save_txt=illum_save_txt)
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
bpespecs = {
    'bxmin': 172,  # x = 0 is the left of the image
    'bxmax': 251,
    'bymin': 10,
    'bymax': 400,  # y = 0 is the bottom of the image
    'multiplier': 2
}

cropspecs = {
    'xmin': 120,  # x = 0 is the left of the image
    'xmax': 320,
    'ymin': 10,
    'ymax': 400  # y = 0 is the bottom of the image
}

filterspecs = {
    'denoise_wavelet': {'args': [], 'kwargs': dict(method='BayesShrink', mode='soft', rescale_sigma=True)},
    'median': {'args': [disk(3)]},
    'gaussian': {'args': [2]},
    'equalize_adapthist': {'args': [], 'kwargs': dict(kernel_size=int(np.round(bpe_iceo_bpe.length*1e6/x50_objective.pixel_to_micron/6)))},
    'rescale_intensity': {'args': [(70, 99.25), ('dtype')]} # 70, 99.999
}

resizespecs = {
    'method': 'pyramid_expand',
    'scale': 2
}
backsubspecs = {
    'bg_method': 'KNN'
}
# --- --- --- --- --- ---
# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------



# ---------- ----------  STEP 2: INSTANTIATE TEST COLLECTION and TEST SETUP ----------  ----------

# instantiate CurlypivTestCollection and CurlypivTestSetup classes
testCol = CurlypivTestCollection(collectionName=name, dirpath=dirPath, file_type=img_type, testid=testid, runid=runid, seqid=seqid, frameid=frameid, load_files=load_files,
                                 calibration_grid_path=gridPath, calibration_illum_path=flatfieldPath,
                                 calibration_camnoise_path=darkfieldPath,
                                 bpe_specs=bpespecs, cropping_specs=cropspecs, resizing_specs=resizespecs, processing_specs=filterspecs,
                                 backsub_specs=backsubspecs
                                 )
testSetup = CurlypivTestSetup(name='bpe-iceo', chip=bpe_iceo_chip, optics=bpe_iceo_optics, fluid_handling_system=fhs)
# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------

# ---------- ----------  STEP 3: LOAD EXAMPLE (TEST) IMAGE COLLECTION ----------  ----------
# initialize
loc = 1
test = (150.0, 2500.0)
run = 2
seq = 1
num_files_in_testset = 200

# load a set of images for testing
testCol.add_img_testset(loc=loc, test=test, run=run, seq=seq)
# get subset of images for compactness
testCol.img_testset.get_subset(num_files=num_files_in_testset)
# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------


# ----------  STEP 4: PERFORM FLAT FIELD CORRECTION AND CALCULATE IMAGE QUALITY ----------  ----------
if apply_flatfield_correction:
    # apply flatfield correction
    testCol.img_testset.apply_flatfield_correction(flatfield=testSetup.optics.microscope.illumination.flatfieldPath,
                                                   darkfield=testSetup.optics.microscope.ccd.darkfield.img)
elif apply_darkfield_correction:
    # apply darkfield correction
    testCol.img_testset.apply_darkfield_correction(darkfield=testSetup.optics.microscope.ccd.darkfield.img)

# calculate image quality
testCol.img_testset.get_img_quality()
# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- -----


# ----------  STEP 6: TEST BACKGROUND SUBTRACTOR ON FILTERED IMAGES ----------  ----------

# apply image processing
testCol.img_testset.apply_image_processing(bpespecs, cropspecs, resizespecs, filterspecs, backsubspecs)

# calculate background
testCol.img_testset.apply_background_subtractor(bg_method=backsubspecs['bg_method'], apply_to='filtered')

# animate background
img_animate = 'filtered'
testCol.img_testset.animate_images(img_animate=img_animate, start=0, stop=200, savePath=join(testCol.dirpath, testCol.dir_bg)) #

# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------
j =1