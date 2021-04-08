"""
Notes about the program
"""


# ---------- ----------  STEP 1: LOAD TEST COLLECTION ----------  ---------- ---------- ---------- ---------- ----------

# import modules
from curlypiv.CurlypivTestCollection import CurlypivRun, CurlypivTest, CurlypivTestCollection

# load test files
name = 'testCol'
base_path = '/Users/mackenzie/Desktop/iceo-analysis'
img_type = '.tif'
testid = ('E','Vmm')
runid = ('run', 'num')
seqid = ('test_', '_X')
frameid = '_X'

# load calibration files
gridPath = None
illumPath = None
camNoisePath = None

# instantiate CurlypivTestCollection class
testCol = CurlypivTestCollection(name, base_path, file_type=img_type, testid=testid, runid=runid, seqid=seqid, frameid=frameid, load_files=True,
                            calibration_grid_path=gridPath, calibration_illum_path=illumPath, calibration_camnoise_path=camNoisePath,
                              )

# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------



# ---------- ----------  STEP 2: LOAD TEST SETUP ----------  ----------

# import modules
from curlypiv.CurlypivTestSetup import CurlypivTestSetup, chip, material_liquid, material_solid, fluorescent_particles
from curlypiv.CurlypivTestSetup import illumination, electrode_configuration, microscope, objective, channel, bpe, darkfield
from curlypiv.CurlypivTestSetup import reservoir, optics, fluid_handling_system, tubing, ccd, microgrid

# determine actions
# grid
grid_show=False
# illumination / flatfield
calculateIllumDistribution=True
illum_save_txt = False
illum_save_plot = False
illum_save_image = False
illum_show_plot = False
# darkfield
darkfield_show_image = False
darkfield_save_image = False
darkfield_save_plot = False
# objective
obj_show_depth_plot = False
obj_save_depth_plot = False

# instantiate CurlypivTestSetup class
# low level materials
sio2_channel = material_solid(zeta=-0.0826)         # Ref: 2/13/21, ZuPIV of SiO2 w/ 100 uM KCl
pdms_channel = material_solid(zeta=-0.01)           # Ref: NONE
sio2_chip = material_solid(transparency=0.99, index_of_refraction=1.45) # Ref: None
gold_bpe = material_solid(transparency=0.75)        # Ref: 4/6/21, 30 nm Au - 75% @ 50X, 51% @ 20X
alkane_thiol = material_solid(transparency=0.99)    # Ref: None
polystyrene = material_solid(transparency=0.9, zeta=-0.045, index_of_refraction=1.5) # Ref: 3/4/21 BNL Zetasizer
tygon = material_solid(zeta=-0.001)                 # Ref: None
kcl = material_liquid(species='KCl', conductivity=18.6e-4, concentration=0.1, pH=6.3, density=1000, viscosity=0.00089) # Ref: 3/4/21 BNL Conductivity Meter + Zetasizer
# fluidic
bpe_iceo_reservoir = reservoir(diameter=2e-3, height=2e-3, height_of_reservoir=0, material=kcl)
fhs_reservoir = reservoir(diameter=20e-3, height=20e-3, height_of_reservoir=50e-3, material=kcl)
tubing = tubing(inner_diameter=0.5e-3, length=100e-3, material=tygon)
fhs = fluid_handling_system(fluid_reservoir=bpe_iceo_reservoir, all_tubing=tubing, onchip_reservoir=bpe_iceo_reservoir)
# physical
fluoro_particles = fluorescent_particles(diameter=500e-9, concentration=2e-5, materials=polystyrene, electrophoretic_mobility=-3.531e-8)    # Ref: 3/4/21 BNL Zetasizer
bpe_iceo_channel = channel(length=25e-3, width=500e-6, height=15e-6, material_wall_surface=sio2_channel, material_fluid=kcl)                # Ref: 3/11/21, NCF Dektak of PDMS channel height
bpe_iceo_bpe = bpe(length=250e-6, width=500e-6, height=30e-9, material=gold_bpe, adhesion_material=alkane_thiol)                            # Ref: 4/6/21, Brightfield BPE images + NCF Dektak
bpe_iceo_electrode_config = electrode_configuration(material='Stainless Steel', length=bpe_iceo_channel.length, entrance_length=1e-3)
# optics
microgrid_100um = microgrid(gridPath=testCol.grid_path, center_to_center_spacing=50e-6, feature_width=10e-6, grid_type='grid', show_grid=grid_show)
darkfield = darkfield(basePath=testCol.dirpath, show_image=darkfield_show_image, save_image=darkfield_save_image, save_plot=darkfield_save_plot)
emccd = ccd(exposure_time=.035, img_acq_rate=28.353, EM_gain=300, darkfield=darkfield)
illumination = illumination(basePath=testCol.dirpath, source='Hg', excitation=560e-9, emission=595e-9, calculate_illumination_distribution=calculateIllumDistribution, illumPath=testCol.illum_path, showIllumPlot=illum_show_plot, save_plot=illum_save_plot, save_image=illum_save_image, save_txt=illum_save_txt)
x50_objective = objective(numerical_aperture=0.7, magnification=50, field_number=26.5e-3, fluoro_particle=fluoro_particles, illumination=illumination, pixel_to_micron=6.04e-7, basePath=testCol.dirpath, channel_height=bpe_iceo_channel.height, show_depth_plot=obj_show_depth_plot, save_depth_plot=obj_save_depth_plot) # Ref: 4/6/21, Brightfield grid @ 50X
microscope = microscope(type='Olympus iX73', objective=x50_objective, illumination=illumination, ccd=emccd)
# higher-level
bpe_iceo_optics = optics(microscope=microscope, fluorescent_particles=fluoro_particles, calibration_grid=microgrid_100um, pixel_to_micron_scaling=x50_objective.pixel_to_micron)
bpe_iceo_chip = chip(channel=bpe_iceo_channel, material=sio2_chip, bpe=bpe_iceo_bpe, reservoir=bpe_iceo_reservoir, electrodes=bpe_iceo_electrode_config, material_in_optical_path=sio2_chip, thickness_in_optical_path=500e-6)
# test Setup Class
testSetup = CurlypivTestSetup(name='bpe-iceo', chip=bpe_iceo_chip, optics=bpe_iceo_optics, fluid_handling_system=fhs)

# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------


# ---------- ----------  STEP 3: LOAD EXAMPLE (TEST) IMAGE COLLECTION ----------  ----------

# import modules

# initialize
loc = 4
test = 6.0
run = 1
seq = 1
num_files_in_testset = 200

# load a set of images for testing
testCol.add_img_testset(loc=loc, test=test, run=run, seq=seq)

# get subset of images for compactness
testCol.img_testset.get_subset(num_files=num_files_in_testset)

# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------


# ----------  STEP 4: PERFORM FLAT FIELD CORRECTION AND CALCULATE IMAGE QUALITY ----------  ----------

# import modules

# initialize
""" 
STILL UNDER DEVELOPMENT - NOT WORKING 

# apply flatfield correction
testCol.img_testset.apply_flatfield_correction(flatfield=testSetup.optics.microscope.illumination.flatfield,
                                               darkfield=testSetup.optics.microscope.ccd.darkfield.img)

# calculate image quality
testCol.img_testset.get_img_quality()

# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------


# ----------  STEP 5: TEST BACKGROUND SUBTRACTOR ON RAW IMAGES ----------  ----------

# import modules
from os.path import join

# calculate background
testCol.img_testset.apply_background_subtractor(bg_method='KNN', apply_to='raw')

# animate background
testCol.img_testset.animate_background_subtractor(start=0, stop=200, savePath=join(testCol.dirpath,testCol.dir_bg))

"""
# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------


# ----------  STEP 5: TEST BACKGROUND SUBTRACTOR ON FILTERED IMAGES ----------  ----------

# import modules
from os.path import join

# initialize
cropspecs = {
    'xmin': 70,  # x = 0 is the left of the image
    'xmax': 450,
    'ymin': 256,
    'ymax': 480  # y = 0 is the bottom of the image
}

filterspecs = {
    'denoise_wavelet': {'args': [], 'kwargs': dict(method='BayesShrink', mode='soft', rescale_sigma=True)},
    'rescale_intensity': {'args': [(96, 99.9999), ('dtype')]}
}

resizespecs = {
        'method': 'rescale',
        'scale': 2
}

backsubspecs = {
        'bg_method': 'KNN'
}

# apply image processing
testCol.img_testset.apply_image_processing(cropspecs, resizespecs, filterspecs, backsubspecs)

# calculate background
testCol.img_testset.apply_background_subtractor(bg_method='KNN', apply_to='filtered')

# animate background
testCol.img_testset.animate_background_subtractor(img='filtered', start=0, stop=200, savePath=join(testCol.dirpath,testCol.dir_bg)) #



# import modules
print('ha')








