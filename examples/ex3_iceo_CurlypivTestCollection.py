"""
Notes about the program
"""

# -------- import modules -----------
# import modules
from os.path import join
from skimage.morphology import disk
import numpy as np
import pandas as pd

from curlypiv.CurlypivTestSetup import CurlypivTestSetup, chip, material_liquid, material_solid, fluorescent_particles
from curlypiv.CurlypivTestSetup import illumination, electrode_configuration, microscope, objective, channel, bpe, darkfield
from curlypiv.CurlypivTestSetup import reservoir, optics, fluid_handling_system, tubing, ccd, microgrid
from curlypiv.CurlypivTestCollection import CurlypivRun, CurlypivTest, CurlypivTestCollection
from curlypiv.CurlypivPIV import CurlypivPIV
from curlypiv.CurlypivPIVSetup import CurlypivPIVSetup
from curlypiv.utils.calculate_ICEO import calculate_ICEO

# ---------- ----------  STEP 1: LOAD TEST COLLECTION and TEST SETUP settings ----------  ---------- ---------- ---------- ---------- ----------

# --- Load file paths ---
dirPath = '/Users/mackenzie/Desktop/05.10.21-iceo-test'
gridPath = join(dirPath, 'setup/calibration/microgrid/grid_10umLines_50umSpacing/20X')
flatfieldPath = None #join(dirPath, 'setup/calibration/illumination/flatfield.tif')
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
# iceo stats
plot_iceo = False
# examine testset
examine_testset = False
examine_testset_raw = False
img_animate = 'filtered'
# piv
img_piv = 'filtered'
window_size1 = 96
window_size2 = 48
piv_init_frame = 2
backSub_init_frames = 2
piv_num_analysis_frames = 10
save_text=False
vectors_on_image=True
show_plot=False
save_plot=False
save_u_mean_plot=True
img_piv_plot = img_piv
save_plot_path = join(dirPath, 'results', 'piv-plots')

# low level materials
sio2_channel = material_solid(name='SiO2', zeta=-0.0826, permittivity=4.6, index_of_refraction=1.5, conductivity=1e-18, thickness=500e-6, youngs_modulus=71.7e9, poissons_ratio=0.17, density=2.203e3, dielectric_strength=470e6)         # Ref: 2/13/21, ZuPIV of SiO2 w/ 100 uM KCl
pdms_channel = material_solid(name='PDMS', zeta=-0.005, permittivity=2.5, index_of_refraction=1.4, conductivity=4e-13, thickness=3e-3, youngs_modulus=500e3, poissons_ratio=0.5, density=0.97, dielectric_strength=20e6)                  # Ref: <http://www.mit.edu/~6.777/matprops/pdms.htm>
sio2_chip = material_solid(name='SiO2', transparency=0.99, permittivity=4.6, index_of_refraction=1.5, conductivity=1e-18, thickness=500e-6, youngs_modulus=71.7e9, poissons_ratio=0.17, density=2.203e3, dielectric_strength=470e6)       # Ref: None
gold_bpe = material_solid(name='Au', transparency=0.5, conductivity=22e-9)        # Ref: 4/6/21, 30 nm Au - 75% @ 50X, 51% @ 20X
sio2_dielectric = material_solid(name='SiO2', zeta=-0.08, permittivity=4.6, index_of_refraction=1.5, conductivity=1e-18, thickness=3e-9, youngs_modulus=71.7e9, poissons_ratio=0.17, density=2.203e3, dielectric_strength=470e6, Ka=6, reaction_site_density=5)          # Ref: 2/13/21, ZuPIV of SiO2 w/ 100 uM KCl
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
fluoro_particles = fluorescent_particles(diameter=800e-9, concentration=2e-5, materials=polystyrene, electrophoretic_mobility=-3.531e-8)    # Ref: 3/4/21 BNL Zetasizer
bpe_iceo_channel = channel(length=20e-3, width=500e-6, height=16.5e-6, material_wall_surface=sio2_channel, material_fluid=kcl)                # Ref: 3/11/21, NCF Dektak of PDMS channel height
bpe_iceo_bpe = bpe(length=100e-6, width=500e-6, height=20e-9, material=gold_bpe, adhesion_material=alkane_thiol, dielectric_coating=sio2_dielectric)                            # Ref: 4/6/21, Brightfield BPE images + NCF Dektak
bpe_iceo_electrode_config = electrode_configuration(material='Stainless Steel', length=bpe_iceo_channel.length, entrance_length=1e-3)
# optics
microgrid_100um = microgrid(gridPath=gridPath, center_to_center_spacing=50e-6, feature_width=10e-6, grid_type='grid', show_grid=grid_show)
darkfield = darkfield(basePath=dirPath, show_image=darkfield_show_image, save_image=darkfield_save_image, save_plot=darkfield_save_plot)
emccd = ccd(exposure_time=.040, img_acq_rate=24.826, EM_gain=8, darkfield=darkfield)
flatfield = illumination(basePath=dirPath, source='Hg', excitation=490e-9, emission=525e-9, calculate_illumination_distribution=calculateIllumDistribution, illumPath=flatfieldPath, showIllumPlot=illum_show_plot, save_plot=illum_save_plot, save_image=illum_save_image, save_txt=illum_save_txt)
x20_objective = objective(numerical_aperture=0.45, magnification=50, field_number=26.5e-3, fluoro_particle=fluoro_particles, illumination=flatfield, pixel_to_micron=1.533, basePath=dirPath, channel_height=bpe_iceo_channel.height, show_depth_plot=obj_show_depth_plot, save_depth_plot=obj_save_depth_plot) # Ref: 4/6/21, Brightfield grid @ 20X
microscope = microscope(type='Olympus iX73', objective=x20_objective, illumination=flatfield, ccd=emccd)
# higher-level
bpe_iceo_optics = optics(microscope=microscope, fluorescent_particles=fluoro_particles, calibration_grid=microgrid_100um, pixel_to_micron_scaling=x20_objective.pixel_to_micron)
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
    # NOTE: The BPE coordinates need to be significantly within the image crop coordinates otherwise will ERROR.
    'bxmin': 175,  # x = 0 is the left of the image
    'bxmax': 230,
    'bymin': 50,
    'bymax': 285,  # y = 0 is the bottom of the image
    'multiplier': 1.5
}
cropspecs = {
    'xmin': 125,  # x = 0 is the left of the image
    'xmax': 280,
    'ymin': 45,
    'ymax': 290  # y = 0 is the bottom of the image
}
filterspecs = {
    'denoise_wavelet': {'args': [], 'kwargs': dict(method='BayesShrink', mode='soft', rescale_sigma=True)},
    'median': {'args': [disk(3)]},
    'gaussian': {'args': [1.5]},
    'equalize_adapthist': {'args': [], 'kwargs': dict(kernel_size=int(np.round(bpe_iceo_bpe.length*1e6/x20_objective.pixel_to_micron/7)))},
    'rescale_intensity': {'args': [(0, 99.99), ('dtype')]} # 70, 99.999
}
resizespecs = {
    'method': 'pyramid_expand',
    'scale': 2
}
backsubspecs = {
    'bg_method': 'min',
    'darkfield': darkfield.img,
    'flatfield': flatfield.flatfield
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

# --- calculate ICEO ---
iceo_stats, header = calculate_ICEO(testSetup=testSetup, testCol=testCol, plot_figs=plot_iceo, savePath=None)
header = header.split(',')


# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------

if examine_testset:
    # ---------- ----------  STEP 3: LOAD EXAMPLE (TEST) IMAGE COLLECTION ----------  ----------

    # import modules

    # initialize
    loc = 1
    test = (200.0, 250.0)
    run = 1
    seq = 1
    num_files_in_testset = 50

    # load a set of images for testing
    testCol.add_img_testset(loc=loc, test=test, run=run, seq=seq)

    # get subset of images for compactness
    testCol.img_testset.get_subset(num_files=num_files_in_testset)

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------


    # ----------  STEP 4: PERFORM FLAT FIELD CORRECTION AND CALCULATE IMAGE QUALITY ----------  ----------
    if apply_flatfield_correction:
        # apply flatfield correction
        testCol.img_testset.apply_flatfield_correction(flatfield=testSetup.optics.microscope.flatfield.flatfieldPath,
                                                       darkfield=testSetup.optics.microscope.ccd.darkfield.img)
    elif apply_darkfield_correction:
        # apply darkfield correction
        testCol.img_testset.apply_darkfield_correction(darkfield=testSetup.optics.microscope.ccd.darkfield.img)

    # calculate image quality
    testCol.img_testset.get_img_quality()

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------

    if examine_testset_raw:
        # ----------  STEP 5: TEST BACKGROUND SUBTRACTOR ON RAW IMAGES ----------  ----------

        # calculate background
        testCol.img_testset.apply_background_subtractor(bg_method='KNN', apply_to='raw')

        # animate background
        testCol.img_testset.animate_images(start=0, stop=50, savePath=join(testCol.dirpath, testCol.dir_bg))

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------


    # ----------  STEP 6: TEST BACKGROUND SUBTRACTOR ON FILTERED IMAGES ----------  ----------

    # initialize

    # apply image processing
    testCol.img_testset.apply_image_processing(bpespecs, cropspecs, resizespecs, filterspecs, backsubspecs)

    # calculate background
    testCol.img_testset.apply_background_subtractor(bg_method=backsubspecs['bg_method'], apply_to='filtered')

    # animate background
    img_animate = 'bgs'
    testCol.img_testset.animate_images(img_animate=img_animate, start=0, stop=50, savePath=join(testCol.dirpath, testCol.dir_bg)) #

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- --------


# ----------  STEP 5: TEST PIV ON FULL DATASET FILTERED IMAGES ----------  ----------

pivSetup = CurlypivPIVSetup(name='testPIV', save_text=save_text, save_plot=save_plot, show_plot=show_plot,
                            save_plot_path=save_plot_path, vectors_on_image=vectors_on_image,
                            testCollection=testCol, testSetup=testSetup, win1=window_size1, win2=window_size2,
                            calculate_zeta=False, replace_Nans_with_zeros=True, save_u_mean_plot=save_u_mean_plot)

piv = CurlypivPIV(testCollection=testCol, img_piv=img_piv, img_piv_plot=img_piv_plot, testSetup=testSetup,
                  pivSetup=pivSetup, bpespecs=bpespecs, cropspecs=cropspecs, filterspecs=filterspecs, resizespecs=resizespecs,
                  backsubspecs=backsubspecs, init_frame=piv_init_frame, num_analysis_frames=piv_num_analysis_frames, backSub_init_frames=backSub_init_frames,
                  piv_mask='bpe')


piv.piv_analysis(level='all')


# import modules


