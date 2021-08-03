from curlypiv.CurlypivTestSetup import material_liquid, material_solid, bpe, channel, fluorescent_particles, ccd, \
    objective, microscope, illumination
from curlypiv.datasets import BpeIceoDataset
from curlypiv.metrics import BpeIceoDatasetMetric


# low level materials
sio2_channel = material_solid(name='SiO2', zeta=-0.0826, permittivity=4.6, index_of_refraction=1.5,
                              conductivity=1e-18, thickness=500e-6, youngs_modulus=71.7e9, poissons_ratio=0.17,
                              density=2.203e3,
                              dielectric_strength=470e6)  # Ref: 2/13/21, ZuPIV of SiO2 w/ 100 uM KCl
pdms_channel = material_solid(name='PDMS', zeta=-0.005, permittivity=2.5, index_of_refraction=1.4,
                              conductivity=4e-13, thickness=2e-3, youngs_modulus=500e3, poissons_ratio=0.5,
                              density=0.97,
                              dielectric_strength=20e6)  # Ref: <http://www.mit.edu/~6.777/matprops/pdms.htm>
sio2_chip = material_solid(name='SiO2', transparency=0.92, permittivity=4.6, index_of_refraction=1.5,
                           conductivity=1e-18, thickness=500e-6, youngs_modulus=71.7e9, poissons_ratio=0.17,
                           density=2.203e3, dielectric_strength=470e6)  # Ref: None
gold_bpe = material_solid(name='Au', transparency=0.5,
                          conductivity=22e-9)  # Ref: 4/6/21, 30 nm Au - 75% @ 50X, 51% @ 20X
NO_DIELECTRIC = material_solid(name='SiO2', zeta=-0.08, permittivity=4.6, index_of_refraction=1.5,
                               conductivity=1e-18, thickness=1e-10, youngs_modulus=71.7e9, poissons_ratio=0.17,
                               density=2.203e3, dielectric_strength=470e6, Ka=6, Kb=2,
                               reaction_site_density=5)  # Ref: 2/13/21, ZuPIV of SiO2 w/ 100 uM KCl
alkane_thiol = material_solid(name='Alkane-thiol', thickness=1e-10, transparency=0.99)  # Ref: None
polystyrene = material_solid(name='Polystyrene', transparency=0.9, zeta=-0.045, index_of_refraction=1.59,
                             density=1.05e3, permittivity=2.6)  # Ref: 3/4/21 BNL Zetasizer
kcl = material_liquid(name='KCl', species='KCl', conductivity=18.6e-4, concentration=0.1, pH=6.3, density=1000,
                      viscosity=0.001, permittivity=80.1,
                      temperature=298)  # Ref: 3/4/21 BNL Conductivity Meter + Zetasizer
# components
fluoro_particles = fluorescent_particles(diameter=180e-9, concentration=2e-5, materials=polystyrene,
                                         electrophoretic_mobility=-9e-8)  # Needs to be measured. I used 3X the 800 nm particles.
bpe_iceo_channel = channel(length=20e-3, width=550e-6, height=39.0e-6, material_bottom_wall_surface=sio2_channel,
                           material_top_wall_surface=pdms_channel,
                           material_fluid=kcl)  # Ref: 3/11/21, NCF Dektak of PDMS channel height
BPE = bpe(length=500e-6, width=500e-6, height=20e-9, material=gold_bpe, adhesion_material=alkane_thiol,
          dielectric_coating=NO_DIELECTRIC)  # Ref: 4/6/21, Brightfield BPE images + NCF Dektak
emccd = ccd(exposure_time=.040, img_acq_rate=24.83, EM_gain=5, darkfield=None)
flatfield = illumination(basePath=None, source='Hg', excitation=490e-9, emission=525e-9,
                         calculate_illumination_distribution=False, illumPath=None,
                         showIllumPlot=False, save_plot=False, save_image=False,
                         save_txt=False)
x20_objective = objective(name='LCPLFLN20xLCD', fluoro_particle=fluoro_particles, illumination=flatfield,
                          basePath=None, channel_height=bpe_iceo_channel.height,
                          show_depth_plot=None, save_depth_plot=None)
microscope = microscope(type='Olympus iX73', objective=x20_objective, illumination=flatfield, ccd=emccd)

# test BpeIceoDataset
standardDataset = {
    "channel": bpe_iceo_channel,
    "buffer": kcl,
    "microscope": microscope,
    "ccd": emccd,
    "fluorescent_particles": fluoro_particles,
    "metadata": {"notes": 'everything went ok'}
}

bpeiceoDataset = BpeIceoDataset(bpe=BPE, dielectric_coating=NO_DIELECTRIC, **standardDataset)

bpeiceoMetrics = BpeIceoDatasetMetric(dataset=bpeiceoDataset, electric_field_strength=10e3, frequency=100)

j=1