# test CurlypivSetup
"""
Notes about program
"""

# 1.0 import modules
import numpy as np
from skimage import io
import glob
from os.path import join
import matplotlib.pyplot as plt
from curlypiv.utils.calibrateCamera import measureIlluminationDistributionXY, calculate_depth_of_correlation, calculate_darkfield, plot_field_depth

# 2.0 define class

class CurlypivTestSetup(object):

    def __init__(self, name, chip, optics, fluid_handling_system):
        """
        All the "settings" used in the experimental setup:
        1. chip (class)
            1.1 solid material (class) (e.g. SiO2)
                1.1.1 transparency
                1.1.2 fluorescence spectral characteristics
                1.1.3 surface charge density
                1.1.4 %/vol (here would be 100%)
            1.2 channel (class)
                1.2.1 height
                1.2.2 width
                1.2.3 length
            1.3 reservoir volume
            1.4 electrode configuration (class)
                1.4.1 material
                1.4.2 separation distance
                1.4.3 distance to channel entrance
        2. test solution (class)
            2.1 liquid material (class) (e.g. electrolyte)
                2.1.1 chemical species (e.g. KCl)
                2.1.2 concentration
                2.1.3 measurable quantity (class) (e.g. conductivity)
                    2.1.3.1 theoretical
                    2.1.3.2 measured
                        2.1.3.2.1 measured conductivity
                        2.1.3.2.1 measured date
                2.1.4 measurable quantity (class) (e.g. pH)
                    2.1.4.1 theoretical
                    2.1.4.2 measured
                        2.1.4.2.1 measured conductivity
                        2.1.4.2.1 measured date
            2.2 fluorescent particles (class)
                2.2.0 diameter
                2.2.. measurable quantity (class) (e.g. zeta)
                2.2.. measurable quantity (class) (e.g electrophoretic mobility)
                2.2.. spectral characteristics
                2.2.1 solid materials (class) (e.g. polystyrene)
                    2.2.1.1 %/vol
                2.2.2 liquid materials (class) (e.g. DI water)
                2.2.3 liquid materials (Class) (e.g. sodium azide)
                    2.2.3.1 conductivity
                    2.2.3.2 concentration
        3. illumination (class)
            3.1 source (class)
                3.1.1 type (e.g. Hg lamp)
                3.1.2 intensity
                3.1.3 emission spectra
            3.2 optical element (class) (e.g. excitation filter)
            3.3 optical element (class) (e.g. emission filter)
            3.4 optical element (class) (e.g. dichroic mirror)
        4. microscope
            4.1 type (Olympus iX 73)
            4.2 objective (class)
                4.2.1 numerical aperature (e.g. 0.3)
                4.2.2 magnification (e.g. 20X)
                4.2.3 field of view (e.g. 500 x 500 um)
                4.2.4 depth of focus (e.g 4.1 microns)
        """
        self.name = name
        self.chip = chip
        self.optics = optics
        self.fluid_handling_system = fluid_handling_system

class chip(object):

    def __init__(self, channel=None, bpe=None, reservoir=None, electrodes=None, fluid_handling_system=None,
                 material_in_optical_path=None, thickness_in_optical_path=None):
        """
        Everything important about the chip
        """
        #self.material = material        # deprecated so the channel class can hold this information
        self.channel = channel
        self.bpe = bpe
        self.electrodes = electrodes
        self.fluid_handling_system = fluid_handling_system
        self.material_in_optical_path = material_in_optical_path
        self.thickness_in_optical_path = thickness_in_optical_path

class channel(object):

    def __init__(self, length=None, width=None, height=None,
                 material_bottom_wall_surface=None, material_top_wall_surface=None, material_fluid=None):
        """
        Everything important about the chip
        """
        self.length = length
        self.width = width
        self.height = height
        self.material_bottom_wall_surface = material_bottom_wall_surface  # material should only hold relevant electrokinetic data
        self.material_top_wall_surface = material_top_wall_surface        # material should only hold relevant elect
        self.material_fluid = material_fluid                # could be a mixture of liquid materials + fluorescent particles

class bpe(object):

    def __init__(self, length=None, width=None, height=None, material=None, adhesion_material=None,
                 dielectric_coating=None):
        """
        Everything important about the chip
        """
        self.length = length
        self.linspace_x = np.linspace(-length/2, length/2, num=100)
        self.width = width
        self.height = height
        self.material = material

        if self.material.thickness:
            if self.material.thickness != self.height:
                raise ValueError("BPE height must equal BPE material thickness")

        # adhesion layer used for thin metal film BPE
        self.adhesion_material = adhesion_material

        # dielectric coating on top of BPE
        if dielectric_coating:
            self.dielectric_coating = dielectric_coating
        else:
            self.dielectric_coating = material_solid(name='no_dielectric', permittivity=1, thickness=1e-12,  Ka=6, Kb=2, reaction_site_density=5)

class optics(object):
    def __init__(self, microscope, fluorescent_particles=None, calibration_grid=None, pixel_to_micron_scaling=None):

        self.microscope = microscope
        self.fluorescent_particles = fluorescent_particles
        self.calibration_grid = calibration_grid

        if self.microscope.objective.magnification == 50:
            self.pixel_to_micron_scaling = 0.60      # (microns/pixels)
        elif self.microscope.objective.magnification == 20:
            self.pixel_to_micron_scaling = 1.55      # (microns/pixels)
        else:
            raise ValueError("Unable to determine microns/pixels scaling because objective magnification not 50X or 20X")

        if pixel_to_micron_scaling is not None:
            print("Manual input of pixel_to_micron_scaling is deprecated. A scaling factor of {} um/pix for {} magnification was instantiated.".format(self.pixel_to_micron_scaling, self.microscope.objective.magnification))
        """
        --- I THINK THIS SECTION IS DEPRECATED ---
        Notes: deprecated because calculating the scaling factor or entering it manually is too confusing. I have
        permanently figured out the correct scaling.
        
        if microscope.objective.pixel_to_micron is not None and pixel_to_micron_scaling is None:
            self.pixel_to_micron = microscope.objective.pixel_to_micron
        elif microscope.objective.pixel_to_micron is not None and pixel_to_micron_scaling is not None and microscope.objective.pixel_to_micron != pixel_to_micron_scaling:
            raise ValueError("Conflicting scaling factors: microscope.objective={}, optics={}".format(microscope.objective.pixel_to_micron, pixel_to_micron_scaling))
        elif microscope.objective.pixel_to_micron is None and pixel_to_micron_scaling is not None:
            self.pixel_to_micron = pixel_to_micron_scaling
        """

class illumination(object):

    def __init__(self, basePath=None, source=None, excitation=None, emission=None, dichroic=None, illumination_distribution=None,
                 calculate_illumination_distribution=False,
                 illumPath=None, illumSavePath=None, illumSaveName=None, showIllumPlot=False, save_txt=False, save_plot=False, save_image=False):
        """
        details about the optical setup
        :param source:
        :param excitation:
        :param emission:
        :param dichroic:
        """
        self.basePath = basePath    # this should come from CurlypivTestCollection
        self.source = source
        self.excitation_wavelength = excitation
        self.emission_wavelength = emission
        self.dichroic = dichroic

        if illumination_distribution is not None:
            self.illumination_distribution = illumination_distribution
        elif illumPath is not None:
            flatfield = io.imread(illumPath, plugin='tifffile')
            if len(np.shape(flatfield)) > 2:
                flatfield = np.asarray(np.rint(np.mean(flatfield, axis=0)), dtype='uint16')
            self.illumination_distribution = flatfield
        elif calculate_illumination_distribution and illumination_distribution is None:
            self.illumination_distribution = measureIlluminationDistributionXY(basePath=self.basePath, illumPath=illumPath,
                                                                           show_image=showIllumPlot, save_image=save_image, save_img_type='.tif',
                                                                        save_txt=save_txt, show_plot=showIllumPlot, save_plot=save_plot,
                                                                        savePath=illumSavePath, savename=illumSaveName)
        else:
            self.illumination_distribution = illumination_distribution

        self.flatfield = self.illumination_distribution

        if self.flatfield is not None:
            self.flatfield_mean = np.mean(self.flatfield)
            self.flatfield_std = np.std(self.flatfield)

class darkfield(object):

    def __init__(self, basePath,  darkframePath=None, flip_image_across_axis=None, show_image=False, save_image=False, save_img_type='.tif',
                                      savePath=None, savename=None, save_plot=False):
        """
        details about dark field image

        """
        self.basePath = basePath

        img, mean, std = calculate_darkfield(self.basePath, darkframePath=darkframePath, flip_image_axes=flip_image_across_axis, show_image=show_image, save_image=save_image, save_img_type=save_img_type,
                                      savePath=savePath, savename=savename, save_plot=save_plot)

        self.img = img
        self.mean = mean
        self.std = std

class microscope(object):

    def __init__(self, type, objective, illumination, ccd):
        """
        describes the micrscope setup
        :param type:
        :param objective:
        """
        self.type = type            # e.g. Olympus iX73
        self.objective = objective
        self.illumination = illumination
        self.ccd = ccd

class ccd(object):

    def __init__(self, exposure_time, img_acq_rate, EM_gain, name='iXon Ultra 897', img_acq_type='emcdd', darkfield=None, binning=None,
                 vertical_pixel_shift_speed=0.5e-6, horizontal_pixel_shift_speed=0.1e-6, horizontal_pixel_shift_rate_bits=14,
                 frame_transfer=True, crop_mode=False, acquisition_mode='kinetic', triggering='internal', readout_mode='image',
                 pixels=512, pixel_size=16e-6):
        """
        describe the CCD class
        """
        self.name = name
        self.img_acq_type = img_acq_type

        self.exposure_time = exposure_time
        self.img_acq_rate = img_acq_rate
        self.em_gain = EM_gain
        self.darkfield = darkfield
        self.binning = binning

        # supporting camera acquisition settings
        self.vpss = vertical_pixel_shift_speed
        self.hpss = horizontal_pixel_shift_speed
        self.hpss_bits = horizontal_pixel_shift_rate_bits
        self.frame_transfer = frame_transfer
        self.crop_mode = crop_mode
        self.acquisition_mode = acquisition_mode
        self.triggering = triggering
        self.readout_mode = readout_mode

        if isinstance(pixels, int):
            self.pixels = (pixels, pixels)
        else:
            self.pixels = pixels
        self.pixel_size = pixel_size
        self.image_area = (self.pixels[0]*pixel_size, self.pixels[1]*pixel_size)


class objective(object):

    def __init__(self, fluoro_particle, name=None, numerical_aperture=None, magnification=None, basePath=None, channel_height=None, illumination=None, wavelength=None, microgrid=None, auto_calc_pix_to_micron_scaling=False, pixel_to_micron=None, field_number=None, n0=1, show_depth_plot=False, save_depth_plot=False):
        """

        Objectives in the Pennathur Lab Dark Room uScope:

        20X - LCPlanFL N 20X LCD        [LCPLFLN20xLCD]
            magnification:          20
            numerical_aperture:     0.45
            field_number:           26.5
            working distance:       7.4 - 8.3 mm
            transmittance:          90% @ 425 - 670 nm
            correction collar:      0 - 1.2 mm
            microns per pixel:      1.55
        50X - LCPlanFL N 50x LCD        [LCPLFLN50xLCD]
            magnification:          50
            numerical aperture:     0.7
            field number:           26.5
            working distance:       2.2 - 3 mm
            transmittance:          90% @ 425 - 650 nm
            correction collar:      0 - 1.2 mm
            microns per pixel:      0.6

        Manufacturer website: https://www.olympus-ims.com/en/microscope/lcplfln-lcd/#!cms[focus]=cmsContent11428
        """

        # if name is entered, then pull all the terms directly
        self.name = name

        if name == 'LCPLFLN20xLCD':
            self.magnification = 20
            self.numerical_aperture = 0.45
            self.field_number = 26.5
            self.transmittance = 0.9
            self.pixel_to_micron = 1.55
        elif name == 'LCPLFLN50xLCD':
            self.magnification = 50
            self.numerical_aperture = 0.7
            self.field_number = 26.5
            self.transmittance = 0.9
            self.pixel_to_micron = 0.6
        else:
            self.numerical_aperture = numerical_aperture
            self.magnification = magnification
            self.field_number = field_number

        # general terms
        self._illumination = illumination
        if self._illumination is not None:
            self._wavelength = self._illumination.emission_wavelength
        elif wavelength is not None:
            self._wavelength = wavelength
        else:
            raise ValueError("A wavelength is required via the <illumination> class or <wavelength> input parameter")
        self._pd = fluoro_particle.diameter
        self._n0 = n0
        self.calculate_depth_of_field()
        self.calculate_depth_of_correlation()

        if field_number:
            self.calculate_field_of_view()

        if show_depth_plot or save_depth_plot:
            plot_field_depth(depth_of_corr=self.depth_of_correlation, depth_of_field=self.depth_of_field, show_depth_plot=show_depth_plot, save_depth_plot=save_depth_plot,
                                 basePath=basePath, savename=None, channel_height=channel_height, objective=magnification)

        # grids and scaling factors
        if auto_calc_pix_to_micron_scaling and self.pixel_to_micron is None:
            self.microgrid = microgrid
            self.calculate_pixel_to_micron_scaling()


    def calculate_field_of_view(self):
        self.field_of_view = self.field_number / self.magnification

    def calculate_depth_of_field(self, e=16e-6, n=1):
        """
        e: CCD pixel resolution     example: e = 16 um (16 microns is the pixel size)
        """
        self.depth_of_field = self._wavelength*n/self.numerical_aperture**2+e*n/(self.magnification*self.numerical_aperture)

    def calculate_depth_of_correlation(self, eps=0.01):
        # step 0: define
        n = self._n0
        dp = self._pd
        NA = self.numerical_aperture
        M = self.magnification
        lmbda = self._wavelength

        # step 1: calculate the depth of correlation for the optical setup
        depth_of_correlation = calculate_depth_of_correlation(M=M, NA=NA, dp=dp, n=n, lmbda=lmbda, eps=eps)

        self.depth_of_correlation = depth_of_correlation

    def calculate_pixel_to_micron_scaling(self):
        if self.microgrid is None:
            raise ValueError("Need objective.microgrid property in order to calculate scaling factor")
        # script to calculate scaling factor from grid
            # would go here

    @property
    def NA(self):
        return self.numerical_aperture

    @property
    def M(self):
        return self.magnification

class microgrid(object):

    def __init__(self, gridPath=None, center_to_center_spacing=None, feature_width=None, grid_type='grid', show_grid=False):
        """
        this class holds images for the microgrid and performs pixel to micron scaling calculations
        """
        if gridPath is not None:
            self.gridPath = gridPath
            self.spacing = center_to_center_spacing
            self.width = feature_width
            self.grid_type = grid_type

            # find files in directory
            file_list = glob.glob(join(self.gridPath, 'grid*.tif'))

            if len(file_list) < 1:
                raise ValueError("No grid*.tif files found in {}".format(self.gridPath))

            img_grid = np.zeros(shape=(512,512))
            for f in file_list:
                img = io.imread(f, plugin='tifffile')
                if len(np.shape(img)) > 2:
                    img = np.mean(img, axis=0)
                img_grid += img

            img_grid = img_grid / len(file_list)

            self.img_grid = img_grid

            if show_grid is True:
                fig, ax = plt.subplots()
                ax.imshow(img_grid, cmap='gray')

                ax.set_xlabel('pixels')
                ax.set_ylabel('pixels')
                plt.title('grid: 10 um Lines; 50 um Spacing')
                plt.show()


class fluorescent_particles(object):

    def __init__(self, name=None, materials=None,diameter=None,fluorescence_spectra=None, concentration=None,
                 electrophoretic_mobility=None, zeta=None):
        """
        the details of the fluroescent particles used
        :param materials:
        :param diameter:
        :param fluorescence_spectra:
        :param concentration:
        :param electrophoretic_mobility:
        :param zeta:
        """

        self.name = name
        self.materials=materials
        self.concentration=concentration
        self.electrophoretic_mobility=electrophoretic_mobility
        self.zeta=zeta
        self.diameter=diameter
        if diameter:
            k_b = 1.3806e-23
            T=298
            mu=0.001
            self.diffusivity = k_b*T/(6*np.pi*mu*diameter/2)

        self.fluorescence_spectra=fluorescence_spectra


class reservoir(object):

    def __init__(self, diameter, height, height_of_reservoir=None, material=None):
        """
        describes the micrscope setup
        :param type:
        :param objective:
        """
        g = 9.81 # m/s**2

        self.material = material
        self.diameter = diameter
        self.height = height
        self.volume = np.pi*self.diameter**2/4
        self.height_of_reservoir = height_of_reservoir
        if material and height_of_reservoir:
            self.hydrostatic_pressure = material.density*g*self.height_of_reservoir

class fluid_handling_system(object):

    def __init__(self, fluid_reservoir=None, all_tubing=None, onchip_reservoir=None):
        """
        describes the fluid handling system
        """
        self.fluid_reservoir=fluid_reservoir
        self.all_tubing = all_tubing
        self.onchip_reservoir = onchip_reservoir

class tubing(object):

    def __init__(self, inner_diameter=None, length=None, material=None):
        """
        describes each segment of tubing

        """
        self.inner_diameter = inner_diameter
        self.length = length
        self.material = material

class optical_element(object):

    def __init__(self, passing_wavelengths=None, reflectivity=None):
        """
        this class describes the optical characteristics of any material or element
        :param wavelength_bandpass:
        """
        self.passing_wavelengths=passing_wavelengths
        self.reflectivity=reflectivity

class measurable_quantity(object):

    def __init__(self, reference_value=None, measured_value=None):
        """
        what value was measured and when
        """
        self.reference_value = reference_value
        self.measured_value = measured_value

class measurement(object):

    def __init__(self, value=None, date=None):
        """
        Object for storing measurements
        :param value:
        :param date:
        """
        self.value = value
        self.date = date

class electrode_configuration(object):

    def __init__(self, material=None, length=None, entrance_length=None):
        """
        Object for holding electrode configuration details
        :param material:
        :param length:
        :param entrance_length:
        """
        self.material = material
        self.length = length
        self.entrance_length = entrance_length

class material_solid(object):

    def __init__(self,  name=None, zeta=None, concentration=None, index_of_refraction=None, transparency=None, fluorescence_spectra=None,
                 permittivity=None, conductivity=None, thickness=None, youngs_modulus=None, poissons_ratio=None,
                 density=None, dielectric_strength=None, reaction_site_density=None, Ka=None, Kb=None, width=None, length=None):
        """
        everything about a material
        :param transparency:
        :param fluorescence_spectra:
        :param zeta:
        """
        # identity
        self.name = name

        # geometry
        self.length = length
        self.width = width
        self.thickness = thickness

        # mechanical
        self.density = density
        self.concentration = concentration      # For a solid, this is % by volume.
        self.youngs_modulus = youngs_modulus
        self.poissons_ratio = poissons_ratio

        # optical
        self.index_of_refraction = index_of_refraction
        self.fluorescence_spectra = fluorescence_spectra
        self.transparency = transparency
        if self.transparency:
            self.reflectivity = 1 / self.transparency

        # electrochemical
        self.conductivity = conductivity
        if permittivity:
            self.permittivity = permittivity
        self.zeta = zeta
        self.dielectric_strength = dielectric_strength
        if reaction_site_density:
            self.reaction_site_density = reaction_site_density*1e18     # (#/nm2) surface density of reaction sites: accepts nm2 and converts to m2 (see Squires)
        self.Ka = Ka                                                    # reaction equilibrium constant - upper bound
        self.Kb = Kb                                                    # reaction equilibrium constant - lower bound

class material_liquid(object):

    def __init__(self, name=None, species=None, concentration=None, conductivity=None, pH=None, density=None, viscosity=None,
                 permittivity=None, temperature=None, valence=1.0):
        """
        everything about a liquid
        :param species:
        :param concentration:
        :param conductivity:
        :param pH:
        """
        # identity
        self.name = name

        # electro/chemical
        self.species = species
        self.concentration = concentration      # (mmol) = (mmol/L) = (mol/m3)
        self.conductivity = conductivity
        if permittivity:
            self.permittivity = permittivity
        if pH:
            self.pH = pH
            self.c_H = 10**-pH * 1e3          # (mmol) = (mmol/L) = (mol/m3); (concentration of Hydrogen ions (H+)
        self.valence = valence

        # mechanical
        self.density = density
        self.viscosity = viscosity
        self.temperature = temperature
        self.diffusivity = 2e-9                 # (m^2/s) Diffusivity of KCl in DI water [Soni]