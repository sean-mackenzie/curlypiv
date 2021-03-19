# test CurlypivSetup
"""
Notes about program
"""

# 1.0 import modules


# 2.0 define class

class CurlypivSetup(object):

    def __init__(self):
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
        self.chip = None
        self.test_solution = None
        self.illumination = None
        self.microscope = None
        self.particles = None

class chip(object):

    def __init__(self, material=None,channel=None, reservoir=None,electrodes=None):
        """
        Everything important about the chip
        """
        self.material = material
        self.channel = channel
        self.reservoir = reservoir
        self.electrodes = electrodes

class material_solid(object):

    def __init__(self, concentration=None, transparency=None, fluorescence_spectra=None,zeta=None):
        """
        everything about a material
        :param transparency:
        :param fluorescence_spectra:
        :param zeta:
        """
        self.concentration = concentration # For a solid, this is % by volume.
        self.transparency = transparency
        self.fluorescence_spectra = fluorescence_spectra
        self.zeta = zeta

class material_liquid(object):

    def __init__(self, species=None, concentration=None, conductivity=None, pH=None):
        """
        everything about a liquid
        :param species:
        :param concentration:
        :param conductivity:
        :param pH:
        """
        self.species = species
        self.concentration = concentration
        self.conductivity = conductivity
        self.pH = pH

class fluorescent_particles(object):

    def __init__(self, materials=None,diameter=None,fluorescence_spectra=None, concentration=None,
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
        self.materials=materials
        self.diameter=diameter
        self.fluorescence_spectra=fluorescence_spectra
        self.concentration=concentration
        self.electrophoretic_mobility=electrophoretic_mobility
        self.zeta=zeta

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

class illumination(object):

    def __init__(self, source=None, excitation=None, emission=None,dichroic=None):
        """
        details about the optical setup
        :param source:
        :param excitation:
        :param emission:
        :param dichroic:
        """
        self.source=source
        self.excitation=excitation
        self.emission=emission
        self.dichroic=dichroic

class optical_element(object):

    def __init__(self, passing_wavelengths=None, reflectivity=None):
        """
        this class describes the optical characteristics of any material or element
        :param wavelength_bandpass:
        """
        self.passing_wavelengths=passing_wavelengths
        self.reflectivity=reflectivity

class microscope(object):

    def __init__(self, type=None, objective=None):
        """
        describes the micrscope setup
        :param type:
        :param objective:
        """
        self.type = type            # e.g. Olympus iX73
        self.objective = objective

class objective(object):

    def __init__(self, numerical_aperture=None, magnification=None,field_of_view=None,depth_of_focus=None):
        self.numerical_aperture = numerical_aperture
        self.magnification = magnification
        self.field_of_view = field_of_view
        self.depth_of_focus = depth_of_focus

