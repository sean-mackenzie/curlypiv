# test CurlypivSetup
"""
Notes about program
"""

# 1.0 import modules
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import matplotlib.image as mgimg
from matplotlib import animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
matplotlib.rcParams['figure.figsize'] = (7, 6)
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=16, weight='bold')
font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']

# OpenPIV
# ----- imports for OpenPIV -----
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/mackenzie/PythonProjects/openpiv')
import openpiv.piv
from openpiv import windef
from openpiv.windef import Settings


# 2.0 define class

class CurlypivPIVSetup(object):

    def __init__(self, name, save_text, save_plot, testCollection, testSetup,
                 win1=128, win2=64, show_plot=False, save_plot_path=None, vectors_on_image=True,
                 calculate_zeta=False, replace_Nans_with_zeros=True, save_u_mean_plot=False):
        """
        Notes
        """

        # setup
        self._name = name
        self.save_text = save_text
        self.save_plot = save_plot
        self.save_u_mean_plot = save_u_mean_plot
        self.save_plot_path = save_plot_path
        self.show_plot = show_plot
        self.calculate_zeta = calculate_zeta

        # OpenPIV
        self.settings = Settings()

        # plotting
        self.vectors_on_image = vectors_on_image
        self.settings.scale_plot = 1
        self.colorMap = 'plasma'
        self.colorNorm = colors.Normalize(vmin=0, vmax=50)
        self.alpha = 1
        self.scalebar_microns = 25 # units are microns
        self.dpi = 200

        # camera
        self.img_acq = testSetup.optics.microscope.ccd.img_acq_rate
        self.dtf = 1/self.img_acq
        self.pixels_to_microns = testSetup.optics.microscope.objective.pixel_to_micron
        self.pix_per_um = 1/self.pixels_to_microns

        # experimental
        self.E_max = 10e3
        self.particle_diameter = testSetup.optics.fluorescent_particles.diameter
        self.est_zeta = testSetup.chip.channel.material_wall_surface.zeta

        # scientific
        self.epsr = 80
        self.eps = self.epsr*8.854e-12
        self.mu = testSetup.chip.channel.material_fluid.viscosity

        # outputs
        self.est_u_eof = self.eps*self.est_zeta*self.E_max/self.mu*1e6
        self.char_u_eof = -self.est_u_eof*self.pix_per_um*self.dtf
        self.char_u = int(np.round(self.char_u_eof))

        # more OpenPIV
        self.settings.correlation_method = 'linear'
        self.settings.normalized_correlation = True
        self.settings.deformation_method = 'symmetric'  # 'symmetric' or 'second image'
        self.settings.windowsizes = (win1, win2)  # sizex//4, sizex//8 suggestion is these are power of 2 of each other
        self.settings.overlap = (win1//2, win2//2)  # should be 50%-75% of window size (Raffel)
        self.settings.num_iterations = len(self.settings.windowsizes)
        self.settings.subpixel_method = 'gaussian'  # subpixel interpolation: 'gaussian','centroid','parabolic'
        self.settings.interpolation_order = 3  # interpolation order for the window deformation (suggested: 3-5)
        self.settings.scaling_factor = self.pix_per_um  # scaling factor pixel/meter
        self.settings.dt = self.dtf  # time between to frames (in seconds)
        self.settings.ROI = ('full')

        # snr parameters
        self.mask_first_pass = True  # Mask first pass
        self.mask_multi_pass = True
        #self.settings.extract_sig2noise = True  # Compute SNR for last pass / if False: SNR set to NaN in output txt.
        self.settings.image_mask = True  # Do image masking
        self.settings.sig2noise_method = 'peak2peak'  # Method to compute SNR: 'peak2peak' or 'peak2mean'
        self.settings.sig2noise_mask = 3  # (2 - 5) exclusion distance between highest peak and second highest peak in correlation map
        # min/max velocity vectors for validation
        self.settings.MinMax_U_disp = (-self.char_u / 7.5, self.char_u / 7.5)  # filter u (units: pixels/frame)
        self.settings.MinMax_V_disp = (-self.char_u / 50, self.char_u / 50)  # filter v (units: pixels/frame)
        # vector validation
        self.settings.validation_first_pass = True  # Vector validation of first pass
        self.u_uncertainty = 0.35  # if std(u)*2 < uncertainty: don't apply global std threshold
        self.v_uncertainty = 0.35  # if std(v)*2 < uncertainty: don't apply global std threshold
        self.settings.std_threshold = 2.75  # global std validation threshold: global mean +/- stdev * std_threshold
        self.settings.median_threshold = 2.75  # median validation threshold
        self.settings.median_size = 2  # defines the size of the local median kernel
        self.settings.sig2noise_validate = True  # Enables validation by SNR ratio
        self.settings.sig2noise_threshold = 1.2  # [1.2-1.5] Sets snr threshold for removing vectors (R. D. Keane and R. J. Adrian, Measurement Science & Technology, 1990)
        # outlier replacement and smoothing
        self.settings.replace_vectors = True  # Outlier replacement for last pass
        self.replace_Nans_with_zeros = replace_Nans_with_zeros # Outlier replacement where all Nans = 0
        self.settings.smoothn = True  # Enables Garcia smoothing function of velocity fields
        self.settings.smoothn_p = [0.01]  # [0.5] Smoothing parameter or auto-calculated using generalized cross-validation (GCV) method
        self.settings.filter_method = 'distance'  # Replace outlier vector method: localmean [square] or disk (unweighted circle), distance (weighted circle)
        self.settings.max_filter_iteration = 3  # maximum iterations performed to replace the outliers (max 10)
        self.settings.filter_kernel_size = 2 # kernel size for replacing outlier vectors (default

        self.settings._freeze()


        # print PIV settings
        print('Min/Max U-displacement: ', self.settings.MinMax_U_disp, ' (pixels/frame)')
        print('Min/Max U-displacement: ', np.array([self.settings.MinMax_U_disp[0], self.settings.MinMax_U_disp[1]], dtype=int)*self.pixels_to_microns/self.dtf, ' (um/s)')

        print('Min/Max V-displacement: ', self.settings.MinMax_V_disp, ' (pixels/frame)')
        print('Min/Max V-displacement: ', np.array([self.settings.MinMax_V_disp[0],self.settings.MinMax_V_disp[1]], dtype=int)*self.pixels_to_microns/self.dtf, ' (um/s)')





