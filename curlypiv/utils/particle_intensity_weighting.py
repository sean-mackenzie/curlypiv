
# linear algebra
import numpy as np

# scientific
from scipy import signal, misc
from skimage.exposure import rescale_intensity

# plotting
import matplotlib.pyplot as plt



def illuminator(x, y, z=None,
                illum_xy=None, fullx=None, fully=None, framex=None, framey=None, startx=None, starty=None,
                scale_z=False, illum_z=None, fullz=None, focal_z=None, input_type='value', df=None):

    if input_type in ['val', 'value']:

        # --- XY plane illumination ---

        # step 1: ensure there is a mask
        if illum_xy is not None:
            mask = illum_xy
        else:
            # if no mask but mask is inside illumination frame, we must create the mask distribution

            # convolve the frame area with an equal size gaussian
            if framex <= framey:
                r = framex
            else:
                r = framey

            # initialize
            sigma =2
            p_array = np.ones((framey, framex))
            kernel = np.outer(signal.windows.gaussian(M=r, std=sigma),
                              signal.windows.gaussian(M=r, std=sigma ,))

            # convolve ones array with square gaussian of smallest side length
            mask = signal.fftconvolve(p_array, kernel, mode='valid')

            # rescale to max of 1
            g_max = np.max(mask)
            mask = rescale_intensity(in_range=(0 ,g_max), out_range=(0 ,1))

        # step 2: compare the particle location to mask location
        if x > startx and x < startx + framex and y > starty and y < starty + framey:
            xstep = x - startx
            ystep = y - starty
            c_xy_int = mask[int(ystep), int(xstep)]   # assign c_int
        else:
            c_xy_int = 0

        # --- Z plane illumination ---
        if scale_z is True:

            # step 0: adjust for units
            if np.mean(z) > 0:
                z = z * 1e-6
            if np.mean(fullz) > 0:
                fullz = fullz * 1e-6
            if np.mean(focal_z) > 0:
                focal_z = focal_z * 1e-6

            # step 1: calculate the depth of correlation for the optical setup
            eps = 0.01      # scaling factor
            n0 = 1          # refractive index of immersion medium
            dp = 500e-9     # particle diameter
            NA = 0.4        # numerical aperture of objective
            M = 20          # magnification
            lmbda = 590e-9  # wavelength of light (Pink SpheroTech peak)

            z_corr = 2 * ((1 - np.sqrt(eps)) / np.sqrt(eps) * ((n0 ** 2 * dp ** 2) / (4 * NA ** 2) +
                                                               (5.95 * (M + 1) ** 2 * lmbda ** 2 * n0 ** 4) / (
                                                                           16 * M ** 2 * NA ** 4))) ** 0.5

            # step 2: calculate the weighting function
            z_observe = np.linspace(0, fullz, num=250)
            z_field = np.linspace(-z_corr, z_corr, num=250) + focal_z
            z_weight = 1 / (1 + (3 * z_field / z_corr) ** 2) ** 2

            # step 3: plot the weighting function
            plot_weight = False
            """
            if plot_weight is True:
                fig, ax = plt.subplots()

                # plot weighting function
                ax.scatter(z_field, z_weight, s=5)
                ax.plot(z_field, z_weight, alpha=0.25, linewidth=2)

                # plot focal plane
                plt.vlines(x=focal_z, ymin=0, ymax=1, colors='r', linestyles='dashed', alpha=0.25, label='focal plane')

                # plot channel walls
                plt.vlines(x=-focal_z * 1e6, ymin=0, ymax=1, colors='gray', linestyles='solid', alpha=0.25,
                           label='channel walls')
                plt.vlines(x=(fullz - focal_z) * 1e6, ymin=0, ymax=1, colors='gray', linewidth=2, linestyles='solid',
                           alpha=0.5)

                ax.set_xlabel('z-position (um)')
                ax.set_ylabel('weighted-contribution to PIV')
                plt.title("Real Depth-Correlated Weight")
                plt.legend()

                plt.show()
            """

            # step 3: compare particle z-height and weighting function distribution
            if z < np.max(z_field) and z > np.min(z_field):
                c_z_int = 1 / (1 + (3 * np.abs(focal_z - z) / z_corr) ** 2) ** 2
                jjjj = z * 1e6
                jjj = focal_z * 1e6
                jj = (z - focal_z) * 1e6
                j = c_z_int
            else:
                jj = z
                c_z_int = 0

        # step 5: combined particle intensity
        c_int = c_xy_int * c_z_int


    return c_int, z_corr
