# CurlypivPlotting
"""
Notes about program
"""

# 1.0 import modules
# --- scientific ---
import numpy as np
from skimage import io
from skimage.transform import resize as skimage_resize
from skimage.exposure import rescale_intensity

# --- plotting ---
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.image as mgimg
from matplotlib import animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
matplotlib.rcParams['figure.figsize'] = (4, 8)
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=10, weight='bold')
font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']

def plot_u_mean_columns(test, plot_value='u', leftedge=None, rightedge=None, testname=None, num_analysis_frames=None, pivSetup=None):

    if plot_value == 'u':
        test_runs = []
        test_u_mean_columns = []
        test_u_mean_columns_std = []

        for run in test.runs.values():
            test_runs.append(run.name)
            test_u_mean_x = run.u_mean_x
            test_u_mean_columns.append(run.u_mean_columns)
            test_u_mean_columns_std.append(run.u_mean_columns_std)

        num_imgs = num_analysis_frames

        if test_u_mean_x is not None:
            x = test_u_mean_x
        else:
            x = np.arange(len(test_u_mean_columns[0]))
        test_runs = np.array(test_runs, dtype=float)
        test_u_mean_columns = np.array(test_u_mean_columns, dtype=float)
        test_u_mean_columns = np.round(np.mean(test_u_mean_columns, axis=0), 1)
        test_u_std_columns = np.round(np.mean(test_u_mean_columns_std, axis=0), 2)

        # inner BPE analysis
        u_inner_x = x[leftedge+2:rightedge-1]
        u_inner_slope = test_u_mean_columns[leftedge+2:rightedge-1]
        m_inner, b_inner = np.polyfit(u_inner_x, u_inner_slope,1)
        fit_inner_vals = m_inner*u_inner_x+b_inner

        # outer BPE analysis
        u_outer_x = x[leftedge:rightedge+1]
        u_outer_slope = test_u_mean_columns[leftedge:rightedge+1]
        m_outer, b_outer = np.polyfit(u_outer_x, u_outer_slope,1)
        fit_outer_vals = m_outer*u_outer_x+b_outer


        # plotting
        fig, ax = plt.subplots(figsize=(6,4))

        # plot data
        ax.plot(x, test_u_mean_columns, linewidth=2)
        ax.errorbar(x, test_u_mean_columns, yerr=test_u_std_columns*2, fmt='o', ms=5, color='darkblue',alpha=0.25, ecolor='darkgray', elinewidth=3, capsize=5, capthick=2)

        # plot fit lines
        ax.plot(u_outer_x, fit_outer_vals, label=r'$Slope_{BPE edges}$', color='darkred', alpha=1, linestyle='solid', linewidth=4)
        ax.plot(u_inner_x, fit_inner_vals, label=r'$Slope_{BPE center}$', color='indianred', alpha=1, linestyle='solid', linewidth=4)

        if testname is not None:
            if testname[1] > 1000:
                title = 'E{}Vmm f{}kHz'.format(int(np.round(testname[0], 0)), np.round(testname[1], 1))
            else:
                title = 'E{}Vmm f{}Hz'.format(int(np.round(testname[0], 0)), int(testname[1]))
        else:
            title = 'E{}Vmm f{}kHz'.format(int(np.round(test.testname[0], 0)), (test.testname[1]*1e-2))

        ax.set_title(((r'$U_{slope, inner}$=') + '{} '.format(np.round(m_inner,3)) + r'$U_{slip}/\Delta x$, ' + '$U_{slope, outer}$=' + '{} '.format(np.round(m_outer,3)) + r'$U_{slip}/\Delta x$, '), fontsize=12)
        ax.set_xlabel(r'$x$ ($\mu m$)')
        ax.set_ylabel(r'$u_{x,mean}$ ($\mu m/s$)')
        ax.set_ylim(bottom=-35, top=35)


        # plot zero line
        #ax.axhline(y=0, xmin=x[0], xmax=x[-1], linewidth=3, linestyle='dashed', color='gray', alpha=0.5)
        plt.axhline(y=0, linewidth=1, linestyle='dashed', color='black', alpha=0.5)

        plt.suptitle((r'$U_{slip, BPE-ICEO}$: ' + title + ' ({} imgs)'.format(num_imgs)), fontsize=16)
        plt.legend(prop=fontprops)
        plt.tight_layout()

        if pivSetup.save_u_mean_plot is True:
            pth = pivSetup.save_plot_path
            title = 'Uslip_mean_E{}Vmm_f{}Hz_{}images'.format(int(np.round(testname[0], 0)), int(testname[1]), num_imgs)
            savepath = pth + '/' + title + '.jpg'
            plt.savefig(fname=savepath)

        plt.show()




def plot_per_loc(loc, plot_value='Umag'):

    if plot_value == 'Umag':
        loc_tests = []
        loc_u_mag = []
        loc_u_std = []

        for test in loc.tests.values():
            loc_tests.append(test.name)
            loc_u_mag.append(test.u_mag_mean)
            loc_u_std.append(test.u_mag_std)

        loc_tests = np.array(loc_tests, dtype=float)
        loc_u_mag = np.array(loc_u_mag, dtype=float)
        loc_u_std = np.array(loc_u_std, dtype=float)

        m, b = np.polyfit(loc_tests, loc_u_mag,1)
        fit_vals = m*loc_tests+b
        mse = np.mean((loc_u_mag-fit_vals)**2)

        # plotting
        fig, ax = plt.subplots()

        ax.plot(loc_tests, fit_vals, label=r'Fit: $\upsilon$ = '+str(np.round(mse,1))+r' $\frac {um \cdot mm}{V \cdot s}$',
                color='skyblue', alpha=0.25, linestyle='solid', linewidth=3)

        ax.errorbar(loc_tests, loc_u_mag, yerr=loc_u_std*2, fmt='o', ms=5, color='darkred',alpha=0.5, ecolor='darkgray',
                    elinewidth=4, capsize=7, capthick=3)

        ax.set_xlabel(r'$|E|$ ($V/mm$)')
        ax.set_ylabel(r'$|u_{x,mean}|$ ($\mu m/s$)')

        plt.legend(prop=fontprops)

        plt.show()


def plot_quiver(x, y, u, v, img, pivSetup, img_piv_plot='filtered',
                u_mag_mean=None, u_mag_std=None,
                locname=None, testname=None, runname=None, seqname=None):

    # initialize plot
    fig, ax = plt.subplots()
    if np.mean(u) < 0:
        dir = "-"
    else:
        dir = "+"

    if u_mag_mean is not None and u_mag_std is not None:
        title = 'E{}Vmm Seq{} Frame{} - U={}{} +- {}'.format(testname, seqname, img.frame[0],
                                                           dir, u_mag_mean, 2*u_mag_std)
    else:
        title = 'E{}Vmm f{}Hz Seq{} Frame{}'.format(int(np.round(testname[0],0)), int(testname[1]), seqname, img.frame[0])

    ax.set_title(title, fontsize=10)

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,  # labels along the bottom edge are off
        reset=True)
    scalebar = AnchoredSizeBar(ax.transData,
                               pivSetup.scalebar_microns * pivSetup.pix_per_um,
                               str(pivSetup.scalebar_microns) + r'$\mu$m', 'lower right',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=5,
                               fontproperties=fontprops)

    ax.add_artist(scalebar)

    if pivSetup.vectors_on_image is True:

        if img_piv_plot == 'raw':
            img_plot = img.raw.copy()
            vmin, vmax = np.percentile(img_plot, (20, 99.5))
            img_plot = rescale_intensity(img_plot, in_range=(vmin, vmax), out_range='dtype')
        if img_piv_plot == 'filtered':
            img_plot = img.filtered.copy()
        if img_piv_plot == 'bg':
            img_plot = img.bg.copy()
        if img_piv_plot == 'bgs':
            img_plot = img.bgs.copy()
        if img_piv_plot == 'masked':
            img_plot = img.masked.copy()

        M = np.hypot(u, v)


        # resize with a bi-quartic interpolation
        img_plot = skimage_resize(img_plot,
                              (int(np.round(np.shape(img_plot)[0] / pivSetup.pix_per_um, 0)),
                               int(np.round(np.shape(img_plot)[1] / pivSetup.pix_per_um, 0))),
                              order=2, preserve_range=True)

        # plot the image
        ax.imshow(img_plot, cmap='gray')

        # plot the vector field
        quiver_scale = 5
        width = 2
        minlength = 0.1
        headwidth = width * 5
        headlength = headwidth * 3
        headaxislength = headlength / 2

        Q = ax.quiver(x, y, u, v, [M],
                      pivot='mid', angles='xy',
                      #scale_units='xy', scale=quiver_scale,
                      width=width * 1e-2,
                      # headwidth=headwidth, headlength=headlength, headaxislength=headaxislength,
                      # minlength=minlength,
                      cmap=pivSetup.colorMap, alpha=pivSetup.alpha, norm=pivSetup.colorNorm)
        cbar = fig.colorbar(Q, extend='max', fraction=0.1, shrink=0.5)
        cbar.set_label(label=r'$\frac{\mu m}{s}$', size=16)
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout()

        if pivSetup.save_plot is True:
            pth = pivSetup.save_plot_path
            savepath = pth + '/' + title + '.jpg'
            plt.savefig(fname=savepath)

        if pivSetup.show_plot is True:
            plt.show()
        else:
            plt.close('all')

def plot_quiver_and_u_mean(x, y, u, v, img, pivSetup, img_piv_plot='filtered',
                u_mean_columns=None,
                locname=None, testname=None, runname=None, seqname=None):

    # initialize plot
    fig, axes = plt.subplots(nrows=2, figsize=(5,8), gridspec_kw={'height_ratios': [1, 5]})
    ax = axes.ravel()

    if np.mean(u) < 0:
        dir = "-"
    else:
        dir = "+"

    if testname[1] > 1000:
        title = 'E{}Vmm f{}kHz Seq{} Frame{}'.format(int(np.round(testname[0], 0)), np.round(testname[1], 1), seqname,
                                                    img.frame[0])
    else:
        title = 'E{}Vmm f{}Hz Seq{} Frame{}'.format(int(np.round(testname[0],0)), int(testname[1]), seqname, img.frame[0])

    ax[0].set_title(title, fontsize=10)
    # plot u_mean_columns
    px = x[0, :]
    pux = u_mean_columns
    ax[0].plot(px, pux)

    ax[0].set_ylim(bottom=-35, top=35)
    ax[0].set_xlabel('x (window)')
    ax[0].set_ylabel(r'$u_{avg} (\mu m/s)$')

    ax[1].tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,  # labels along the bottom edge are off
        reset=True)
    scalebar = AnchoredSizeBar(ax[1].transData,
                               pivSetup.scalebar_microns * pivSetup.pix_per_um,
                               str(pivSetup.scalebar_microns) + r'$\mu$m', 'lower right',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=5,
                               fontproperties=fontprops)

    ax[1].add_artist(scalebar)

    if pivSetup.vectors_on_image is True:

        if img_piv_plot == 'raw':
            img_plot = img.raw.copy()
            vmin, vmax = np.percentile(img_plot, (0, 99.99))
            img_plot = rescale_intensity(img_plot, in_range=(vmin, vmax), out_range='dtype')
        if img_piv_plot == 'filtered':
            img_plot = img.filtered.copy()
        if img_piv_plot == 'bg':
            img_plot = img.bg.copy()
        if img_piv_plot == 'bgs':
            img_plot = img.bgs.copy()
        if img_piv_plot == 'masked':
            img_plot = img.masked.copy()

        M = np.hypot(u, v)


        # resize with a bi-quartic interpolation
        img_plot = skimage_resize(img_plot,
                              (int(np.round(np.shape(img_plot)[0] / pivSetup.pix_per_um, 0)),
                               int(np.round(np.shape(img_plot)[1] / pivSetup.pix_per_um, 0))),
                              order=2, preserve_range=True)

        # plot the image
        ax[1].imshow(img_plot, cmap='gray')

        # plot the vector field
        width=1.5
        Q = ax[1].quiver(x, y, u, v, [M],
                      units='xy', angles='xy', pivot='mid',
                         scale_units='xy', scale=0.25, width=3,
                      cmap=pivSetup.colorMap, alpha=pivSetup.alpha, norm=pivSetup.colorNorm)
        cbar = fig.colorbar(Q, extend='max', fraction=0.1, shrink=0.5)
        cbar.set_label(label=r'$\frac{\mu m}{s}$', size=16)
        cbar.ax.tick_params(labelsize=14)

        #plt.tight_layout()

        if pivSetup.save_plot is True:
            pth = pivSetup.save_plot_path
            savepath = pth + '/' + title + '.jpg'
            plt.savefig(fname=savepath)

        if pivSetup.show_plot is True:
            plt.show()
        else:
            plt.close('all')


def draw_particles(img, particles, color='blue',title='Laplace of Gaussians',figsize=(6,6)):
    # set up figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.imshow(img)

    # particles
    for p in particles:
        y, x, r = p
        c = plt.Circle((x,y), r, color=color, linewidth=0.25, fill=False)
        ax.add_patch(c)
    ax.set_axis_off()

    plt.show()

def plot_linear_cube(x, y, z, dx, dy, dz, ax, ylim, zlim, color='red', alpha=0.5, label='Interrogation Region'):

    if y < 0 or z < 0:
        raise ValueError("y and z must be greater than 0")

    xmax = x + dx
    ymax = y + dy
    zmax = z + dz

    if ymax > ylim:
        ymax = ylim
    if zmax > zlim:
        zmax = zlim

    xx = [x, x, xmax, xmax, x]
    yy = [y, ymax, ymax, y, y]

    kwargs = {'alpha': alpha, 'color': color}
    ax.plot3D(xx, yy, [z]*5, **kwargs)
    ax.plot3D(xx, yy, [zmax]*5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, zmax], **kwargs)
    ax.plot3D([x, x], [ymax, ymax], [z, zmax], **kwargs)
    ax.plot3D([xmax, xmax], [ymax, y+dy], [z, zmax], **kwargs)
    ax.plot3D([xmax, xmax], [y, y], [z, zmax], label=label, **kwargs)

def plot_square(x, y, dx, dy, ax, color='red', alpha=0.5):

    xmax = x + dx
    ymax = y + dy

    xx = [x, xmax, xmax, x, x]
    yy = [y, y, ymax, ymax, y]

    kwargs = {'alpha': alpha, 'color': color}
    ax.plot(xx, yy, **kwargs)



