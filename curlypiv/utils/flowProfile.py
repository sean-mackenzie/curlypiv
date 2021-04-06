# flowProfile.py
"""
Notes
"""

# import modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from math import ceil

def generate_flowProfile(testSetup, flowType='pdf', z_resolution=10, y_mod=1.1, z_mod=5,
                         Umax_pdf=0, slip_near=0, slip_far=0, E=0, ep_mobility=0):
    """
    Notes:  This program calculates the flow profile by using array indices as the channel resolution (except for z)
    Params:
        z_resolution:   a resolution multiplier (e.g. z_resolution = 10 --> the number of z-coordinate points are 10X)
        y_mod:          decay function for y-domain boundaries
        z_mod:          decay function for z-domain boundaries
    """

    # check flowTypes
    valid_flowTypes = ['pdf', 'slip', 'ep']
    for f in flowType:
        if f not in valid_flowTypes:
            raise ValueError("{} not a valid flowType: {}".format(f, valid_flowTypes))

    # coordinate space bounds
    zmax = testSetup.chip.channel.height
    ymax = testSetup.chip.channel.width

    # coordinate space
    z = np.linspace(0, zmax, int(zmax * z_resolution * 1e6 + 1))
    y = np.linspace(0, ymax, int(ymax * 1e6 + 1))

    # decay function to help handle boundary conditions
    z_decay = decay_fucntion(z, decay_rate=z_mod, invert=False)
    y_decay = decay_fucntion(y, decay_rate=y_mod, invert=False)

    # initialize flow types
    ux_z_pdf = np.zeros_like(z)
    ux_z_slip = np.zeros_like(z)
    ux_y_pdf = np.zeros_like(y)
    ux_y_ep = np.zeros_like(y)
    ux_mag_pdf = 0
    ux_mag_pdf_dir = 1
    ux_mag_slip = 0
    ux_mag_slip_dir = 1
    ux_mag_ep = 0
    ux_mag_ep_dir = 1

    # type 1: pressure driven flow in rectangular channel
    if 'pdf' in flowType and Umax_pdf != 0:

        ux_z_pdf = 4 * Umax_pdf * (z / zmax) * (1 - z / zmax)           # Ux ( z )
        ux_y_pdf = 4 * Umax_pdf * (y / ymax) * (1 - y / ymax) / y_mod   # Ux ( z )
        ux_mag_pdf = np.max(np.abs(ux_z_pdf))
        ux_mag_pdf_dir = np.mean(ux_z_pdf)/np.abs(np.mean(ux_z_pdf))

    if 'slip' in flowType and (slip_near != 0 or slip_far != 0):

        ux_z_slip = (slip_far - slip_near) * (z / zmax) + slip_near      # Ux ( z )
        ux_mag_slip = np.max(np.abs(ux_z_slip))
        ux_mag_slip_dir = np.mean(ux_z_slip) / np.abs(np.mean(ux_z_slip))

    if 'ep' in flowType and E != 0 and ep_mobility != 0:

        ux_y_ep = ep_mobility * E * np.ones_like(y)
        ux_mag_ep = np.max(np.abs(ux_y_ep))
        ux_mag_ep_dir = np.mean(ux_y_pdf) / np.abs(np.mean(ux_y_ep))

    # add z and y arrays
    ux_z = (ux_z_pdf + ux_z_slip)*z_decay
    ux_y = (ux_y_pdf + ux_y_ep)*y_decay

    # stack the arrays
    uxz = []
    for i in range(int(ymax*1e6)+1):
        uxz.append(ux_z)
    uxz = np.array(uxz)

    # stack uxy
    uxy = []
    for i in range(int(zmax * z_resolution * 1e6 + 1)):
        uxy.append(ux_y)
    uxy = np.array(uxy)

    # 3d flow profile
    uxx = uxy + np.transpose(uxz)
    mag_scaling = ux_mag_pdf*ux_mag_pdf_dir + ux_mag_slip*ux_mag_slip_dir + ux_mag_ep*ux_mag_ep_dir
    if mag_scaling == 0:
        mag_scaling = 0.001
    u_xyz = np.abs(uxx)/np.max(np.abs(uxx)) * mag_scaling

    return u_xyz

def decay_fucntion(example_array, decay_rate, invert=False):

    x = np.linspace(0, len(example_array)/2, num=len(example_array)//2)
    if invert:
        half_decay = np.exp(-x*decay_rate)
    else:
        half_decay = 1 - np.exp(-x*decay_rate)

    decay = np.concatenate((half_decay, half_decay[::-1]), axis=0)

    if len(decay) != len(example_array):
        decay = np.insert(decay, len(decay)//2, 1)

    return decay

def plot_flowProfile_2D(u_xyz, show_plot=True,
                        save_plot=False, savePath=None, saveName=None, saveType='.jpg',
                        cmap='viridis', interp='bilinear'):

    x = np.arange(0, np.shape(u_xyz)[1], 1)
    y = np.arange(0, np.shape(u_xyz)[0], 1)

    fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)

    im = NonUniformImage(ax, interpolation=interp, extent=(x[0], x[-1], y[0], y[-1]), cmap=cmap)
    im.set_data(x, y, u_xyz)
    cs = ax.add_image(im)

    cbar = fig.colorbar(cs)
    cbar.set_label(r'$\frac {U_{max, input}}{U_{max}}$')

    plt.title('Normalized microchannel flow profile')
    plt.xlabel('x (um)')
    plt.ylabel('z (um)')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])

    if save_plot:
        plt.savefigure(fname=savePath+'/'+saveName+saveType)

    if show_plot:
        plt.show()

def plot_flowProfile_3D(u_xyz, plotType='surface', show_plot=True,
                        save_plot=False, savePath=None, saveName=None, saveType='.jpg',
                        cmap='viridis'):

    x = np.arange(0, np.shape(u_xyz)[1], 1)
    y = np.arange(0, np.shape(u_xyz)[0], 1)

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 5), tight_layout=True)

    if plotType == 'surface':
        cs = ax.plot_surface(X, Y, u_xyz, rstride=1, cstride=1, cmap=cmap, edgecolor=None)
    if plotType == 'wireframe':
        cs = ax.plot_wireframe(X, Y, u_xyz, cmap=cmap)

    cbar = fig.colorbar(cs, shrink=0.5, orientation='vertical')
    cbar.set_label(r'$\frac {U_{max, input}}{U_{max}}$', fontsize=14)
    cbar.set_ticks([-np.min(u_xyz), np.max(u_xyz)])

    ax.set_xlabel("x (um)", fontsize=12)
    ax.set_ylabel("z (um)", fontsize=12)
    ax.set_title('Normalized microchannel flow profile', fontsize=14)
    ax.view_init(elev=40, azim=300)

    plt.tight_layout()

    if save_plot:
        plt.savefigure(fname=savePath+'/'+saveName+saveType)

    if show_plot:
        plt.show()


