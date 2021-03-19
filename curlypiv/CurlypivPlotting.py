# CurlypivPlotting
"""
Notes about program
"""

# 1.0 import modules
# --- scientific ---
import numpy as np
from skimage import io

# --- plotting ---
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.image as mgimg
from matplotlib import animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
matplotlib.rcParams['figure.figsize'] = (10, 8)
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=20, weight='bold')
font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']


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




