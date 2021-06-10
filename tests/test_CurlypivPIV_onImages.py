# test CurlypivImage
"""
Notes about program
"""

# 1.0 import modules
# data I/O
import sys
import os

# scientific
import numpy as np

# Image processing
import cv2 as cv
# skimage
from skimage import io
from skimage.morphology import disk, white_tophat
from skimage.filters import median, gaussian
from skimage.restoration import denoise_wavelet
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, pyramid_expand
import skimage.transform as skt
from skimage.feature import blob_log

# plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import matplotlib.image as mgimg
from matplotlib import animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
matplotlib.rcParams['figure.figsize'] = (10, 9)
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=16, weight='bold')
font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']

# OpenPIV
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv"))
sys.path.append(os.path.abspath("/Users/mackenzie/PythonProjects/openpiv/openpiv"))
from openpiv import *
from windef import Settings
import openpiv.piv
from openpiv import windef
from openpiv.windef import Settings
from openpiv import tools, scaling, validation, filters, preprocess
from openpiv.pyprocess import extended_search_area_piv, get_field_shape, get_coordinates
from openpiv import smoothn
from openpiv.preprocess import mask_coordinates
testset = Settings()

# Curlypiv
from curlypiv.CurlypivTestCollection import CurlypivTestCollection
from curlypiv.CurlypivTestSetup import CurlypivTestSetup
from curlypiv.CurlypivPIV import CurlypivPIV
from curlypiv.CurlypivPIVSetup import CurlypivPIVSetup
from curlypiv.CurlypivFile import CurlypivFile
from curlypiv.CurlypivImageProcessing import img_resize, img_subtract_background, img_filter


# ------------------------- test CurlypivPIV below ------------------------------------

# inputs
nameTestCol = 'testCol'
nameTestSetup = 'testSet'
namePIVSetup = 'testPIV'
base_path = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test'
test_dir = 'tests'
test_level = 'seq' # ['all','loc','test','run','seq','file']
img_type = '.tif'
loc = 1
test = 2.5
testid = ('E','Vmm')
run = 3
runid = ('run', 'num')
seq = 1
seqid = ('test_', '_X')
frameid = '_X'

# processing inputs
scale = 2
bg_method = 'KNN'

if bg_method == 'KNN':
    backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)

testCol = CurlypivTestCollection(nameTestCol, base_path, file_type=img_type, dir_tests=test_dir,
                                                              testid=testid, runid=runid, seqid=seqid, frameid=frameid)
testSet = CurlypivTestSetup(name=nameTestSetup)

pivSet = CurlypivPIVSetup(name=namePIVSetup, save_text=False, save_plot=True, vectors_on_image=True,
                        testCollection=testCol,testSetup=testSet
                        )

# instantiate PIV class object.
piv = CurlypivPIV(testCollection=testCol, testSetup=testSet, pivSetup=pivSet)

# get appropriate metrics level
imgs = piv.get_analysis_level(level=test_level, loc=loc, test=test, run=run, seq=seq)


# ----- TEST IMAGE PROCESSING METHODS ON SEVERAL CURLYPIV.IMAGES IN A SEQUENCE -----

img_baseline = imgs.get_sublevel(key='test_1_X1.tif')

img1 = imgs.get_sublevel_all()

cropping = {
    'xmin': 100,  # x = 0 is the left of the image
    'xmax': 356,
    'ymin': 300,
    'ymax': 428  # y = 0 is the bottom of the image
}

processing = {
    # 'none': {'none'},
    # 'median': {'args': [disk(5)]},
    # 'gaussian': {'args': [3]},
    # 'white_tophat': {'args': [disk(5)]}, # returns bright spots smaller than the structuring element.
    'denoise_wavelet': {'args': [], 'kwargs': dict(method='BayesShrink', mode='soft', rescale_sigma=True)},
    'rescale_intensity': {'args': [(4, 99.995), ('dtype')]}
}

y = range(len(img1))
pth = '/Users/mackenzie/Desktop/03.18.21-ZuPIV_test/tests/loc1/E2.5Vmm/run3num/piv'

for i in range(len(img1)-1):


    print(i)

    if i == 0:

        # crop
        #img1[i].image_crop(cropspecs=cropping)

        # resize
        #img1[i].image_resize(method='pyramid_expand', scale=scale)

        # filter
        img1[i].image_filter(filterspecs=processing, image_input='raw', image_output='filtered', force_rawdtype=True)

        # subtract background
        img1[i].image_subtract_background(image_input='filtered', backgroundSubtractor=backSub)

    else:
        i = i - 1

        # crop
        #img1[i+1].image_crop(cropspecs=cropping)

        # resize
        #img1[i+1].image_resize(method='pyramid_expand', scale=scale)

        # filter
        img1[i+1].image_filter(filterspecs=processing, image_input='raw', image_output='filtered', force_rawdtype=True)

        # subtract background
        img1[i+1].image_subtract_background(image_input='filtered', backgroundSubtractor=backSub)

        if i > 7:

            piv_pass = 0

            # 3.1.4 - Start First Pass PIV
            x, y, u, v, s2n = windef.first_pass(
                img1[i].masked,
                img1[i + 1].masked,
                pivSet.settings
            )

            if np.isnan(u[0][0]) is True:
                print("PIV First-Pass gives no results: (u,v) = Nan")
                raise KeyboardInterrupt

            mask_coords = []
            u = np.ma.masked_array(u, mask=np.ma.nomask)
            v = np.ma.masked_array(v, mask=np.ma.nomask)

                # 3.2.0 - Start Multi Pass PIV
            piv_pass += 1

            # 3.2.0 - Run multi pass windows deformation loop
            for current_iteration in range(0, pivSet.settings.num_iterations):
                x, y, u, v, s2n, mask = windef.multipass_img_deform(
                    img1[i].masked,
                    img1[i + 1].masked,
                    current_iteration,
                    x,
                    y,
                    u,
                    v,
                    pivSet.settings,
                    mask_coords=mask_coords
                )

            # If the smoothing is active, we do it at each pass
            # but not the last one
            if pivSet.settings.smoothn is True and current_iteration < pivSet.settings.num_iterations - 1:
                u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(
                    u, s=pivSet.settings.smoothn_p
                )
                v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(
                    v, s=pivSet.settings.smoothn_p
                )

            # 3.2.2 - Adjust scaling
            u = u / pivSet.settings.dt
            v = v / pivSet.settings.dt
            x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=pivSet.settings.scaling_factor)

            if np.isnan(u[0][0]) == True:
                print("PIV Multi-Pass gives no results: (u,v) = Nan")
                raise KeyboardInterrupt

            # Save vector field
            if pivSet.settings.save_plot is True or pivSet.settings.show_plot is True:

                # ------------ INITIALIZE PLOTTING DETAILS ------------

                # initialize plot
                fig, ax = plt.subplots(figsize=(10, 9))
                ax.set_title("Frame: " + str(img1[i].name))

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
                                           pivSet.scalebar_microns * pivSet.pix_per_um,
                                           str(pivSet.scalebar_microns) + r'$\mu$m', 'lower right',
                                           pad=0.1,
                                           color='white',
                                           frameon=False,
                                           size_vertical=5,
                                           fontproperties=fontprops)

                ax.add_artist(scalebar)

                if pivSet.vectors_on_image is True:

                    img_plot = img1[i].filtered.copy()

                    M = np.hypot(u, v)

                    # ------------ ENHANCE IMAGE CONTRAST FOR VISUALIZATION ------------
                    if pivSet.settings.windowsizes[-1] < 30:
                        # 4 passes with a grid of 8
                        quiver_scale = 5 # bigger number = smaller arrows
                    else:
                        # 3 passes with a grid of 16
                        quiver_scale = 5

                    # print(quiver_scale)

                    """
                    # brighten brigth areas and darken dark areas
                    #img_plot = np.where(img_plot>np.mean(img_plot)*np.std(img_plot)*0.1,img_plot*5,img_plot*0.8)

                    # set max and min intensities
                    img_plot = np.where(img_plot<vmax,img_plot,vmax)          # clip upper percentile
                    img_plot = np.where(img_plot>vmin,img_plot,vmin)          # clip lower percentile

                    # gaussian blur
                    img_plot = gaussian(img_plot, sigma=0.5)
                    """

                    # match the histogram to the reference image (first image in stack)
                    #img_plot = match_histograms(img_plot, img_reference)

                    # resize with a bi-quartic interpolation
                    img_plot = skt.resize(img_plot,
                                      (int(np.round(np.shape(img_plot)[0] / pivSet.pix_per_um, 0)),
                                       int(np.round(np.shape(img_plot)[1] / pivSet.pix_per_um, 0))),
                                      order=2, preserve_range=True)

                    # recast as uint16
                    img_plot = np.rint(img_plot)
                    img_plot = img_plot.astype(np.uint16)

                    # check max/min pixels values to make sure nothing is getting funky
                    # print("Max pixel value: " + str(np.max(img_plot)))
                    # print("Min pixel value: " + str(np.min(img_plot)))

                    # plot the image
                    ax.imshow(img_plot, cmap='gray')
                    # ax.imshow(img_mask_a)

                    # plot the vector field
                    width = 1
                    minlength = 0.1
                    headwidth = width * 5
                    headlength = headwidth * 3
                    headaxislength = headlength / 2

                    Q = ax.quiver(x, y, u, v, [M],
                                  pivot='mid', angles='xy', scale_units='xy', scale=quiver_scale,
                                  #width=width * 1e-2,
                                  #headwidth=headwidth, headlength=headlength, headaxislength=headaxislength,
                                  #minlength=minlength,
                                  cmap=pivSet.colorMap, alpha=pivSet.alpha, norm=pivSet.colorNorm)
                    cbar = fig.colorbar(Q, extend='max', fraction=0.1, shrink=0.5)
                    cbar.set_label(label=r'$\frac{\mu m}{s}$', size=16)
                    cbar.ax.tick_params(labelsize=14)

                    #plt.show()

                    savepath = pth + '/' + str(i) + '.png'
                    plt.savefig(fname=savepath,)