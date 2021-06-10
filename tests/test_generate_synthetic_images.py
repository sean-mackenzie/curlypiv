# numerics
import numpy as np
from numpy.random import default_rng
import pandas as pd
# scientific


# matplotlib
import matplotlib.pyplot as plt

# curlypiv
from curlypiv.synthetics.generate_synthetic_imageset import generate_sig_settings
from curlypiv.utils.particle_intensity_weighting import illuminator


# synthetic image generator details
setup_params = dict(
    magnification = 50,
    numerical_aperture = 0.7,
    focal_length = 3,
    ri_medium = 1,
    ri_lens = 1.5,
    pixel_size = 16,
    pixel_dim_x = 512,
    pixel_dim_y = 512,
    pixel_dim_z = 20,
    background_mean = 300,
    background_noise = 10,
    points_per_pixel = 15,
    n_rays = 500,
    gain = 3,
    cyl_focal_length = 0
)

n_images = 50           # # of images
buffer_images = 4
range_z = (0, 20)        # 0 - 20 microns
zstep = 21
z_focal_plane = 5       # height of focal plane
density = 5e-5          # 1:100 density
particle_diameter = 0.5   # particle size

# flow dynamics
z = np.linspace(range_z[0], range_z[1], zstep)
y = np.linspace(0, setup_params['pixel_dim_y'], setup_params['pixel_dim_y']+1)
zmax = z[-1] - z[0]
pixel_step_per_frame = 5

# Ux ( z )
ux_z_pdf = z*(zmax-z)
ux_z_pdf = ux_z_pdf/np.max(ux_z_pdf)*pixel_step_per_frame

# Ux ( y )
ux_y_pdf = y*(np.max(y)-y)
ux_y_pdf = ux_y_pdf/np.max(ux_y_pdf)*pixel_step_per_frame

# stack the arrays
uxz = []
for i in range(len(ux_y_pdf)):
    uxz.append(ux_z_pdf)
uxz = np.array(uxz)

# stack uxy
uxy = []
for i in range(len(ux_z_pdf)):
    uxy.append(ux_y_pdf)
uxy = np.array(uxy)

# 3d flow profile
uxx = uxy + np.transpose(uxz)
uxx = uxx/np.max(uxx)

# generate the initial array
x, y, z = (int(setup_params['pixel_dim_y']+0.5*pixel_step_per_frame*(n_images+buffer_images)), int(setup_params['pixel_dim_y']), int(zmax))
rng = default_rng()
rints = rng.integers(low=0, high=1//density+1, size=(x, y, z))
rints_mask = rints < 2
rints[~rints_mask] = 0
fx = np.shape(rints)[0]
fy = np.shape(rints)[1]

# find coordinates
coords = np.argwhere(rints==1)
rints_val = rints.ravel()
rints_x = np.indices((x, y)).transpose((1,2,0))
rints_x = np.array(rints_x[:,:,0].flatten())
rints_y = np.indices((y, x)).transpose((1,2,0))
rints_y = np.array(rints_y[:,:,1].flatten())

rints_xx = []
rints_yy = []
for i in range(int(zmax)):
    rints_xx = np.append(rints_xx, rints_x)
    rints_yy = np.append(rints_yy, rints_y)

rints_z = np.indices((int(zmax), int(x*y))).transpose((1,2,0))
rints_z = np.array(rints_z[:,:,0].flatten())


rints_full = np.stack((rints_xx, rints_yy, rints_z, rints_val), axis=1)
dfz = pd.DataFrame(data=rints_full, index=None, columns=['x','y','z','val'])

# import illumination data
illumPath = '/Users/mackenzie/PythonProjects/microsig/examples/example_sean/img_uneven_illumination.txt'
illum = np.loadtxt(illumPath)
ily, ilx = np.shape(illum)

# determine interrogation window
ix = int(x-(buffer_images)*pixel_step_per_frame//2-ilx)
iy = int(y-ily)

# create a zeros illumination array
illum_zeros = np.zeros((y, x), dtype=float)

# replace with measured illumination array
illum_zeros[0:ily, ix:ix+ilx] = illum

# find coords
illum_coords = np.argwhere(illum_zeros >= 0)
illum_y, illum_x = zip(*illum_coords)
illum_x = np.array(illum_x)
illum_y = np.array(illum_y)
illum_c = illum_zeros.flatten()
illum_full = np.stack((illum_x, illum_y, illum_c), axis=1)

# import into pandas
dfi = pd.DataFrame(data=illum_full, index=None, columns=['x','y','c_int'])

# import into pandas
df_particles = pd.DataFrame(data=coords, index=None, columns=['x','y','z'], dtype=None)

# join
df = pd.merge(dfz, dfi, how='inner', left_on=['x','y'], right_on=['x','y'])

# -----  -----  -----  -----  -----  -----  -----  -----  -----

# ----- reduce master array to only non-zero values -----

# filter full array for only true values
dfp = df[df['val']>0.0]
dfp = dfp.astype({'x':np.uint16, 'y':np.uint16, 'z':np.uint16})

# add column: particle diameter
p_dia = np.ones(len(dfp))*particle_diameter
dfp.insert(loc=4, column="dp", value=p_dia, allow_duplicates=False)


"""
This plots a 3D scatter where the color corresponds to the illumination factor

# setup the figure and colormap
fig = plt.figure()
ax = plt.axes(projection='3d')
cmap = plt.get_cmap('cool')

# plot
sctt = ax.scatter(dfp.x, dfp.y, dfp.z, c=dfp.c_int, cmap=cmap)

# plot formatting
plt.title('Initial particle distribution')
ax.set_xlabel('x')
ax.set_ylabel('y')

# colorbar
cbar = fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
cbar.ax.set_title('Illumination Scaling')

plt.show()
"""

"""
This plot demonstrates the masked area

#fig, ax = plt.subplots()
#ax0 = fig.add_subplot(1,2,1)
#ax0.imshow(illum_zeros, cmap='cool')
#ax0.set_title('Illumination Mask')
#ax1 = fig.add_subplot(1,2,2, projection='3d')
#ax1.scatter(illum_x, illum_y, illum_c)
#plt.show()
"""


"""
Plots all particles as a scatter plot time-series: points are streaking across the figure. 

# cycle through and write text output file
fig = plt.figure()
ax = fig.add_subplot(3,1,(1,2), projection='3d')
axz = fig.add_subplot(3,2,5)
axx = fig.add_subplot(3,2,6)

# dataframe {x, y, z} columns need to be integers to perform operations
dfp = dfp.astype({'x':np.uint8, 'y':np.uint8, 'z':np.uint8})

n = 0
for n in range(n_images):

    ax.scatter(dfp.x, dfp.y, dfp.z)
    ax.set_title('3D projection')

    axz.scatter(dfp.x, dfp.z)
    axz.set_title('X-Z projection')

    axx.scatter(dfp.x, dfp.y)
    axx.set_title('X-Y projection')

    # particle stepper that moves particles according to uxx flow profile.
    dfp.x = dfp.x + slip_near*uxx[dfp.z, dfp.y]

plt.show()
"""

"""
Plots scatter streaks with initial illumination dependence but does not update with time stepping

# cycle through and write text output file
cmap = plt.get_cmap('cool')
fig = plt.figure()
ax = fig.add_subplot(3,1,(1,2), projection='3d')
axz = fig.add_subplot(3,2,5)
axx = fig.add_subplot(3,2,6)

saveloc = '/Users/mackenzie/PythonProjects/microsig/examples/example_sean/images/'
saveimg = 'img'

n = 0

print(dfp)
dfp = dfp.astype({'x':np.uint8, 'y':np.uint8, 'z':np.uint8})

for n in range(n_images):

    # 3D project
    sctt = ax.scatter(dfp.x, dfp.y, dfp.z, c=dfp.c_int, cmap=cmap)

    # 2D projections
    sctt1 = axz.scatter(dfp.x, dfp.z, c=dfp.c_int, cmap=cmap)
    sctt2 = axx.scatter(dfp.x, dfp.y, c=dfp.c_int, cmap=cmap)

    # particle motion stepper
    dfp.x = dfp.x + slip_near*uxx[dfp.z, dfp.y]

    #np.savetxt(saveloc+saveimg+str(n)+'.txt', df.values, fmt='%d')

# figure setup
ax.set_title('3D projection')
axz.set_title('X-Z projection')
axx.set_title('X-Y projection')


# colorbar
cbar = fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
cbar.ax.set_title('Illumination Scaling')

plt.show()
"""

"""
Plot all initial points (blue) and illumination zone (red) as a scatter plot

dfp = dfz[dfz['val']>0.0]
cmap='cool'
fig = plt.figure()
ax = fig.add_subplot(3,1,(1,2), projection='3d')
axz = fig.add_subplot(3,2,5)
axx = fig.add_subplot(3,2,6)
sctt = ax.scatter(dfp.x, dfp.y, dfp.z, cmap=cmap, s=3, alpha=0.15)
sctt1 = axz.scatter(dfp.x, dfp.z, cmap=cmap)
sctt2 = axx.scatter(dfp.x, dfp.y, cmap=cmap)

# draw the illumination area
xx, yy, zz = np.indices((ilx, ily, 10))
xx = xx + ix
yy = yy + iy

cube = (xx <= xx + ix) & (yy <= yy + iy) & (zz <= 10)
voxel = cube
#color = np.empty_like(voxel.shape, dtype=object)
color = 'red'

#sctt_illum = ax.voxels(voxel, facecolor=color, alpha=0.25, edgecolor='k')
#sctt1_illum = axz.voxels(voxel, facecolor=color, alpha=0.25, edgecolor='k')
#sctt1_illum = axx.voxels(voxel, facecolor=color, alpha=0.25, edgecolor='k')

sctt_illum = ax.scatter(xx, yy, zz, alpha=0.05, color='red')
sctt1_illum = axz.scatter(xx, zz, alpha=0.05, color='red')
sctt2_illum = axx.scatter(xx, yy, alpha=0.05, color='red')

# figure setup
ax.set_title('3D projection')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
axz.set_title('X-Z projection')
ax.set_xlabel('x')
ax.set_ylabel('z')
axx.set_title('X-Y projection')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()
"""

# Plots scatter streaks with initial illumination dependence but does not update with time stepping

# cycle through and write text output file
cmap = plt.get_cmap('cool')

fig = plt.figure()
ax = fig.add_subplot(3,1,(1,2), projection='3d')
axz = fig.add_subplot(3,2,5)
axx = fig.add_subplot(3,2,6)

# add column: c_int
particle_intensity = np.zeros(len(dfp))
dfp.insert(loc=5, column="particle_intensity", value=particle_intensity, allow_duplicates=False)
n = 0

for n in range(n_images):

    dfp.drop(columns='c_int')
    n_c_int = []

    for index, particle in dfp.iterrows():
        # particle intensity checker
        p_c_int, depth_of_corr = illuminator(particle.x, particle.y, particle.z,
                                illum_xy=illum, fullx=x, fully=y, fullz=z,
                                framex=ilx, framey=ily, startx=ix, starty=iy,
                                scale_z=True, focal_z=z_focal_plane)
        n_c_int.append(p_c_int)

    dfp['c_int'] = n_c_int

    # scatter plot
    sctt = ax.scatter(dfp.x, dfp.y, dfp.z, c=dfp.c_int, cmap=cmap, s=3,alpha=0.15)
    sctt1 = axz.scatter(dfp.x, dfp.z, c=dfp.c_int, cmap=cmap)
    sctt2 = axx.scatter(dfp.x, dfp.y, c=dfp.c_int, cmap=cmap)

    # particle motion stepper
    dfp.x = dfp.x + pixel_step_per_frame*uxx[dfp.z, dfp.y]

"""
# draw the illumination area
xx, yy, zz = np.indices((ilx, ily, int(depth_of_corr*1e6)))
xx = xx + ix
yy = yy + iy
zz = zz + z_focal_plane


color = 'red'
sctt_illum = ax.scatter(xx, yy, zz, alpha=0.05, color='red')
sctt1_illum = axz.scatter(xx, zz, alpha=0.05, color='red')
sctt2_illum = axx.scatter(xx, yy, alpha=0.05, color='red')
"""


# figure setup
ax.set_title('Shift -z down')
axz.set_title('X-Z projection')
axx.set_title('X-Y projection')


# colorbar
cbar = fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
cbar.ax.set_title('Illumination Scaling')

plt.show()


# nothing below here in progress
folder = '/Users/mackenzie/PythonProjects/microsig/examples/example_sean/images/'
# step 1 - generate the settings file
settings_dict, settings_path = generate_sig_settings(setup_params, folder=folder)
# step 2 - generate the test images
#x, y, z = (setup_params['pixel_dim_x'], setup_params['pixel_dim_y'], setup_params['pixel_dim_z'])