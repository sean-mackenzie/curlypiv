import numpy as np
from os.path import isdir, join, dirname
from os import mkdir
from pathlib import Path

DEFAULTS = dict(
    magnification = 50,
    numerical_aperture = 0.7,
    focal_length = 3,
    ri_medium = 1,
    ri_lens = 1.5,
    pixel_size = 16,
    pixel_dim_x = 512,
    pixel_dim_y = 512,
    background_mean = 300,
    background_noise = 10,
    points_per_pixel = 15,
    n_rays = 500,
    gain = 3,
    cyl_focal_length = 0
)

def generate_sig_settings(settings, folder=None):
    assert isinstance(settings, dict)
    assert folder is not None

    settings_dict = {}
    settings_dict.update(DEFAULTS)
    settings_dict.update(settings)

    if not isdir(folder):
        mkdir(folder)

    # Generate settings.txt
    settings = ''
    for key, val in settings_dict.items():
        settings += '{} = {}\n'.format(key, val)

    with open(join(folder, 'settings.txt'), 'w') as file:
        file.write(settings)

    return settings_dict, join(folder, 'settings.txt')

def generate_random_coordinates(density, imshape, particle_d, dpm,
                                pixel_step_per_frame=5,
                                 z=None, numimages=20, flowprofile='pdf'):

    # image size depends on flow velocity and numimages

    # particle density
    m_per_px = 1 / dpm
    a_total = imshape[0] * imshape[1] * m_per_px** 2    # total area
    a_particle = (particle_d / 2) ** 2 * np.pi          # particle area

    n_particles = int(a_total * density / a_particle)
    xy_coords = np.hstack([np.random.randint(0, imshape[0], size=(n_particles, 1)),
                           np.random.randint(0, imshape[1], size=(n_particles, 1))])

    if isinstance(z, int) or isinstance(z, float):
        z_coords = z * np.ones((n_particles, 1))
    else:
        z_coords = np.random.uniform(z[0], z[1], size=(n_particles, 1))

    coords = np.hstack([xy_coords, z_coords])

    return coords

