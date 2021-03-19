"""
Notes about the program
"""

# import modules
from curlypiv import CurlypivImageCollection

# test zeta potential measurement via micro particle image velocimetry analysis

# step 1 - load data files, etc
# --- notes ---

# load image
base_path = '/Users/mackenzie/Desktop/03.04.21-ZuPIVelastosil'
dir_read = '/Users/mackenzie/Desktop/03.04.21-ZuPIVelastosil/tests/tests_loc2'
img_type = '.tif'

# folder structure (change if different than suggested)
test_folder = 'tests/tests_loc1'
background_folder = 'background'
results_folder = 'results'


# step 2 - perform background subtraction
# --- notes ---

# step 2.1 -- initialize the image collection
collection = CurlypivImageCollection(dirpath=base_path, file_type=img_type, backgroundsub_specs=None,
                                     dir_tests=test_folder)


