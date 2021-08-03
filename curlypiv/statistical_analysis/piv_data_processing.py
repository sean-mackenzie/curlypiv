"""
Notes about program

"""

# import modules
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# read PIV data into DataFrame
file = '/Users/mackenzie/Desktop/BPE-ICEO/06.11.21 - BPE-ICEO 500 um/chip2 - 500 nm pink/results/piv-data/PIV_data_E20.0Vmm_f500.0Hz_seq2.csv'
df = pd.read_csv(filepath_or_buffer=file, delimiter=',', index_col=False)

# average y-position velocities
dfx = df.groupby(['x']).mean()
print(dfx.head(10))

fig, ax = plt.subplots()
plt.scatter(dfx.index, dfx.u)
plt.xticks(ticks=dfx.index)
plt.show()

x = df.x.unique()
y = df.y.unique()