# Parse XML file

# import modules
import numpy as np
import csv
import xml.etree.ElementTree as ET
from os.path import join

# define the XML filepath
xml_path = '/Users/mackenzie/Desktop/04.23.21-iceo-test/results/benchmark/Pascall_and_Squires_2009_experimental_data.xml'
save_path = '/Users/mackenzie/Desktop/04.23.21-iceo-test/results/benchmark'
save_name = 'Pascall_and_Squires_2009_SMoutputs.csv'
save = join(save_path, save_name)

# create parser
tree = ET.parse(xml_path)
root = tree.getroot()

# ------ PARSE ------

# VOLTAGES
voltages = []                           # (V)
for i in root.iter('voltage'):
    voltages.append(i.text)
voltages = np.array(voltages, dtype=float)

# ELECTRODE SPACING
electrode_spacings = []                 # (m)
for i in root.iter('elecspace'):
    electrode_spacings.append(i.text)
electrode_spacings = np.array(electrode_spacings, dtype=float)

# -- COMPUTE -- ELECTRIC FIELD STRENGTH
electric_fields = voltages / electrode_spacings     # (V/m)

# FREQUENCIES
frequencys = []                         # (Hz)
for i in root.iter('frequency'):
    frequencys.append(i.text)
frequencys = np.array(frequencys, dtype=float)

# DIELECTRICS
dielectrics = []                            # name of dielectric coating
d_eps = []                                  # relative dielectric constant
d_pKa = []                                  # pKa of surface acid groups
d_pKb = []                                  # pKb of surface acid groups
d_Ns = []                                   # (#/m^2) acid site surface density
d_thick = []                                # (nm)
for i in root.iter('dielectric'):
    for j in i.iter('name'):
        for k in j.iter('item'):
            dielectrics.append(k.text)
    for j in i.iter('eps'):
        d_eps.append(j.text)
    for j in i.iter('pKa'):
        d_pKa.append(j.text)
    for j in i.iter('pKb'):
        d_pKb.append(j.text)
    for j in i.iter('Ns'):
        d_Ns.append(j.text)
    for j in i.iter('dithick'):
        d_thick.append(j.text)
d_eps = np.array(d_eps, dtype=float)
d_pKa = np.array(d_pKa, dtype=float)
d_pKb = np.array(d_pKb, dtype=float)
d_Ns = np.array(d_Ns, dtype=float)
d_thick = np.array(d_thick, dtype=float)

# BUFFERS
buffers = []
b_conc = []
b_conduct = []
b_pH = []
b_viscosity = []
b_eps = []
b_debye = []
for i in root.iter('buffer'):
    for j in i.iter('spec'):
        for k in j.iter('item'):
            buffers.append(k.text)
    for j in i.iter('conc'):
        b_conc.append(j.text)
    for j in i.iter('conduct'):
        b_conduct.append(j.text)
    for j in i.iter('pH'):
        b_pH.append(j.text)
    for j in i.iter('viscosity'):
        b_viscosity.append(j.text)
    for j in i.iter('eps'):
        b_eps.append(j.text)
    for j in i.iter('debye'):
        b_debye.append(j.text)
b_conc = np.array(b_conc, dtype=float)
b_conduct = np.array(b_conduct, dtype=float)
b_pH = np.array(b_pH, dtype=float)
b_viscosity = np.array(b_viscosity, dtype=float)
b_eps = np.array(b_eps, dtype=float)
b_debye = np.array(b_debye, dtype=float)

# BETA
beta = []                       # ratio of buffer capacitance to dielectric capacitance
for i in root.iter('beta'):
    beta.append(i.text)
beta = np.array(beta, dtype=float)

# DELTA
delta = []                      # ratio of dblyr capacitance to dielectric capacitance
for i in root.iter('delta'):
    delta.append(i.text)
delta = np.array(delta, dtype=float)

# TAU
tau = []                        # Double layer charging time computed in bare metal cases
for i in root.iter('tau'):
    tau.append(i.text)
tau = np.array(tau, dtype=float)

# PIV CORRECTION FACTOR
pivcorr = []
for i in root.iter('pivcorr'):
    pivcorr.append(i.text)
pivcorr = np.array(pivcorr, dtype=float)

# SLOPE NORMALZIED BY THEORETICAL SLOPE (eps*E^2/mu)
UbyUo = []
for i in root.iter('UbyUo'):
    UbyUo.append(i.text)
UbyUo = np.array(UbyUo, dtype=float)

# --- COMPUTED --- TRUE ESTIMATED VELOCITY
U_true_est = UbyUo * pivcorr

# DIELECTRICS
raw_uvel = []                       # (um/s)
raw_slope = []                      # slope of y-averaged uvel of middle 'interro' microns of electrode
for i in root.iter('raw'):
    for j in i.iter('uvel'):
        uvel = j.text[1:-2].split()
        uvel = [u.replace(";", "") for u in uvel]
        max_uvel = np.max(np.array(uvel, dtype=float))
        raw_uvel.append(max_uvel)
    for j in i.iter('slope'):
        raw_slope.append(j.text)
raw_uvel_max = np.array(raw_uvel, dtype=float)
raw_slope = np.array(raw_slope, dtype=float)


testdata = np.vstack((electric_fields, frequencys, dielectrics, buffers,
                           U_true_est, UbyUo, raw_uvel_max, raw_slope, beta, delta, tau,
                           d_eps, d_pKa, d_pKb, d_Ns, d_thick,
                           b_conc, b_conduct, b_pH, b_viscosity, b_eps, b_debye,
                           voltages, electrode_spacings, pivcorr)).T

header = "electric_fields,frequencys,dielectrics,buffers,U_true_est,UbyUo,raw_uvel_max,raw_slope,beta,delta,tau,d_eps,d_pKa,d_pKb,d_Ns,d_thick,b_conc,b_conduct,b_pH,b_viscosity,b_eps,b_debye,voltages,electrode_spacings,pivcorr"

# Write to .csv file
np.savetxt(save, testdata, fmt='%s', delimiter=',', header=header)
