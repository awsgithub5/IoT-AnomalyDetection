# Script to train

# importation
from numpy.fft import rfft, rfftfreq
from sklearn import preprocessing
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
data_path = '../../data/supervised/classification_training_data/Fan'

totalFiles = 0
totalDir = 0

for base, dirs, files in os.walk(data_path):
    print('Searching in : ', base)
    for directories in dirs:
        totalDir += 1
    for Files in files:
        totalFiles += 1

print('Total number of files', totalFiles)
print('Total number of directories', totalDir)

# Collecting number data
dir_path1 = data_path + '/Normal/'
print('Total data Normal :', len([entry for entry in os.listdir(
    dir_path1) if os.path.isfile(os.path.join(dir_path1, entry))]))
dir_path2 = data_path + '/Misalignment/'
print('Total data misalignment :', len([entry for entry in os.listdir(
    dir_path2) if os.path.isfile(os.path.join(dir_path2, entry))]))
dir_path3 = data_path + '/Unbalance/'
print('Total data unbalance :', len([entry for entry in os.listdir(
    dir_path3) if os.path.isfile(os.path.join(dir_path3, entry))]))
dir_path4 = data_path + '/Looseness/'
print('Total data bearing fault:', len([entry for entry in os.listdir(
    dir_path4) if os.path.isfile(os.path.join(dir_path4, entry))]))
dir_path5 = data_path + '/Impact/'
print('Total data Impact fault:', len([entry for entry in os.listdir(
    dir_path4) if os.path.isfile(os.path.join(dir_path4, entry))]))

# Collecting file names
normal_file_names = glob.glob(data_path + '/Normal/*.csv')
print("AAAAAAAAAAAA",normal_file_names)
imnormal_misalignment = glob.glob(data_path + '/Misalignment/*.csv')
imnormal_unbalance = glob.glob(data_path + '/Unbalance/*.csv')
imnormal_looseness = glob.glob(data_path + '/Looseness/*.csv')
imnormal_impact = glob.glob(data_path + '/Impact/*.csv')


def FFT(data):
    '''FFT process, take real values only'''
    print("@@@@@@@@@@@@",data)
    data = np.asarray(data)
    n = len(data)
    dt = 1/20000  # time increment in each data
    data = rfft(data)*dt
    freq = rfftfreq(n, dt)
    data = abs(data).T
    data = (np.delete(data, range(500*5,len(data)), axis=0)).T
    print("Inside FFT",data)
    return (data)

# Feature Extraction function
def std(data):
    '''Standard Deviation features'''
    data = np.asarray(data)
    stdev = pd.DataFrame(np.std(data, axis=1))
    return stdev


def mean(data):
    '''Mean features'''
    data = np.asarray(data)
    M = pd.DataFrame(np.mean(data, axis=1))
    return M


def pp(data):
    '''Peak-to-Peak features'''
    data = np.asarray(data)
    PP = pd.DataFrame(np.max(data, axis=1) - np.min(data, axis=1))
    return PP


def Variance(data):
    '''Variance features'''
    data = np.asarray(data)
    Var = pd.DataFrame(np.var(data, axis=1))
    return Var


def rms(data):
    '''RMS features'''
    data = np.asarray(data)
    Rms = pd.DataFrame(np.sqrt(np.mean(data**2, axis=1)))
    return Rms


def Shapef(data):
    '''Shape factor features'''
    data = np.asarray(data)
    shapef = pd.DataFrame(rms(data)/Ab_mean(data))
    return shapef


def Impulsef(data):
    '''Impulse factor features'''
    data = np.asarray(data)
    impulse = pd.DataFrame(np.max(data)/Ab_mean(data))
    return impulse


def crestf(data):
    '''Crest factor features'''
    data = np.asarray(data)
    crest = pd.DataFrame(np.max(data)/rms(data))
    return crest


def kurtosis(data):
    '''Kurtosis features'''
    data = pd.DataFrame(data)
    kurt = data.kurt(axis=1)
    return kurt


def skew(data):
    '''Skewness features'''
    data = pd.DataFrame(data)
    skw = data.skew(axis=1)
    return skw


# Helper functions to calculate features
def Ab_mean(data):
    data = np.asarray(data)
    Abm = pd.DataFrame(np.mean(np.absolute(data), axis=1))
    return Abm


def SQRT_AMPL(data):
    data = np.asarray(data)
    SQRTA = pd.DataFrame((np.mean(np.sqrt(np.absolute(data, axis=1))))**2)
    return SQRTA


def clearancef(data):
    data = np.asarray(data)
    clrf = pd.DataFrame(np.max(data, axis=1)/SQRT_AMPL(data))
    return clrf


# Extract features from X, Y, Z axis
def data_1x(normal_file_names):
    data1x = pd.DataFrame()
    for f1x in normal_file_names:
        df1x = pd.read_csv(f1x, usecols=[5], header=None) 
          # read the csv file
        data1x = pd.concat([data1x, df1x], axis=1, ignore_index=True)
       
    return data1x


def data_1y(normal_file_names):
    data1y = pd.DataFrame()
    for f1y in normal_file_names: 
        df1y = pd.read_csv(f1y, usecols=[6], header=None)  # read the csv file
        data1y = pd.concat([data1y, df1y], axis=1, ignore_index=True)       
    return data1y


def data_1z(normal_file_names):
    data1z = pd.DataFrame()
    for f1z in normal_file_names:
        df1z = pd.read_csv(f1z, usecols=[7], header=None)  # read the csv file
        data1z = pd.concat([data1z, df1z], axis=1, ignore_index=True)
    return data1z


def data_2x(imnormal_misalignment):
    data2x = pd.DataFrame()
    for f2x in imnormal_misalignment:
        df2x = pd.read_csv(f2x, usecols=[5], header=None)  # read the csv file
        data2x = pd.concat([data2x, df2x], axis=1, ignore_index=True)
    return data2x


def data_2y(imnormal_misalignment):
    data2y = pd.DataFrame()
    for f2y in imnormal_misalignment:
        df2y = pd.read_csv(f2y, usecols=[6], header=None)  # read the csv file
        data2y = pd.concat([data2y, df2y], axis=1, ignore_index=True)
    return data2y


def data_2z(imnormal_misalignment):
    data2z = pd.DataFrame()
   
    for f2z in imnormal_misalignment:
        df2z = pd.read_csv(f2z, usecols=[7], header=None)  # read the csv file
        data2z = pd.concat([data2z, df2z], axis=1, ignore_index=True)
    return data2z


def data_3x(imnormal_unbalance):
    data3x = pd.DataFrame()
    for f3x in imnormal_unbalance:
        df3x = pd.read_csv(f3x, usecols=[5], header=None)  # read the csv file
        data3x = pd.concat([data3x, df3x], axis=1, ignore_index=True)
    return data3x


def data_3y(imnormal_unbalance):
    data3y = pd.DataFrame()
    for f3y in imnormal_unbalance:
        df3y = pd.read_csv(f3y, usecols=[6], header=None)  # read the csv file
        data3y = pd.concat([data3y, df3y], axis=1, ignore_index=True)
    return data3y


def data_3z(imnormal_unbalance):
    data3z = pd.DataFrame()
    for f3z in imnormal_unbalance:
        df3z = pd.read_csv(f3z, usecols=[7], header=None)  # read the csv file
        data3z = pd.concat([data3z, df3z], axis=1, ignore_index=True)
    return data3z


def data_4x(imnormal_bearing):
    data4x = pd.DataFrame()
    for f4x in imnormal_bearing:
        df4x = pd.read_csv(f4x, usecols=[5], header=None)  # read the csv file
        data4x = pd.concat([data4x, df4x], axis=1, ignore_index=True)
    return data4x


def data_4y(imnormal_bearing):
    data4y = pd.DataFrame()
    for f4y in imnormal_bearing:
        df4y = pd.read_csv(f4y, usecols=[6], header=None)  # read the csv file
        data4y = pd.concat([data4y, df4y], axis=1, ignore_index=True)
    return data4y


def data_4z(imnormal_bearing):
    data4z = pd.DataFrame()
    for f4z in imnormal_bearing:
        df4z = pd.read_csv(f4z, usecols=[7], header=None)  # read the csv file
        data4z = pd.concat([data4z, df4z], axis=1, ignore_index=True)
    return data4z


# Data normal transpose x y z, remove NaN
data_normal_x = data_1x(normal_file_names).T.dropna(axis=1)
#data_normal_x = data_normal_x[:, 1:]
print("data_normal_x",data_normal_x.shape)
data_normal_x = data_normal_x.iloc[:, 2:]
data_normal_y = data_1y(normal_file_names).T.dropna(axis=1)
data_normal_y = data_normal_y.iloc[:, 2:]
data_normal_z = data_1z(normal_file_names).T.dropna(axis=1)
data_normal_z = data_normal_z.iloc[:, 2:]

# Data misalignment transpose x y z
data_misalignment_x = data_2x(imnormal_misalignment).T.dropna(axis=1)
data_misalignment_x = data_misalignment_x.iloc[:, 2:]
data_misalignment_y = data_2y(imnormal_misalignment).T.dropna(axis=1)
data_misalignment_y = data_misalignment_y.iloc[:, 2:]
data_misalignment_z = data_2z(imnormal_misalignment).T.dropna(axis=1)
data_misalignment_z = data_misalignment_z.iloc[:, 2:]

# Data unbalance transpose x y z
data_unbalance_x = data_3x(imnormal_unbalance).T.dropna(axis=1)
data_unbalance_x = data_unbalance_x.iloc[:, 2:]
data_unbalance_y = data_3y(imnormal_unbalance).T.dropna(axis=1)
data_unbalance_y = data_unbalance_y.iloc[:, 2:]
data_unbalance_z = data_3z(imnormal_unbalance).T.dropna(axis=1)
data_unbalance_z = data_unbalance_z.iloc[:, 2:]

# Data looseness transpose x y z
data_looseness_x = data_4x(imnormal_looseness).T.dropna(axis=1)
data_looseness_x = data_looseness_x.iloc[:, 2:]
data_looseness_y = data_4y(imnormal_looseness).T.dropna(axis=1)
data_looseness_y = data_looseness_y.iloc[:, 2:]
data_looseness_z = data_4z(imnormal_looseness).T.dropna(axis=1)
data_looseness_z = data_looseness_z.iloc[:, 2:]

# Data impact transpose x y z
data_impact_x = data_4x(imnormal_impact).T.dropna(axis=1)
data_impact_x = data_impact_x.iloc[:, 2:]
data_impact_y = data_4y(imnormal_impact).T.dropna(axis=1)
data_impact_y = data_impact_y.iloc[:, 2:]
data_impact_z = data_4z(imnormal_impact).T.dropna(axis=1)
data_impact_z = data_impact_z.iloc[:, 2:]

# # Concatenate data for each X, Y, and Z
# data_x = np.concatenate(
#     (data_normal_x, data_misalignment_x, data_unbalance_x, data_bearing_x))
# data_y = np.concatenate(
#     (data_normal_y, data_misalignment_y, data_unbalance_y, data_bearing_y))
# data_z = np.concatenate(
#     (data_normal_z, data_misalignment_z, data_unbalance_z, data_bearing_z))


# Doing FFT
fft_1x = FFT(data_normal_x)
fft_1y = FFT(data_normal_y)
fft_1z = FFT(data_normal_z)

fft_2x = FFT(data_misalignment_x)
fft_2y = FFT(data_misalignment_y)
fft_2z = FFT(data_misalignment_z)

fft_3x = FFT(data_unbalance_x)
fft_3y = FFT(data_unbalance_y)
fft_3z = FFT(data_unbalance_z)

fft_4x = FFT(data_looseness_x)
fft_4y = FFT(data_looseness_y)
fft_4z = FFT(data_looseness_z)

fft_5x = FFT(data_impact_x)
fft_5y = FFT(data_impact_y)
fft_5z = FFT(data_impact_z)

# merge data
data_merged = np.concatenate((fft_1x, fft_2x, fft_3x, fft_4x, fft_5x,
                             fft_1y, fft_2y, fft_3y, fft_4y, fft_5y, fft_1z, fft_2z, fft_3z, fft_4z,fft_5z))


# normalize data
def NormalizeData(data):  # Normalisasi (0-1)
    data_max = np.max(data_merged)
    data_min = np.min(data_merged)
    return (data - np.min(data_min)) / (np.max(data_max) - np.min(data_min))


fft_1x = NormalizeData(fft_1x)
fft_1y = NormalizeData(fft_1y)
fft_1z = NormalizeData(fft_1z)

fft_2x = NormalizeData(fft_2x)
fft_2y = NormalizeData(fft_2y)
fft_2z = NormalizeData(fft_2z)

fft_3x = NormalizeData(fft_3x)
fft_3y = NormalizeData(fft_3y)
fft_3z = NormalizeData(fft_3z)

fft_4x = NormalizeData(fft_4x)
fft_4y = NormalizeData(fft_4y)
fft_4z = NormalizeData(fft_4z)

fft_5x = NormalizeData(fft_5x)
fft_5y = NormalizeData(fft_5y)
fft_5z = NormalizeData(fft_5z)


# Feature Extraction
# shape factor
shapef_1x = Shapef(fft_1x)
shapef_1y = Shapef(fft_1y)
shapef_1z = Shapef(fft_1z)
shapef_2x = Shapef(fft_2x)
shapef_2y = Shapef(fft_2y)
shapef_2z = Shapef(fft_2z)
shapef_3x = Shapef(fft_3x)
shapef_3y = Shapef(fft_3y)
shapef_3z = Shapef(fft_3z)
shapef_4x = Shapef(fft_4x)
shapef_4y = Shapef(fft_4y)
shapef_4z = Shapef(fft_4z)
shapef_5x = Shapef(fft_5x)
shapef_5y = Shapef(fft_5y)
shapef_5z = Shapef(fft_5z)


shapef_1 = pd.concat([shapef_1x, shapef_1y, shapef_1z],
                     axis=1, ignore_index=True)
shapef_2 = pd.concat([shapef_2x, shapef_2y, shapef_2z],
                     axis=1, ignore_index=True)
shapef_3 = pd.concat([shapef_3x, shapef_3y, shapef_3z],
                     axis=1, ignore_index=True)
shapef_4 = pd.concat([shapef_4x, shapef_4y, shapef_4z],
                     axis=1, ignore_index=True)
shapef_5 = pd.concat([shapef_5x, shapef_5y, shapef_5z],
                     axis=1, ignore_index=True)

# root mean square
rms_1x = rms(fft_1x)
rms_1y = rms(fft_1y)
rms_1z = rms(fft_1z)
rms_2x = rms(fft_2x)
rms_2y = rms(fft_2y)
rms_2z = rms(fft_2z)
rms_3x = rms(fft_3x)
rms_3y = rms(fft_3y)
rms_3z = rms(fft_3z)
rms_4x = rms(fft_4x)
rms_4y = rms(fft_4y)
rms_4z = rms(fft_4z)
rms_5x = rms(fft_5x)
rms_5y = rms(fft_5y)
rms_5z = rms(fft_5z)

rms_1 = pd.concat([rms_1x, rms_1y, rms_1z], axis=1, ignore_index=True)
rms_2 = pd.concat([rms_2x, rms_2y, rms_2z], axis=1, ignore_index=True)
rms_3 = pd.concat([rms_3x, rms_3y, rms_3z], axis=1, ignore_index=True)
rms_4 = pd.concat([rms_4x, rms_4y, rms_4z], axis=1, ignore_index=True)
rms_5 = pd.concat([rms_5x, rms_5y, rms_5z], axis=1, ignore_index=True)

# impulse factor
Impulsef_1x = Impulsef(fft_1x)
Impulsef_1y = Impulsef(fft_1y)
Impulsef_1z = Impulsef(fft_1z)
Impulsef_2x = Impulsef(fft_2x)
Impulsef_2y = Impulsef(fft_2y)
Impulsef_2z = Impulsef(fft_2z)
Impulsef_3x = Impulsef(fft_3x)
Impulsef_3y = Impulsef(fft_3y)
Impulsef_3z = Impulsef(fft_3z)
Impulsef_4x = Impulsef(fft_4x)
Impulsef_4y = Impulsef(fft_4y)
Impulsef_4z = Impulsef(fft_4z)
Impulsef_5x = Impulsef(fft_5x)
Impulsef_5y = Impulsef(fft_5y)
Impulsef_5z = Impulsef(fft_5z)

Impulsef_1 = pd.concat(
    [Impulsef_1x, Impulsef_1y, Impulsef_1z], axis=1, ignore_index=True)
Impulsef_2 = pd.concat(
    [Impulsef_2x, Impulsef_2y, Impulsef_2z], axis=1, ignore_index=True)
Impulsef_3 = pd.concat(
    [Impulsef_3x, Impulsef_3y, Impulsef_3z], axis=1, ignore_index=True)
Impulsef_4 = pd.concat(
    [Impulsef_4x, Impulsef_4y, Impulsef_4z], axis=1, ignore_index=True)
Impulsef_5 = pd.concat(
    [Impulsef_5x, Impulsef_5y, Impulsef_5z], axis=1, ignore_index=True)

# peak factor
pp_1x = pp(fft_1x)
pp_1y = pp(fft_1y)
pp_1z = pp(fft_1z)
pp_2x = pp(fft_2x)
pp_2y = pp(fft_2y)
pp_2z = pp(fft_2z)
pp_3x = pp(fft_3x)
pp_3y = pp(fft_3y)
pp_3z = pp(fft_3z)
pp_4x = pp(fft_4x)
pp_4y = pp(fft_4y)
pp_4z = pp(fft_4z)
pp_5x = pp(fft_5x)
pp_5y = pp(fft_5y)
pp_5z = pp(fft_5z)

pp_1 = pd.concat([pp_1x, pp_1y, pp_1z], axis=1, ignore_index=True)
pp_2 = pd.concat([pp_2x, pp_2y, pp_2z], axis=1, ignore_index=True)
pp_3 = pd.concat([pp_3x, pp_3y, pp_3z], axis=1, ignore_index=True)
pp_4 = pd.concat([pp_4x, pp_4y, pp_4z], axis=1, ignore_index=True)
pp_5 = pd.concat([pp_5x, pp_5y, pp_5z], axis=1, ignore_index=True)

# kurtosis factor
kurtosis_1x = kurtosis(fft_1x)
kurtosis_1y = kurtosis(fft_1y)
kurtosis_1z = kurtosis(fft_1z)
kurtosis_2x = kurtosis(fft_2x)
kurtosis_2y = kurtosis(fft_2y)
kurtosis_2z = kurtosis(fft_2z)
kurtosis_3x = kurtosis(fft_3x)
kurtosis_3y = kurtosis(fft_3y)
kurtosis_3z = kurtosis(fft_3z)
kurtosis_4x = kurtosis(fft_4x)
kurtosis_4y = kurtosis(fft_4y)
kurtosis_4z = kurtosis(fft_4z)
kurtosis_5x = kurtosis(fft_5x)
kurtosis_5y = kurtosis(fft_5y)
kurtosis_5z = kurtosis(fft_5z)

kurtosis_1 = pd.concat(
    [kurtosis_1x, kurtosis_1y, kurtosis_1z], axis=1, ignore_index=True)
kurtosis_2 = pd.concat(
    [kurtosis_2x, kurtosis_2y, kurtosis_2z], axis=1, ignore_index=True)
kurtosis_3 = pd.concat(
    [kurtosis_3x, kurtosis_3y, kurtosis_3z], axis=1, ignore_index=True)
kurtosis_4 = pd.concat(
    [kurtosis_4x, kurtosis_4y, kurtosis_4z], axis=1, ignore_index=True)
kurtosis_5 = pd.concat(
    [kurtosis_5x, kurtosis_5y, kurtosis_5z], axis=1, ignore_index=True)

# crest factor
crestf_1x = crestf(fft_1x)
crestf_1y = crestf(fft_1y)
crestf_1z = crestf(fft_1z)
crestf_2x = crestf(fft_2x)
crestf_2y = crestf(fft_2y)
crestf_2z = crestf(fft_2z)
crestf_3x = crestf(fft_3x)
crestf_3y = crestf(fft_3y)
crestf_3z = crestf(fft_3z)
crestf_4x = crestf(fft_4x)
crestf_4y = crestf(fft_4y)
crestf_4z = crestf(fft_4z)
crestf_5x = crestf(fft_5x)
crestf_5y = crestf(fft_5y)
crestf_5z = crestf(fft_5z)

crestf_1 = pd.concat([crestf_1x, crestf_1y, crestf_1z],
                     axis=1, ignore_index=True)
crestf_2 = pd.concat([crestf_2x, crestf_2y, crestf_2z],
                     axis=1, ignore_index=True)
crestf_3 = pd.concat([crestf_3x, crestf_3y, crestf_3z],
                     axis=1, ignore_index=True)
crestf_4 = pd.concat([crestf_4x, crestf_4y, crestf_4z],
                     axis=1, ignore_index=True)
crestf_5 = pd.concat([crestf_5x, crestf_5y, crestf_5z],
                     axis=1, ignore_index=True)

# mean
mean_1x = mean(fft_1x)
mean_1y = mean(fft_1y)
mean_1z = mean(fft_1z)
mean_2x = mean(fft_2x)
mean_2y = mean(fft_2y)
mean_2z = mean(fft_2z)
mean_3x = mean(fft_3x)
mean_3y = mean(fft_3y)
mean_3z = mean(fft_3z)
mean_4x = mean(fft_4x)
mean_4y = mean(fft_4y)
mean_4z = mean(fft_4z)
mean_5x = mean(fft_5x)
mean_5y = mean(fft_5y)
mean_5z = mean(fft_5z)

mean_1 = pd.concat([mean_1x, mean_1y, mean_1z], axis=1, ignore_index=True)
mean_2 = pd.concat([mean_2x, mean_2y, mean_2z], axis=1, ignore_index=True)
mean_3 = pd.concat([mean_3x, mean_3y, mean_3z], axis=1, ignore_index=True)
mean_4 = pd.concat([mean_4x, mean_4y, mean_4z], axis=1, ignore_index=True)
mean_5 = pd.concat([mean_5x, mean_5y, mean_5z], axis=1, ignore_index=True)

# std
std_1x = std(fft_1x)
std_1y = std(fft_1y)
std_1z = std(fft_1z)
std_2x = std(fft_2x)
std_2y = std(fft_2y)
std_2z = std(fft_2z)
std_3x = std(fft_3x)
std_3y = std(fft_3y)
std_3z = std(fft_3z)
std_4x = std(fft_4x)
std_4y = std(fft_4y)
std_4z = std(fft_4z)
std_5x = std(fft_5x)
std_5y = std(fft_5y)
std_5z = std(fft_5z)


std_1 = pd.concat([std_1x, std_1y, std_1z], axis=1, ignore_index=True)
std_2 = pd.concat([std_2x, std_2y, std_2z], axis=1, ignore_index=True)
std_3 = pd.concat([std_3x, std_3y, std_3z], axis=1, ignore_index=True)
std_4 = pd.concat([std_4x, std_4y, std_4z], axis=1, ignore_index=True)
std_5 = pd.concat([std_5x, std_5y, std_5z], axis=1, ignore_index=True)

# skew
skew_1x = skew(fft_1x)
skew_1y = skew(fft_1y)
skew_1z = skew(fft_1z)
skew_2x = skew(fft_2x)
skew_2y = skew(fft_2y)
skew_2z = skew(fft_2z)
skew_3x = skew(fft_3x)
skew_3y = skew(fft_3y)
skew_3z = skew(fft_3z)
skew_4x = skew(fft_4x)
skew_4y = skew(fft_4y)
skew_4z = skew(fft_4z)
skew_5x = skew(fft_5x)
skew_5y = skew(fft_5y)
skew_5z = skew(fft_5z)

skew_1 = pd.concat([skew_1x, skew_1y, skew_1z], axis=1, ignore_index=True)
skew_2 = pd.concat([skew_2x, skew_2y, skew_2z], axis=1, ignore_index=True)
skew_3 = pd.concat([skew_3x, skew_3y, skew_3z], axis=1, ignore_index=True)
skew_4 = pd.concat([skew_4x, skew_4y, skew_4z], axis=1, ignore_index=True)
skew_5 = pd.concat([skew_5x, skew_5y, skew_5z], axis=1, ignore_index=True)

x_1 = pd.concat([mean_1, std_1, shapef_1, rms_1, Impulsef_1,
                pp_1, kurtosis_1, crestf_1, skew_1], axis=1, ignore_index=True)
x_2 = pd.concat([mean_2, std_2, shapef_2, rms_2, Impulsef_2,
                pp_2, kurtosis_2, crestf_2, skew_2], axis=1, ignore_index=True)
x_3 = pd.concat([mean_3, std_3, shapef_3, rms_3, Impulsef_3,
                pp_3, kurtosis_3, crestf_3, skew_3], axis=1, ignore_index=True)
x_4 = pd.concat([mean_4, std_4, shapef_4, rms_4, Impulsef_4,
                pp_4, kurtosis_4, crestf_4, skew_4], axis=1, ignore_index=True)
x_5 = pd.concat([mean_5, std_5, shapef_5, rms_5, Impulsef_5,
                pp_5, kurtosis_5, crestf_5, skew_5], axis=1, ignore_index=True)

x = pd.concat([x_1, x_2, x_3, x_4, x_5], axis=0, ignore_index=True)
x = np.asarray(x)
print(f"Shape of feature: {x.shape}")


fft_x = pd.DataFrame(x).to_csv(
    '../../data/supervised/classification_training_data/Fan/feature_VBL-VA001.csv', index=None, header=False)

y_1 = np.full((int(len(x_1)), 1), 0)
y_2 = np.full((int(len(x_2)), 1), 1)
y_3 = np.full((int(len(x_3)), 1), 2)
y_4 = np.full((int(len(x_4)), 1), 3)
y_5 = np.full((int(len(x_5)), 1), 4)
y = np.concatenate((y_1, y_2, y_3, y_4, y_5), axis=None)
#y = pd.DataFrame(y)
print(f"Shape of labels: {y.shape}")


y_label = pd.DataFrame(y).to_csv(
    '../../data/supervised/classification_training_data/Fan/label_VBL-VA001.csv', index=None, header=False)


def create_feature_names():
    feature_names = []
    for axis in ['X', 'Y', 'Z']:
        feature_names.extend([
            f'Mean_{axis}',
            f'Std_{axis}',
            f'ShapeFactor_{axis}',
            f'RMS_{axis}',
            f'ImpulseFactor_{axis}',
            f'PeakToPeak_{axis}',
            f'Kurtosis_{axis}',
            f'CrestFactor_{axis}',
            f'Skewness_{axis}'
        ])
    return feature_names

# Create the feature names
feature_names = create_feature_names()

def plot_features(data, feature_names):
    plt.figure(figsize=(20, 15))
    num_features = len(feature_names)
    rows = (num_features + 2) // 3  # Calculate number of rows needed
    for i, feature in enumerate(feature_names):
        plt.subplot(rows, 3, i+1)
        plt.plot(data.iloc[:, i])
        plt.title(feature)
    plt.tight_layout()
    plt.savefig('../../plots/supervised/feature_plots.png')
    plt.close()

# When you want to plot, read the CSV and use the feature names
data_for_plotting = pd.read_csv('../../data/supervised/classification_training_data/Fan/feature_VBL-VA001.csv', header=None)
plot_features(data_for_plotting, feature_names)