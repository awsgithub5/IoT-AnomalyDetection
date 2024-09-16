import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import rfft
from scipy.signal import welch

def extract_features(data):
    features = []
    for axis in ['X', 'Y', 'Z']:
        axis_data = data[axis].values
        
        # Time domain features
        features.extend([
            np.mean(axis_data),
            np.std(axis_data),
            np.max(axis_data),
            np.min(axis_data),
            kurtosis(axis_data),
            skew(axis_data)
        ])
        
        # Frequency domain features
        fft_vals = np.abs(rfft(axis_data))
        features.extend([
            np.mean(fft_vals),
            np.std(fft_vals),
            np.max(fft_vals),
            np.argmax(fft_vals)
        ])
        
        # Power spectral density features
        freqs, psd = welch(axis_data)
        features.extend([
            np.sum(psd),
            np.mean(psd),
            np.std(psd),
            np.max(psd),
            freqs[np.argmax(psd)]
        ])
    
    return np.array(features)

def process_data(file_names):
    all_features = []
    for file in file_names:
        df = pd.read_csv(file, header=None, skiprows=1)
        features = extract_features(df[[4, 5, 6]].rename(columns={4: 'X', 5: 'Y', 6: 'Z'}))
        all_features.append(features)
    return np.array(all_features)

# Main execution
data_path = '../../data/supervised/classification_training_data/Fan'
normal_files = glob.glob(os.path.join(data_path, 'Normal', '*.csv'))
misalignment_files = glob.glob(os.path.join(data_path, 'Misalignment', '*.csv'))
unbalance_files = glob.glob(os.path.join(data_path, 'Unbalance', '*.csv'))
looseness_files = glob.glob(os.path.join(data_path, 'Looseness', '*.csv'))
impact_files = glob.glob(os.path.join(data_path, 'Impact', '*.csv'))

x_normal = process_data(normal_files)
x_misalignment = process_data(misalignment_files)
x_unbalance = process_data(unbalance_files)
x_looseness = process_data(looseness_files)
x_impact = process_data(impact_files)

x = np.vstack((x_normal, x_misalignment, x_unbalance, x_looseness, x_impact))
y = np.concatenate((
    np.zeros(len(x_normal)),
    np.ones(len(x_misalignment)),
    np.full(len(x_unbalance), 2),
    np.full(len(x_looseness), 3),
    np.full(len(x_impact), 4)
))

# Save features and labels
np.savetxt('../../data/supervised/classification_training_data/Fan/feature_VBL-VA001.csv', x, delimiter=',')
np.savetxt('../../data/supervised/classification_training_data/Fan/label_VBL-VA001.csv', y, delimiter=',')

print(f"Shape of features: {x.shape}")
print(f"Shape of labels: {y.shape}")