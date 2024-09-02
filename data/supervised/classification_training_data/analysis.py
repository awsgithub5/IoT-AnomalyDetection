import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft

# Function to read and process a single file
def process_file(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Assuming columns are named: 'Time', 'X', 'Y', 'Z'
    time = df['Time(sec)'].values
    x = df['Vib Waveform Data(Vib-X) Acceleration (g)'].values
    y = df['Vib Waveform Data(Vib-Y) Acceleration (g)'].values
    z = df['Vib Waveform Data(Vib-Z) Acceleration (g)'].values
    
    # Calculate frequency for FFT
    sample_rate = 1 / (time[1] - time[0])
    freq = np.fft.fftfreq(len(time), 1/sample_rate)
    
    # Perform FFT
    x_fft = np.abs(fft(x))
    y_fft = np.abs(fft(y))
    z_fft = np.abs(fft(z))
    
    return time, x, y, z, freq, x_fft, y_fft, z_fft

# Function to plot waveform
def plot_waveform(time, x, y, z, filename):
    plt.figure(figsize=(12, 8))
    plt.plot(time, x, label='X')
    plt.plot(time, y, label='Y')
    plt.plot(time, z, label='Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.title(f'Waveform - {filename}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Fan/Unbalance/plots/{filename}_waveform.png')
    plt.close()

# Main function to process all files in a folder
def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  # Assuming CSV files
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            
            # Process the file
            time, x, y, z, freq, x_fft, y_fft, z_fft = process_file(file_path)
            
            # Generate plots
            plot_waveform(time, x, y, z, filename[:-4])  # Remove .csv extension

# Specify the folder path containing your vibration data files
folder_path = 'Fan/Unbalance'

# Process all files in the folder
process_folder(folder_path)