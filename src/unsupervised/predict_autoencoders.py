import numpy as np
import pandas as pd
from tensorflow import keras
import os
import shutil
import joblib
import matplotlib.pyplot as plt

# Constants
TIME_STEPS = 288
IOT_DATA_PATH = '../../data/unsupervised/raw_data'
OUTPUT_NORMAL_PATH = '../../data/unsupervised/output_normal'
OUTPUT_ABNORMAL_PATH = '../../data/unsupervised/output_abnormal'
MODEL_PATH = '../../models/unsupervised'
PARAMS_PATH = '../../models/unsupervised/saved_params.joblib'
PLOTS_PATH = '../../plots/unsupervised'

def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def predict_anomalies(model, data, training_mean, training_std, threshold):
    df_test_value = (data - training_mean) / training_std
    x_test = create_sequences(df_test_value.values)
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    return test_mae_loss > threshold

def plot_dataset(data, is_anomaly, filename):
    plt.figure(figsize=(15, 8))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    
    plt.legend()
    plt.title(f'{"Anomaly" if is_anomaly else "Normal"} Dataset - {filename}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    plot_dir = os.path.join(PLOTS_PATH, 'anomaly' if is_anomaly else 'normal')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{filename}.png"))
    plt.close()

def process_and_plot_folder(folder_path, is_anomaly):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, parse_dates=True, index_col="Time(sec)")
            df = df.iloc[:, 4:]  # Assuming we want to skip the first 4 columns
            plot_dataset(df, is_anomaly, filename[:-4])  # Remove .csv extension

def main():
    # Load the saved model
    model = keras.models.load_model(MODEL_PATH)

    # Load saved parameters
    params = joblib.load(PARAMS_PATH)
    training_mean = params['training_mean']
    training_std = params['training_std']
    threshold = params['threshold']

    # Process IOT data
    for filename in os.listdir(IOT_DATA_PATH):
        if filename.endswith('.csv'):
            file_path = os.path.join(IOT_DATA_PATH, filename)
            df = pd.read_csv(file_path, parse_dates=True, index_col="Time(sec)")
            df = df.iloc[:, 4:]  # Assuming we want to skip the first 4 columns
            
            anomalies = predict_anomalies(model, df, training_mean, training_std, threshold)          
            
            # Determine if the file is normal or abnormal
            if np.any(anomalies):
                shutil.copy(file_path, os.path.join(OUTPUT_ABNORMAL_PATH, filename))
                print(f"Anomaly detected in {filename}. Moved to abnormal folder.")
            else:
                shutil.copy(file_path, os.path.join(OUTPUT_NORMAL_PATH, filename))
                print(f"No anomaly detected in {filename}. Moved to normal folder.")

    # Plot normal and abnormal datasets
    print("Plotting normal datasets...")
    process_and_plot_folder(OUTPUT_NORMAL_PATH, is_anomaly=False)
    
    print("Plotting abnormal datasets...")
    process_and_plot_folder(OUTPUT_ABNORMAL_PATH, is_anomaly=True)

    print("Processing complete. Check the output folders for categorized files and plots.")

if __name__ == "__main__":
    main()