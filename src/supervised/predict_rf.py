import os
import shutil
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.fft import rfft
from scipy.signal import welch
from sklearn.preprocessing import LabelEncoder


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

def process_data(file_path):
    try:
        data = pd.read_csv(file_path, usecols=[4, 5, 6], names=['X', 'Y', 'Z'], header=0, skiprows=1)
        return extract_features(data).reshape(1, -1)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def process_folder_and_predict(input_folder, model_path):
    all_features = []
    filenames = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            features = process_data(file_path)
            if features is not None:
                all_features.append(features)
                filenames.append(filename)
    
    if not all_features:
        raise ValueError("No valid features extracted from any file")
    
    X = np.vstack(all_features)
    
    model = joblib.load(model_path)
    predictions = model.predict(X)
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return X, predictions, shap_values, filenames

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_data(data, filename, anomaly_type, output_folder):
    plt.figure(figsize=(12, 6))
    for axis in ['X', 'Y', 'Z']:
        plt.plot(data[axis], label=axis)
    plt.title(f'{anomaly_type} - {filename}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{filename[:-4]}_plot.png'))
    plt.close()

def plot_shap_values(shap_values, X, feature_names, anomaly_types, output_folder):
    for i, anomaly_type in enumerate(anomaly_types):
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[i], X, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f'SHAP Values for {anomaly_type}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{anomaly_type}_shap_plot.png'))
        plt.close()

if __name__ == "__main__":
    raw_data_folder = "../../data/supervised/test_data"
    model_path = '../../models/supervised/rf_model.joblib'
    output_base_folder = "../../results"
    
    X, predictions, shap_values, filenames = process_folder_and_predict(raw_data_folder, model_path)
    
    results = pd.DataFrame({
        'filename': filenames,
        'predicted_anomaly': predictions
    })

    feature_names = [
        f"{feat}_{axis}" for axis in ['X', 'Y', 'Z']
        for feat in ['Mean', 'Std', 'Max', 'Min', 'Kurtosis', 'Skew',
                     'FFT_Mean', 'FFT_Std', 'FFT_Max', 'FFT_ArgMax',
                     'PSD_Sum', 'PSD_Mean', 'PSD_Std', 'PSD_Max', 'PSD_PeakFreq']
    ]

    # Handle multi-class SHAP values
    anomaly_type_map = {
        0: "Normal",
        1: "Misalignment",
        2: "Unbalance",
        3: "Looseness",
        4: "Impact"
    }
    
    le = LabelEncoder()
    le.fit(list(anomaly_type_map.values()))
    
    # Create a DataFrame for each class's SHAP values
    shap_dfs = []
    for i, anomaly_type in enumerate(le.classes_):
        df = pd.DataFrame(shap_values[i], columns=feature_names)
        df['anomaly_type'] = anomaly_type
        shap_dfs.append(df)
    
    # Concatenate all SHAP DataFrames
    shap_df = pd.concat(shap_dfs, axis=0, ignore_index=True)
    shap_df['filename'] = np.repeat(filenames, len(le.classes_))
    shap_df['predicted_anomaly'] = np.repeat(predictions, len(le.classes_))

    features_df = pd.DataFrame(X, columns=feature_names)
    features_df['filename'] = filenames
    features_df['predicted_anomaly'] = predictions

    # Create output folders
    for anomaly_type in anomaly_type_map.values():
        anomaly_folder = os.path.join(output_base_folder, anomaly_type)
        plots_folder = os.path.join(anomaly_folder, 'plots')
        os.makedirs(anomaly_folder, exist_ok=True)
        os.makedirs(plots_folder, exist_ok=True)

    
    for index, row in results.iterrows():
        filename = row['filename']
        predicted_anomaly = row['predicted_anomaly']
        anomaly_name = anomaly_type_map.get(predicted_anomaly, f"unknown_{predicted_anomaly}")
        
        anomaly_folder = os.path.join(output_base_folder, anomaly_name)
        plots_folder = os.path.join(anomaly_folder, 'plots')
        
        src_path = os.path.join(raw_data_folder, filename)
        dst_path = os.path.join(anomaly_folder, filename)
        shutil.copy2(src_path, dst_path)
        
        data = pd.read_csv(src_path, usecols=[4, 5, 6], names=['X', 'Y', 'Z'], header=0, skiprows=1)
        plot_data(data, filename, anomaly_name, plots_folder)
        
        file_features = features_df[features_df['filename'] == filename]
        file_shap = shap_df[shap_df['filename'] == filename]
        
        features_csv = os.path.join(anomaly_folder, f'{anomaly_name}_features.csv')
        shap_csv = os.path.join(anomaly_folder, f'{anomaly_name}_shap_values.csv')
        
        file_features.to_csv(features_csv, mode='a', header=not os.path.exists(features_csv), index=False)
        file_shap.to_csv(shap_csv, mode='a', header=not os.path.exists(shap_csv), index=False)

   # Plot SHAP values for each class
    plot_shap_values(shap_values, X, feature_names, le.classes_, output_base_folder)

    print("Processing completed successfully.")