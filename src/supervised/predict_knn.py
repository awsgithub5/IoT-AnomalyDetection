import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import rfft
import joblib
import shap
import shutil 
import matplotlib.pyplot as plt
def extract_features(data):
    """
    Extract features from the input data for X, Y, and Z axes.
    
    :param data: DataFrame with columns for X, Y, and Z axis data
    :return: numpy array of extracted features
    """
    features = []
    
    for axis in ['X', 'Y', 'Z']:
        axis_data = data[axis].dropna().values
        print(f"Processing {axis} axis data:")
        print(axis_data[:5])  # Print first 5 values for debugging
        
        if len(axis_data) == 0:
            print(f"Warning: No data for {axis} axis")
            features.extend([0] * 9)  # Add zero features for empty axis
            continue
        
        # Perform FFT
        n = len(axis_data)
        dt = 1/20000  # time increment in each data
        fft_data = rfft(axis_data) * dt
        fft_data = np.abs(fft_data[:min(500*5, len(fft_data))])
        
        # Normalize FFT data
        fft_range = np.max(fft_data) - np.min(fft_data)
        if fft_range != 0:
            fft_data = (fft_data - np.min(fft_data)) / fft_range
        else:
            print(f"Warning: Constant data for {axis} axis")
            features.extend([0] * 9)  # Add zero features for constant data
            continue
        
        # Extract features
        mean_val = np.mean(fft_data)
        std_val = np.std(fft_data)
        rms_val = np.sqrt(np.mean(fft_data**2))
        shape_factor = rms_val / np.mean(np.abs(fft_data)) if np.mean(np.abs(fft_data)) != 0 else 0
        impulse_factor = np.max(fft_data) / np.mean(np.abs(fft_data)) if np.mean(np.abs(fft_data)) != 0 else 0
        peak_to_peak = np.max(fft_data) - np.min(fft_data)
        kurtosis_val = kurtosis(fft_data)
        crest_factor = np.max(fft_data) / rms_val if rms_val != 0 else 0
        skewness = skew(fft_data)
        
        features.extend([
            mean_val, std_val, shape_factor, rms_val, impulse_factor,
            peak_to_peak, kurtosis_val, crest_factor, skewness
        ])
    
    print("Extracted features:", features)
    return np.array(features).reshape(1, -1)

def process_data(file_path):
    """
    Process a single file and extract features.
    
    :param file_path: Path to the CSV file
    :return: numpy array of extracted features
    """
    try:
        # Read the CSV file, skipping the first 4 columns and using the 5th, 6th, and 7th columns
        data = pd.read_csv(file_path, usecols=[4, 5, 6], names=['X', 'Y', 'Z'], header=0, skiprows=1)
        return extract_features(data)
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
    print(X)
    print("Shape of X:", X.shape)
    
    # Load the model and predict
    model = joblib.load(model_path)
    predictions = model.predict(X)
    
    # Calculate SHAP values
    explainer = shap.KernelExplainer(model.predict_proba, X)
    shap_values = explainer.shap_values(X)
    
    return X, predictions, shap_values, filenames


# Create a function to ensure a directory exists
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

def plot_shap_values(shap_values, feature_names, anomaly_type, output_folder):
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f'SHAP Values for {anomaly_type}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{anomaly_type}_shap_plot.png'))
    plt.close()

if __name__ == "__main__":
    raw_data_folder = "../../data/unsupervised/output_abnormal"
    model_path = '../../models/supervised/knn_model.joblib'
    output_base_folder = "results"
    
    X, predictions, shap_values, filenames = process_folder_and_predict(raw_data_folder, model_path)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'filename': filenames,
        'predicted_anomaly': predictions
    })

    # Define feature names
    feature_names = [
        f"{feat}_{axis}" for axis in ['X', 'Y', 'Z']
        for feat in ['Mean', 'Std', 'ShapeFactor', 'RMS', 'ImpulseFactor',
                     'PeakToPeak', 'Kurtosis', 'CrestFactor', 'Skewness']
    ]

    # Create DataFrames for SHAP values and extracted features
    shap_df = pd.DataFrame(np.array([shap_values[p][i] for i, p in enumerate(predictions)]), columns=feature_names)
    shap_df['filename'] = filenames
    shap_df['predicted_anomaly'] = predictions

    features_df = pd.DataFrame(X, columns=feature_names)
    features_df['filename'] = filenames
    features_df['predicted_anomaly'] = predictions


    # Group results by predicted anomaly
    grouped_results = results.groupby('predicted_anomaly')
    anomaly_type_map = {
    0: "normal",
    1: "Misalignment",
    2: "Unbalance",
    3: "Looseness",
    4: "Impact"
}

        # Process each anomaly type
   
    for anomaly_type, group in grouped_results:
        # Convert numeric anomaly_type to descriptive name
        anomaly_name = anomaly_type_map.get(anomaly_type, f"unknown_{anomaly_type}")
        
        # Create folders for this anomaly type
        anomaly_folder = os.path.join(output_base_folder, anomaly_name)
        plots_folder = os.path.join(anomaly_folder, 'plots')
        ensure_dir(anomaly_folder)
        ensure_dir(plots_folder)
        
        # Save the anomaly-specific data to CSV files
        group_files = group['filename'].tolist()
        
        # Save anomaly-specific extracted features
        anomaly_features = features_df[features_df['filename'].isin(group_files)]
        anomaly_features.to_csv(os.path.join(anomaly_folder, f'{anomaly_name}_features.csv'), index=False)
        
        # Save anomaly-specific SHAP values
        anomaly_shap = shap_df[shap_df['filename'].isin(group_files)]
        anomaly_shap.to_csv(os.path.join(anomaly_folder, f'{anomaly_name}_shap_values.csv'), index=False)
        
        # Plot and save SHAP values
        anomaly_indices = np.where(predictions == anomaly_type)[0]
        if isinstance(shap_values, list):  # Multi-class case
            anomaly_shap_values = [shap_values[i][anomaly_indices] for i in range(len(shap_values))]
        else:  # Binary classification case
            anomaly_shap_values = shap_values[anomaly_indices]
        
        plot_shap_values(anomaly_shap_values, feature_names, anomaly_name, plots_folder)
        
        # Copy and plot original data files
        for filename in group_files:
            src_path = os.path.join(raw_data_folder, filename)
            dst_path = os.path.join(anomaly_folder, filename)
            shutil.copy2(src_path, dst_path)
            
            # Read and plot the original data
            data = pd.read_csv(src_path, usecols=[4, 5, 6], names=['X', 'Y', 'Z'], header=0, skiprows=1)
            plot_data(data, filename, anomaly_name, plots_folder)

    print("Processing complete. Results saved in the 'results' folder.")