import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Constants
TIME_STEPS = 288
NORMAL_DATA_PATH = '../../data/unsupervised/train_data'
MODEL_PATH = '../../models/unsupervised/improved_model.keras'
PARAMS_PATH = '../../models/unsupervised/improved_saved_params.joblib'

def load_csv_files(directory):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename), parse_dates=True, index_col="Time(sec)")
            df = df.iloc[:, 4:]  # Assuming we want to skip the first 4 columns
            dataframes.append(df)
    return pd.concat(dataframes)

def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=64, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1DTranspose(filters=64, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.Conv1DTranspose(filters=3, kernel_size=7, padding="same"),
    ])
    return model

def train_model(x_train):
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    x_train_seq = create_sequences(x_train_scaled)
    print("Training input shape: ", x_train_seq.shape)

    model = build_model((x_train_seq.shape[1], x_train_seq.shape[2]))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    history = model.fit(
        x_train_seq, x_train_seq,
        epochs=2,
        batch_size=64,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001)
        ],
    )

    # Get train MAE loss and threshold
    x_train_pred = model.predict(x_train_seq)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train_seq), axis=(1,2))
    threshold = np.percentile(train_mae_loss, 95)  # Use 95th percentile as threshold
    print("Reconstruction error threshold: ", threshold)

    return model, scaler, threshold, history

def main():
    # Load normal data for training
    df_normal = load_csv_files(NORMAL_DATA_PATH)

    # Train the model
    model, scaler, threshold, history = train_model(df_normal)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    # Save parameters
    params = {
        'scaler': scaler,
        'threshold': threshold
    }
    joblib.dump(params, PARAMS_PATH)

    print("Improved model and parameters saved successfully.")

if __name__ == "__main__":
    main()