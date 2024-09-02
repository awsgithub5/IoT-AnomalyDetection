import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import os
import joblib
import matplotlib.pyplot as plt
 
# Constants
TIME_STEPS = 288

NORMAL_DATA_PATH = '../../data/unsupervised/train_data'
MODEL_PATH = '../../models/unsupervised/model.keras'
PARAMS_PATH = '../../models/unsupervised/saved_params.joblib'
 
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
        layers.Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.Dropout(rate=0.2),
        layers.Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ])
    return model
 
def train_model(x_train):
    training_mean = x_train.mean()
    training_std = x_train.std()
    df_training_value = (x_train - training_mean) / training_std
 
    x_train = create_sequences(df_training_value.values)
    print("Training input shape: ", x_train.shape)
 
    model = build_model((x_train.shape[1], x_train.shape[2]))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")
 
    history = model.fit(
        x_train, x_train,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )
 
    # Get train MAE loss and threshold
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
    threshold = np.max(train_mae_loss)
    print("Reconstruction error threshold: ", threshold)
 
    return model, training_mean, training_std, threshold
 
def main():
    # Load normal data for training
    df_normal = load_csv_files(NORMAL_DATA_PATH)
 
    # Train the model
    model, training_mean, training_std, threshold = train_model(df_normal)
 
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
 
    # Save parameters
    params = {
        'training_mean': training_mean,
        'training_std': training_std,
        'threshold': threshold
    }
    joblib.dump(params, PARAMS_PATH)
 
    print("Model and parameters saved successfully.")
 
if __name__ == "__main__":
    main()