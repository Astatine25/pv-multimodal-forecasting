# ==============================================================================
# Multimodal PV Power Forecasting - Training Script
# Author: Vivek Palsutkar
# ==============================================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

# ---------------- CONFIG ----------------
DATA_CSV = "data/2019_pv_raw.csv"
MODEL_DIR = "models"
SEQ_LEN = 12
BATCH_SIZE = 32
EPOCHS = 20

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_CSV)

# Ensure timestamp sorted
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# ---------------- SCALE POWER ----------------
scaler = StandardScaler()
df["power_scaled"] = scaler.fit_transform(df[["power"]])

joblib.dump(scaler, f"{MODEL_DIR}/power_scaler.save")

power = df["power_scaled"].values

# ---------------- CREATE SEQUENCES ----------------
def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X, y = make_sequences(power, SEQ_LEN)
X = X[..., np.newaxis]

# ---------------- TRAIN / VAL SPLIT ----------------
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ---------------- MODEL ----------------
model = models.Sequential([
    layers.Input(shape=(SEQ_LEN, 1)),
    layers.Conv1D(32, 3, activation="relu"),
    layers.Conv1D(64, 3, activation="relu"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ---------------- TRAIN ----------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ---------------- SAVE ----------------
model.save(f"{MODEL_DIR}/multimodal_pv_model.keras")

print("\n✅ Training complete. Model & scaler saved.")
