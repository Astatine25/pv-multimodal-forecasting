# training/train_multimodal.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

from training.quantile_loss import quantile_loss

# -----------------------------
# Config
# -----------------------------
HISTORY_STEPS = 12
FUTURE_STEPS = 6
D_MODEL = 128
BATCH_SIZE = 32
EPOCHS = 30

DATA_PATH = "data/processed/merged_multimodal.csv"
SAVE_PATH = "models/checkpoints/quantile_multimodal"

Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").dropna()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["power_scaled"] = scaler.fit_transform(df[["power"]])
power = df["power_scaled"].values


vit_cols = [c for c in df.columns if c.startswith("vit_")]
weather_cols = [
    c for c in df.columns
    if c not in vit_cols + ["timestamp", "power", "power_scaled"]
]

vit_data = df[vit_cols].values
weather_data = df[weather_cols].values

# -----------------------------
# Build sequences
# -----------------------------
def build_sequences():
    X_seq, X_img, y = [], [], []

    for i in range(len(df) - HISTORY_STEPS - FUTURE_STEPS):
        seq = np.hstack([
            power[i:i+HISTORY_STEPS, None],
            np.repeat(weather_data[i+HISTORY_STEPS][None, :],
                      HISTORY_STEPS, axis=0)
        ])

        X_seq.append(seq)
        X_img.append(vit_data[i+HISTORY_STEPS])
        y.append(power[i+HISTORY_STEPS:i+HISTORY_STEPS+FUTURE_STEPS])

    return np.array(X_seq), np.array(X_img), np.array(y)

X_seq, X_img, y = build_sequences()

split = int(0.8 * len(X_seq))
X_seq_tr, X_seq_te = X_seq[:split], X_seq[split:]
X_img_tr, X_img_te = X_img[:split], X_img[split:]
y_tr, y_te = y[:split], y[split:]

# -----------------------------
# Model
# -----------------------------
def transformer_block(x):
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=D_MODEL)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    ff = layers.Dense(D_MODEL * 2, activation="relu")(x)
    ff = layers.Dense(D_MODEL)(ff)
    x = layers.Add()([x, ff])
    return layers.LayerNormalization()(x)

seq_in = Input(shape=(HISTORY_STEPS, X_seq.shape[-1]))
img_in = Input(shape=(vit_data.shape[1],))

x = layers.Dense(D_MODEL)(seq_in)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)

img_feat = layers.Dense(D_MODEL, activation="relu")(img_in)

fusion = layers.Concatenate()([x, img_feat])
fusion = layers.Dense(256, activation="relu")(fusion)
fusion = layers.Dense(128, activation="relu")(fusion)

out = layers.Dense(FUTURE_STEPS * 3)(fusion)
out = layers.Reshape((3, FUTURE_STEPS))(out)

model = models.Model([seq_in, img_in], out)

model.compile(
    optimizer="adam",
    loss=[
        quantile_loss(0.1),
        quantile_loss(0.5),
        quantile_loss(0.9)
    ]
)

model.summary()

# -----------------------------
# Train
# -----------------------------
model.fit(
    [X_seq_tr, X_img_tr],
    [y_tr, y_tr, y_tr],
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

model.save(SAVE_PATH)
print("Model trained and saved successfully.")
