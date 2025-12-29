# ==============================================================================
# training/train_multimodal.py
# Multimodal PV Forecasting with Quantile Regression
# ==============================================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_PATH = "data/processed/merged_multimodal.csv"
MODEL_OUT = "models/checkpoints/multimodal_model"

HISTORY_STEPS = 12
FUTURE_STEPS = 6
N_QUANTILES = 3
IMG_EMB_DIM = 768

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# ==============================================================================
# SAFE QUANTILE LOSS (SINGLE OUTPUT)
# y_true: (batch, FUTURE_STEPS)
# y_pred: (batch, 3, FUTURE_STEPS)
# ==============================================================================
@tf.function
def quantile_loss_multi(y_true, y_pred):
    qs = tf.constant([0.1, 0.5, 0.9], dtype=tf.float32)

    # (B, 1, T)
    y_true = tf.expand_dims(y_true, axis=1)

    error = y_true - y_pred

    loss = tf.maximum(
        qs[:, None] * error,
        (qs[:, None] - 1.0) * error
    )

    return tf.reduce_mean(loss)

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("Loading merged multimodal dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

power_col = "power"
weather_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
img_cols = [c for c in df.columns if c.startswith("vit_")]

assert len(img_cols) == IMG_EMB_DIM, "ViT embedding dimension mismatch"

# Normalize power
power = df[power_col].astype("float32").values
power = (power - power.mean()) / (power.std() + 1e-6)

weather = df[weather_cols].astype("float32").values
images = df[img_cols].astype("float32").values

# ==============================================================================
# BUILD SEQUENCES
# ==============================================================================
def build_sequences(power, weather, images):
    X_pw, X_img, y = [], [], []

    for i in range(len(power) - HISTORY_STEPS - FUTURE_STEPS):
        pw_seq = np.concatenate(
            [
                power[i:i + HISTORY_STEPS, None],
                weather[i:i + HISTORY_STEPS],
            ],
            axis=1,
        )

        X_pw.append(pw_seq)
        X_img.append(images[i + HISTORY_STEPS])
        y.append(power[i + HISTORY_STEPS : i + HISTORY_STEPS + FUTURE_STEPS])

    return np.array(X_pw), np.array(X_img), np.array(y)

X_pw, X_img, y = build_sequences(power, weather, images)

# ==============================================================================
# TRAIN / TEST SPLIT
# ==============================================================================
split = int(0.8 * len(X_pw))

X_pw_tr, X_pw_te = X_pw[:split], X_pw[split:]
X_img_tr, X_img_te = X_img[:split], X_img[split:]
y_tr, y_te = y[:split], y[split:]

# ==============================================================================
# MODEL
# ==============================================================================
def build_model():
    # Power + weather branch
    pw_in = Input(shape=(HISTORY_STEPS, 1 + len(weather_cols)))
    x = layers.Dense(128)(pw_in)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Image embedding branch
    img_in = Input(shape=(IMG_EMB_DIM,))
    img_x = layers.Dense(128, activation="relu")(img_in)

    # Fusion
    fusion = layers.Concatenate()([x, img_x])
    fusion = layers.Dense(256, activation="relu")(fusion)
    fusion = layers.Dense(128, activation="relu")(fusion)

    # Quantile output
    out = layers.Dense(FUTURE_STEPS * N_QUANTILES)(fusion)
    out = layers.Reshape((N_QUANTILES, FUTURE_STEPS))(out)

    model = models.Model(inputs=[pw_in, img_in], outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=quantile_loss_multi,
    )

    return model

model = build_model()
model.summary()

# ==============================================================================
# TRAIN
# ==============================================================================
model.fit(
    [X_pw_tr, X_img_tr],
    y_tr,
    validation_data=([X_pw_te, X_img_te], y_te),
    epochs=30,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
)

# ==============================================================================
# SAVE
# ==============================================================================
model.save(MODEL_OUT)
print("Model training complete and saved")
