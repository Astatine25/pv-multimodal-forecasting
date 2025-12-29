# ==============================================================================
# realtime_inference.py
# Quantile-aware real-time inference for multimodal PV forecasting
# ==============================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_PATH = "models/checkpoints/multimodal_model"
DATA_PATH = "data/processed/merged_multimodal.csv"

HISTORY_STEPS = 12
FUTURE_STEPS = 6
N_QUANTILES = 3
IMG_EMB_DIM = 768

POWER_COL = "power"
WEATHER_COLS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]

# ==============================================================================
# LOAD MODEL (custom loss needed)
# ==============================================================================
@tf.function
def quantile_loss_multi(y_true, y_pred):
    qs = tf.constant([0.1, 0.5, 0.9], dtype=tf.float32)
    y_true = tf.expand_dims(y_true, axis=1)
    error = y_true - y_pred
    loss = tf.maximum(
        qs[:, None] * error,
        (qs[:, None] - 1.0) * error
    )
    return tf.reduce_mean(loss)

print("Loading trained model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"quantile_loss_multi": quantile_loss_multi},
)
print("Model loaded successfully")

# ==============================================================================
# LOAD DATA (for rolling window inference demo)
# ==============================================================================
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

img_cols = [c for c in df.columns if c.startswith("vit_")]
assert len(img_cols) == IMG_EMB_DIM, "ViT embedding dimension mismatch"

# ==============================================================================
# NORMALIZATION (must match training)
# ==============================================================================
power_mean = df[POWER_COL].mean()
power_std = df[POWER_COL].std() + 1e-6

def normalize_power(x):
    return (x - power_mean) / power_std

def denormalize_power(x):
    return x * power_std + power_mean

# ==============================================================================
# BUILD SINGLE INFERENCE SAMPLE
# ==============================================================================
def build_latest_sample(df):
    """
    Returns:
        X_pw : (1, HISTORY_STEPS, 1 + weather_dim)
        X_img: (1, IMG_EMB_DIM)
    """
    recent = df.iloc[-(HISTORY_STEPS + 1):]

    power = normalize_power(recent[POWER_COL].values[:-1])
    weather = recent[WEATHER_COLS].values[:-1]

    pw_seq = np.concatenate(
        [power[:, None], weather],
        axis=1
    )

    img_emb = recent[img_cols].values[-1]

    return (
        pw_seq[None, ...].astype("float32"),
        img_emb[None, ...].astype("float32"),
    )

# ==============================================================================
# REAL-TIME FORECAST FUNCTION
# ==============================================================================
def forecast_quantiles(df):
    X_pw, X_img = build_latest_sample(df)

    # Model output: (1, 3, FUTURE_STEPS)
    preds = model.predict([X_pw, X_img], verbose=0)[0]

    p10 = denormalize_power(preds[0])
    p50 = denormalize_power(preds[1])
    p90 = denormalize_power(preds[2])

    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }

# ==============================================================================
# DEMO RUN
# ==============================================================================
if __name__ == "__main__":
    forecast = forecast_quantiles(df)

    print("\n Quantile Forecast (next {} steps):".format(FUTURE_STEPS))
    for t in range(FUTURE_STEPS):
        print(
            f"T+{t+1:02d} | "
            f"P10: {forecast['p10'][t]:.2f} | "
            f"P50: {forecast['p50'][t]:.2f} | "
            f"P90: {forecast['p90'][t]:.2f}"
        )
