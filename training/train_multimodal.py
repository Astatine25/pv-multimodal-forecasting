# ==============================================================================
# training/train_multimodal.py
# ==============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, Input

from utils.data_utils import (
    load_pv_data,
    load_weather_data,
    load_and_encode_images
)
from models.multimodal_transformer import build_model

# ==============================================================================
# CONFIG
# ==============================================================================
HISTORY_STEPS = 12
FUTURE_STEPS = 6
IMG_EMB_DIM = 64

WEATHER_COLS = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION"
]

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# IMAGE ENCODER (CNN)
# ==============================================================================
def build_image_encoder():
    inp = Input(shape=(64, 64, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(IMG_EMB_DIM, activation="relu")(x)
    return models.Model(inp, out)

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("📥 Loading PV data...")
pv_df, scaler = load_pv_data(DATA_DIR / "2019_pv_raw.csv")

print("📥 Loading weather data...")
weather_df = load_weather_data(
    DATA_DIR / "Plant_2_Weather_Sensor_Data.csv",
    WEATHER_COLS
)

print("📥 Encoding sky images...")
image_encoder = build_image_encoder()
img_emb_df = load_and_encode_images(IMAGE_DIR, image_encoder)

# ==============================================================================
# MERGE MODALITIES (CRITICAL FIX)
# ==============================================================================
print("🔗 Aligning timestamps...")

# PV + Images
full_df = pd.merge_asof(
    pv_df.sort_index(),
    img_emb_df.sort_index(),
    left_index=True,
    right_index=True,
    direction="backward",
    tolerance=pd.Timedelta("2min")
)

# + Weather
full_df = pd.merge_asof(
    full_df.sort_index(),
    weather_df.sort_index(),
    left_index=True,
    right_index=True,
    direction="nearest",
    tolerance=pd.Timedelta("10min")
)

full_df = full_df.ffill().bfill().dropna()

print(f"✅ Final merged rows: {len(full_df)}")

if len(full_df) < 1000:
    raise RuntimeError("❌ Dataset too small after alignment")

# ==============================================================================
# SEQUENCE BUILDER
# ==============================================================================
def build_sequences(df):
    X_p, X_i, X_w, y = [], [], [], []

    power = df["power_scaled"].values
    img_vals = df.iloc[:, 1:1 + IMG_EMB_DIM].values
    weather_vals = df[WEATHER_COLS].values

    for i in range(len(df) - HISTORY_STEPS - FUTURE_STEPS):
        X_p.append(power[i:i + HISTORY_STEPS])
        X_i.append(img_vals[i + HISTORY_STEPS])
        X_w.append(weather_vals[i + HISTORY_STEPS])
        y.append(power[i + HISTORY_STEPS:i + HISTORY_STEPS + FUTURE_STEPS])

    return (
        np.array(X_p)[..., None],
        np.array(X_i),
        np.array(X_w),
        np.array(y)
    )

X_p, X_i, X_w, y = build_sequences(full_df)

# ==============================================================================
# CONCAT WEATHER WITH POWER SEQUENCE
# ==============================================================================
def expand_weather(X_p, X_w):
    w_seq = np.repeat(X_w[:, None, :], HISTORY_STEPS, axis=1)
    return np.concatenate([X_p, w_seq], axis=-1)

X_pw = expand_weather(X_p, X_w)

# ==============================================================================
# TRAIN / TEST SPLIT
# ==============================================================================
split = int(0.8 * len(X_pw))

X_tr, X_te = X_pw[:split], X_pw[split:]
X_i_tr, X_i_te = X_i[:split], X_i[split:]
y_tr, y_te = y[:split], y[split:]

# ==============================================================================
# BUILD & TRAIN MODEL
# ==============================================================================
print("🧠 Building multimodal transformer...")
model = build_model(
    history_steps=HISTORY_STEPS,
    weather_dim=len(WEATHER_COLS),
    img_emb_dim=IMG_EMB_DIM,
    future_steps=FUTURE_STEPS
)

model.summary()

print("🚀 Training started...")
model.fit(
    [X_tr, X_i_tr],
    y_tr,
    epochs=30,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True)
    ],
    verbose=1
)

# ==============================================================================
# SAVE MODEL
# ==============================================================================
model_path = CHECKPOINT_DIR / "multimodal_model.keras"
model.save(model_path)

print(f"✅ Model training complete")
print(f"📦 Saved to: {model_path}")
