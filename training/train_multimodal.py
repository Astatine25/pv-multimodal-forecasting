# ==============================================================================
# Multimodal PV Forecast Training Script
# ==============================================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping

# ==============================================================================
# CONFIG
# ==============================================================================

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"          # extracted 2019_01_images_raw
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 64
IMG_EMB_DIM = 64

HISTORY_STEPS = 12
FUTURE_STEPS = 6

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4

WEATHER_COLS = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION"
]

# ==============================================================================
# 1. LOAD PV DATA
# ==============================================================================

print("[INFO] Loading PV data...")

pv_df = pd.read_csv(DATA_DIR / "2019_pv_raw.csv")
pv_df.columns = ["timestamp", "power"]
pv_df["timestamp"] = pd.to_datetime(pv_df["timestamp"])
pv_df = pv_df.set_index("timestamp").sort_index().dropna()

scaler = StandardScaler()
pv_df["power_scaled"] = scaler.fit_transform(pv_df[["power"]])

# ==============================================================================
# 2. LOAD WEATHER DATA
# ==============================================================================

print("[INFO] Loading weather data...")

weather_df = pd.read_csv(DATA_DIR / "Plant_2_Weather_Sensor_Data.csv")
weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])
weather_df = weather_df.set_index("DATE_TIME")[WEATHER_COLS]

weather_df = (
    weather_df
    .resample("1min")
    .mean()
    .interpolate()
    .ffill()
    .bfill()
)

# ==============================================================================
# 3. IMAGE ENCODER (CNN – later replaceable with ViT)
# ==============================================================================

def build_image_encoder():
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(IMG_EMB_DIM, activation="relu")(x)
    return models.Model(inp, out)

image_encoder = build_image_encoder()

# ==============================================================================
# 4. LOAD & ENCODE IMAGES
# ==============================================================================

print("[INFO] Encoding sky images...")

def load_and_encode_images(img_root):
    images, times = [], []

    for img_path in sorted(img_root.rglob("*.jpg")):
        try:
            # filename: 20190131185630.jpg
            ts = pd.to_datetime(img_path.stem, format="%Y%m%d%H%M%S")
        except Exception:
            continue

        img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
        images.append(np.asarray(img) / 255.0)
        times.append(ts)

    images = np.array(images, dtype=np.float32)
    emb = image_encoder.predict(images, batch_size=32, verbose=1)

    return pd.DataFrame(emb, index=pd.to_datetime(times)).sort_index()

img_emb_df = load_and_encode_images(IMAGE_DIR)

# ==============================================================================
# 5. MERGE MODALITIES (TIMESTAMP-SAFE)
# ==============================================================================

print("[INFO] Merging PV + Images + Weather...")

full_df = pd.merge_asof(
    pv_df.sort_index(),
    img_emb_df.sort_index(),
    left_index=True,
    right_index=True,
    tolerance=pd.Timedelta("2min"),
    direction="nearest"
)

full_df = pd.merge_asof(
    full_df.sort_index(),
    weather_df.sort_index(),
    left_index=True,
    right_index=True,
    tolerance=pd.Timedelta("5min"),
    direction="nearest"
)

full_df = full_df.ffill().bfill().dropna()

assert len(full_df) > 500, "Dataset too small after merge"

# ==============================================================================
# 6. SEQUENCE BUILDING
# ==============================================================================

def build_sequences(df):
    X_p, X_i, X_w, y = [], [], [], []

    power = df["power_scaled"].values
    img_vals = df.iloc[:, 1:1+IMG_EMB_DIM].values
    weather_vals = df[WEATHER_COLS].values

    for i in range(len(df) - HISTORY_STEPS - FUTURE_STEPS):
        X_p.append(power[i:i+HISTORY_STEPS])
        X_i.append(img_vals[i+HISTORY_STEPS])
        X_w.append(weather_vals[i+HISTORY_STEPS])
        y.append(power[i+HISTORY_STEPS:i+HISTORY_STEPS+FUTURE_STEPS])

    return (
        np.array(X_p)[..., None],
        np.array(X_i),
        np.array(X_w),
        np.array(y)
    )

X_p, X_i, X_w, y = build_sequences(full_df)

def expand_weather(X_p, X_w):
    w_seq = np.repeat(X_w[:, None, :], HISTORY_STEPS, axis=1)
    return np.concatenate([X_p, w_seq], axis=-1)

X_pw = expand_weather(X_p, X_w)

# ==============================================================================
# 7. MODEL (Transformer-based Fusion)
# ==============================================================================

def transformer_block(x, d_model=64, num_heads=4, ff_dim=128):
    attn = layers.MultiHeadAttention(
        key_dim=d_model // num_heads,
        num_heads=num_heads
    )(x, x)

    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)

    x = layers.Add()([x, ff])
    return layers.LayerNormalization()(x)

def build_model():
    seq_in = Input(shape=(HISTORY_STEPS, 1 + len(WEATHER_COLS)))
    x = layers.Dense(64)(seq_in)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)

    img_in = Input(shape=(IMG_EMB_DIM,))
    img_feat = layers.Dense(64, activation="relu")(img_in)

    fusion = layers.Concatenate()([x, img_feat])
    fusion = layers.Dense(128, activation="relu")(fusion)
    fusion = layers.Dense(64, activation="relu")(fusion)

    out = layers.Dense(FUTURE_STEPS)(fusion)

    model = models.Model([seq_in, img_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="mse"
    )
    return model

model = build_model()
model.summary()

# ==============================================================================
# 8. TRAIN
# ==============================================================================

print("[INFO] Training model...")

split = int(0.8 * len(X_pw))
model.fit(
    [X_pw[:split], X_i[:split]],
    y[:split],
    validation_data=([X_pw[split:], X_i[split:]], y[split:]),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

# ==============================================================================
# 9. SAVE
# ==============================================================================

model.save(CHECKPOINT_DIR / "multimodal_model.keras")
print("[OK] Model training complete and saved.")
