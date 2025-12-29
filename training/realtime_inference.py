# =================================================================
# realtime_inference.py
# =================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.data_utils import preprocess_weather, preprocess_pv, load_and_encode_images

# ----------------------------
# Config
# ----------------------------
HISTORY_STEPS = 12
FUTURE_STEPS = 6
WEATHER_COLS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]

# ----------------------------
# Load trained model
# ----------------------------
model = load_model("models/checkpoints/multimodal_model", compile=False)

# ----------------------------
# Load latest data
# ----------------------------
pv_latest = preprocess_pv("data/2019_pv_raw.csv")      # last HISTORY_STEPS
weather_latest = preprocess_weather("data/Plant_2_Weather_Sensor_Data.csv", WEATHER_COLS)
img_latest = load_and_encode_images("data/images/latest", image_encoder=None)  # ViT embeddings

# ----------------------------
# Build input
# ----------------------------
X_p = pv_latest[-HISTORY_STEPS:, None]  # shape (HISTORY,1)
X_w = np.repeat(weather_latest[-1][None, :], HISTORY_STEPS, axis=0)
X_pw = np.concatenate([X_p, X_w], axis=-1)[None, ...]  # batch dim

X_i = img_latest[-1][None, :]  # batch dim

# ----------------------------
# Predict quantiles
# ----------------------------
preds = model.predict([X_pw, X_i])  # shape: (1,3,FUTURE_STEPS)
preds = preds[0]                     # remove batch

# Extract P10, P50, P90
P10, P50, P90 = preds[0], preds[1], preds[2]

print("Predicted Power Quantiles for next {} steps:".format(FUTURE_STEPS))
for t in range(FUTURE_STEPS):
    print(f"Step {t+1}: P10={P10[t]:.3f}, P50={P50[t]:.3f}, P90={P90[t]:.3f}")
