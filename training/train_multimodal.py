# =================================================================
# train_multimodal.py
# =================================================================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from utils.data_utils import load_pv_data, load_weather_data, load_and_encode_images
from models.multimodal_transformer import build_model

# ----------------------------
# Config
# ----------------------------
HISTORY_STEPS = 12
FUTURE_STEPS = 6
WEATHER_COLS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]

# ----------------------------
# Load Data
# ----------------------------
pv_df, scaler = load_pv_data("data/processed/merged_multimodal.csv")  # already merged
weather_df = load_weather_data("data/Plant_2_Weather_Sensor_Data.csv", WEATHER_COLS)
img_emb_df = pd.read_csv("data/processed/vit_embeddings.csv", index_col=0)

# Merge modalities
full_df = pd.merge_asof(pv_df.sort_index(), img_emb_df.sort_index(), direction="backward")
full_df = pd.merge_asof(full_df.sort_index(), weather_df.sort_index(), direction="backward")
full_df = full_df.dropna()

# ----------------------------
# Build sequences
# ----------------------------
def build_sequences(df):
    X_p, X_i, X_w, y = [], [], [], []
    power = df["power_scaled"].values
    img_vals = df.iloc[:, 1:1 + 768].values  # ViT embeddings
    weather_vals = df[WEATHER_COLS].values
    for i in range(len(df) - HISTORY_STEPS - FUTURE_STEPS):
        X_p.append(power[i:i + HISTORY_STEPS])
        X_i.append(img_vals[i + HISTORY_STEPS])
        X_w.append(weather_vals[i + HISTORY_STEPS])
        y.append(power[i + HISTORY_STEPS:i + HISTORY_STEPS + FUTURE_STEPS])
    return np.array(X_p)[..., None], np.array(X_i), np.array(X_w), np.array(y)

X_p, X_i, X_w, y = build_sequences(full_df)

# Expand weather to match history
def expand_weather(X_p, X_w):
    w_seq = np.repeat(X_w[:, None, :], HISTORY_STEPS, axis=1)
    return np.concatenate([X_p, w_seq], axis=-1)

X_pw = expand_weather(X_p, X_w)

# Train/test split
split = int(0.8 * len(X_pw))
X_tr, X_te = X_pw[:split], X_pw[split:]
X_i_tr, X_i_te = X_i[:split], X_i[split:]
y_tr, y_te = y[:split], y[split:]

# ----------------------------
# Build model
# ----------------------------
model = build_model(HISTORY_STEPS, X_pw.shape[-1] - HISTORY_STEPS, img_emb_dim=X_i_tr.shape[1])

# ----------------------------
# Multi-quantile loss
# ----------------------------
def multi_quantile_loss(y_true, y_pred):
    quantiles = tf.constant([0.1, 0.5, 0.9], dtype=tf.float32)
    y_true = tf.expand_dims(y_true, axis=1)  # (batch,1,FUTURE)
    e = y_true - y_pred                       # (batch,3,FUTURE)
    q = tf.reshape(quantiles, (1, 3, 1))
    loss = tf.maximum(q * e, (q - 1) * e)
    return tf.reduce_mean(loss)

model.compile(optimizer="adam", loss=multi_quantile_loss)

# ----------------------------
# Train
# ----------------------------
model.fit([X_tr, X_i_tr], y_tr,
          validation_data=([X_te, X_i_te], y_te),
          epochs=30, batch_size=32,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
          verbose=1)

# ----------------------------
# Save model
# ----------------------------
model.save("models/checkpoints/multimodal_model")
print("Model training complete and saved.")
