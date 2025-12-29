# app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/checkpoints/multimodal_model"
DATA_PATH = "data/processed/merged_multimodal.csv"

HISTORY_STEPS = 12
FUTURE_STEPS = 6
WEATHER_COLS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
IMG_EMB_DIM = 768

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

df = load_data()

img_cols = [c for c in df.columns if c.startswith("vit_")]

# Normalize power (same as training)
power_mean = df["power"].mean()
power_std = df["power"].std() + 1e-6

# =========================
# STREAMLIT UI
# =========================
st.title("PV Power Forecast (Quantile-Aware)")
st.markdown("**P10 / P50 / P90 probabilistic forecasting using ViT + Transformer**")

idx = st.slider(
    "Select starting timestep",
    HISTORY_STEPS,
    len(df) - FUTURE_STEPS - 1,
    len(df) - FUTURE_STEPS - 1,
)

# =========================
# BUILD INPUT
# =========================
power_seq = df["power"].values[idx-HISTORY_STEPS:idx]
power_seq = (power_seq - power_mean) / power_std

weather_seq = df[WEATHER_COLS].values[idx-HISTORY_STEPS:idx]
pw_seq = np.concatenate(
    [power_seq[:, None], weather_seq],
    axis=1
)[None, :, :]  # (1, 12, 4)

img_emb = df[img_cols].iloc[idx].values[None, :]  # (1, 768)

# =========================
# PREDICT
# =========================
pred = model.predict([pw_seq, img_emb], verbose=0)

# Output: (1, 3, 6)
p10, p50, p90 = pred[0]

# De-normalize
p10 = p10 * power_std + power_mean
p50 = p50 * power_std + power_mean
p90 = p90 * power_std + power_mean

# =========================
# PLOT
# =========================
t = np.arange(1, FUTURE_STEPS + 1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, p50, label="P50 (Median)", linewidth=2)
ax.fill_between(t, p10, p90, alpha=0.3, label="P10–P90 Uncertainty")
ax.set_xlabel("Forecast Horizon")
ax.set_ylabel("PV Power")
ax.set_title("6-Step Ahead Forecast")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# =========================
# TABLE
# =========================
st.subheader("Forecast Values")
out_df = pd.DataFrame({
    "Horizon": t,
    "P10": p10,
    "P50": p50,
    "P90": p90
})

st.dataframe(out_df, use_container_width=True)
