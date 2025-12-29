# dashboard/app.py

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import timm
from PIL import Image

# --------------------------------------------------
# Path setup
# --------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.realtime_inference import RealtimePredictor

# --------------------------------------------------
# ViT encoder (timm)
# --------------------------------------------------
@st.cache_resource
def load_vit():
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=0
    )
    model.eval()
    return model

vit = load_vit()

def encode_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    x = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0)
    with torch.no_grad():
        emb = vit(x).numpy()[0]
    return emb


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    ckpt = Path("models/checkpoints/multimodal_model")
    return RealtimePredictor(ckpt)

model = load_model()

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="PV Multimodal Forecast Dashboard", layout="wide")
st.title("☀️ PV Multimodal Forecast Dashboard")

st.sidebar.header("Controls")
horizon = st.sidebar.slider("Forecast Horizon (hours)", 1, 24, 6)

# --------------------------------------------------
# Simulated real-time inputs
# --------------------------------------------------
def get_latest_inputs():
    power_hist = np.random.rand(12, 1)
    weather = np.random.rand(12, 3)
    temporal = np.concatenate([power_hist, weather], axis=1)

    img_files = sorted(Path("data/images").rglob("*.jpg"))
    img_emb = encode_image(img_files[-1])

    return temporal[None, ...], img_emb[None, ...]

# --------------------------------------------------
# Inference
# --------------------------------------------------
temporal, img_emb = get_latest_inputs()

q10, q50, q90 = model.predict(temporal, img_emb)

forecast = q50.flatten()
lower = q10.flatten()
upper = q90.flatten()

# --------------------------------------------------
# Plot
# --------------------------------------------------
df = pd.DataFrame({
    "P50 Forecast": forecast,
    "Lower (P10)": lower,
    "Upper (P90)": upper
}, index=np.arange(1, horizon + 1))

st.subheader("Forecast with Uncertainty")
st.line_chart(df)

st.caption(
    "Model: Multimodal Transformer (ViT + Weather + PV) | "
    "Uncertainty: Quantile Regression"
)
