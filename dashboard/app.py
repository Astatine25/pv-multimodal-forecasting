# -------------------------
# dashboard/app.py
# -------------------------

import sys
from pathlib import Path
import time
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------------------------------
# Add repo root to path
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# -------------------------------------------------
# Config
# -------------------------------------------------
IMAGE_DIR = ROOT / "data/raw/2019_01_images_raw"
WEATHER_CSV = ROOT / "data/raw/Plant_2_Weather_Sensor_Data.csv"
CHECKPOINT = ROOT / "models/checkpoints/best_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZON = 24
IMG_EMB_DIM = 128
WEATHER_DIM = 8

# -------------------------------------------------
# CNN Image Encoder (ResNet18)
# -------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        return self.fc(x)

# -------------------------------------------------
# Quantile Forecast Model
# -------------------------------------------------
class QuantileForecastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_EMB_DIM + WEATHER_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, HORIZON * 3)  # P10, P50, P90
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, HORIZON, 3)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = QuantileForecastModel().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

# -------------------------------------------------
# Load CNN Encoder
# -------------------------------------------------
@st.cache_resource
def load_encoder():
    encoder = CNNEncoder(out_dim=IMG_EMB_DIM).to(DEVICE)
    encoder.eval()
    return encoder

# -------------------------------------------------
# Image Utilities
# -------------------------------------------------
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_latest_image():
    images = sorted(IMAGE_DIR.glob("*.jpg"))
    if not images:
        st.error("No images found")
        st.stop()
    img = Image.open(images[-1]).convert("RGB")
    return IMG_TRANSFORM(img).unsqueeze(0).to(DEVICE)

# -------------------------------------------------
# Weather Loader
# -------------------------------------------------
@st.cache_data
def load_weather():
    df = pd.read_csv(WEATHER_CSV)
    return df.iloc[-1][[
        "IRRADIANCE", "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE", "WIND_SPEED",
        "HUMIDITY", "PRESSURE",
        "DEW_POINT", "CLOUD_COVER"
    ]].values.astype(np.float32)

# -------------------------------------------------
# Metrics
# -------------------------------------------------
def metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred)
    )

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("🌤️ Multimodal PV Power Forecasting (Real-Time)")

model = load_model()
encoder = load_encoder()

# Sidebar
st.sidebar.header("Settings")
refresh = st.sidebar.slider("Refresh (seconds)", 2, 30, 5)

# -------------------------------------------------
# Inference
# -------------------------------------------------
img = load_latest_image()
weather = torch.tensor(load_weather(), device=DEVICE).unsqueeze(0)

with torch.no_grad():
    img_emb = encoder(img)
    fused = torch.cat([img_emb, weather], dim=1)
    preds = model(fused).cpu().numpy()[0]

p10, p50, p90 = preds[:, 0], preds[:, 1], preds[:, 2]

# Fake actual for demo (replace with real PV sensor)
actual = p50 + np.random.normal(0, 0.05, size=p50.shape)

rmse, mae = metrics(actual, p50)

# -------------------------------------------------
# Plot
# -------------------------------------------------
df = pd.DataFrame({
    "Actual": actual,
    "P10": p10,
    "P50 (Median)": p50,
    "P90": p90
})

st.line_chart(df)
st.markdown(f"**RMSE:** {rmse:.3f} | **MAE:** {mae:.3f}")

st.caption("Quantile regression uncertainty bands (P10–P90)")

time.sleep(refresh)
