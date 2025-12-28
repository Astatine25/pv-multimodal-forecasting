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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# Add repo root to Python path
# -------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# -------------------------
# Local imports
# -------------------------
from inference.realtime_inference import RealtimePredictor

# -------------------------
# Helper Functions
# -------------------------
def load_model(checkpoint_dir="models/checkpoints"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.pt"))

    if not checkpoints:
        st.error("❌ No model checkpoints found. Train the model first.")
        return None

    latest_ckpt = max(checkpoints, key=lambda x: x.stat().st_mtime)
    st.sidebar.success(f"Loaded model: {latest_ckpt.name}")
    return RealtimePredictor(latest_ckpt)


def preprocess_input(raw_data, horizon):
    """
    Converts input features to float32 tensor
    Shape: (1, horizon, 128)
    """
    return torch.tensor(
        raw_data["features"],
        dtype=torch.float32
    ).reshape(1, horizon, 128)


def generate_confidence_band(preds, scale=0.05):
    lower = preds * (1 - scale)
    upper = preds * (1 + scale)
    return lower, upper


def compute_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="PV Multimodal Forecast Dashboard",
    layout="wide"
)

st.title("🔆 PV Multimodal Forecast Dashboard")

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Forecast Controls")

plants = st.sidebar.multiselect(
    "Select PV Plant(s)",
    ["Plant 1", "Plant 2", "Plant 3"],
    default=["Plant 1"]
)

horizon = st.sidebar.slider(
    "Forecast Horizon (hours)",
    min_value=1,
    max_value=48,
    value=24
)

refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)",
    min_value=1,
    max_value=10,
    value=5
)

data_source = st.sidebar.selectbox(
    "Data Source",
    ["Simulated"]
)

# -------------------------
# Load Model
# -------------------------
model = load_model()
if model is None:
    st.stop()

# -------------------------
# Auto Refresh
# -------------------------
from streamlit_autorefresh import st_autorefresh
st_autorefresh(
    interval=refresh_interval * 1000,
    key="pv_refresh"
)

# -------------------------
# Data Generation (Simulated)
# -------------------------
def get_simulated_data():
    return {
        "features": np.random.rand(horizon, 128)
    }

# -------------------------
# Run Forecast
# -------------------------
chart_data = pd.DataFrame()
metrics_text = ""

for plant in plants:
    raw_data = get_simulated_data()

    input_tensor = preprocess_input(raw_data, horizon)

    forecast = model.predict(input_tensor).cpu().numpy().flatten()

    # Simulated actual PV (demo)
    actual = forecast + np.random.normal(0, 0.05, size=forecast.shape)

    lower, upper = generate_confidence_band(forecast)

    rmse, mae = compute_metrics(actual, forecast)

    metrics_text += (
        f"**{plant}** → RMSE: `{rmse:.3f}` | MAE: `{mae:.3f}`  \n"
    )

    df = pd.DataFrame(
        {
            f"{plant} Predicted": forecast,
            f"{plant} Actual": actual,
            f"{plant} Lower": lower,
            f"{plant} Upper": upper,
        },
        index=np.arange(1, horizon + 1)
    )

    chart_data = pd.concat([chart_data, df], axis=1)

# -------------------------
# Display
# -------------------------
st.subheader("📈 Forecast vs Actual")
st.line_chart(chart_data)

st.subheader("📊 Metrics")
st.markdown(metrics_text)

st.caption(
    "Model: Multimodal PV Forecast | "
    "Inference: Real-time | "
    "Deployment-ready"
)
