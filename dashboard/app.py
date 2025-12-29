# -------------------------
# dashboard/app.py
# -------------------------

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# Fix Python path (repo root)
# -------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# -------------------------
# Local imports
# -------------------------
from inference.realtime_inference import RealtimePredictor

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(
    page_title="PV Multimodal Forecast Dashboard",
    layout="wide",
)

st.title("☀️ PV Multimodal Forecast Dashboard")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Forecast Settings")

plant = st.sidebar.selectbox(
    "Select Plant",
    ["Plant 1", "Plant 2", "Plant 3"]
)

horizon = st.sidebar.slider(
    "Forecast Horizon (hours)",
    min_value=1,
    max_value=48,
    value=24
)

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    ckpt_path = ROOT / "models" / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        st.error("❌ No trained model found. Train the model first.")
        st.stop()
    return RealtimePredictor(ckpt_path)

model = load_model()

# -------------------------
# Generate simulated inputs
# (replace later with real weather + images)
# -------------------------
def generate_inputs():
    weather_feats = torch.rand(1, 128, dtype=torch.float32)
    image_feats = torch.rand(1, 128, dtype=torch.float32)
    return weather_feats, image_feats

# -------------------------
# Prediction
# -------------------------
weather_feats, image_feats = generate_inputs()

with torch.no_grad():
    preds = model.predict(weather_feats, image_feats).numpy()

# preds shape: (horizon, 3)
p10 = preds[:horizon, 0]
p50 = preds[:horizon, 1]
p90 = preds[:horizon, 2]

# Simulated actual PV (for demo only)
actual = p50 + np.random.normal(0, 0.03, size=horizon)

# -------------------------
# Metrics
# -------------------------
rmse = np.sqrt(mean_squared_error(actual, p50))
mae = mean_absolute_error(actual, p50)

# -------------------------
# Plot data
# -------------------------
df = pd.DataFrame({
    "Hour": np.arange(1, horizon + 1),
    "Actual": actual,
    "P10": p10,
    "P50 (Forecast)": p50,
    "P90": p90,
})

df.set_index("Hour", inplace=True)

# -------------------------
# Display chart
# -------------------------
st.subheader("📈 Forecast vs Actual")

st.line_chart(df)

# -------------------------
# Metrics display
# -------------------------
st.subheader("📊 Metrics")

col1, col2 = st.columns(2)
col1.metric("RMSE", f"{rmse:.3f}")
col2.metric("MAE", f"{mae:.3f}")

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    **Model**: Multimodal PV Forecast (Weather + Images)  
    **Uncertainty**: Quantile Regression (P10 / P50 / P90)  
    **Inference**: Real-time, deployment-ready
    """
)
