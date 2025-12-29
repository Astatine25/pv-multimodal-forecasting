import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
from inference.realtime_inference import RealtimePredictor

# Load model
model = RealtimePredictor("models/checkpoints/multimodal_model")

st.title("PV Multimodal Forecast Dashboard (Real-time)")

horizon = st.sidebar.slider("Forecast Horizon", 1, 48, 24)
refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 10, 5)

chart_placeholder = st.empty()
metrics_placeholder = st.empty()

# Simulated loop for demo
while True:
    # Dummy inputs
    X_seq = np.random.rand(1, 12, 1+3)
    X_img = np.random.rand(1,64)

    forecast = model.predict(X_seq, X_img).flatten()
    actual = forecast + np.random.normal(0,0.05, size=forecast.shape)

    df = pd.DataFrame({"Predicted": forecast, "Actual": actual}, index=np.arange(1, len(forecast)+1))
    chart_placeholder.line_chart(df)
    metrics_placeholder.text(f"RMSE: {np.sqrt(np.mean((forecast-actual)**2)):.3f}")

    time.sleep(refresh_interval)
