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
sys.path.append(str(Path(__file__).resolve().parent.parent))

# -------------------------
# Local module imports
# -------------------------
from inference.realtime_inference import RealtimePredictor

# Optional: Kafka / MQTT
try:
    from kafka import KafkaConsumer
    import json
except ImportError:
    KafkaConsumer = None
try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

# -------------------------
# Helper Functions
# -------------------------
def load_model(latest_checkpoint_dir="models/checkpoints"):
    checkpoints = list(Path(latest_checkpoint_dir).glob("*.pt"))
    if not checkpoints:
        st.error("No model checkpoints found!")
        return None
    latest_ckpt = max(checkpoints, key=lambda x: x.stat().st_mtime)
    return RealtimePredictor(latest_ckpt)

def generate_confidence_band(preds, scale=0.05):
    upper = preds * (1 + scale)
    lower = preds * (1 - scale)
    return lower, upper

def compute_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae

def preprocess_input(raw_data, horizon=24):
    # Replace with actual preprocessing pipeline
    return torch.tensor(raw_data["features"]).reshape(1, horizon, 128)

# -------------------------
# Streaming Input
# -------------------------
def kafka_stream(topic="pv_data", bootstrap_servers="localhost:9092"):
    if KafkaConsumer is None:
        st.error("Kafka dependencies not installed!")
        return
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda x: json.loads(x.decode("utf-8"))
    )
    for message in consumer:
        yield message.value

def mqtt_stream(broker="localhost", port=1883, topic="pv/data"):
    if mqtt is None:
        st.error("paho-mqtt not installed!")
        return
    data_queue = []

    def on_message(client, userdata, msg):
        data = json.loads(msg.payload.decode())
        data_queue.append(data)

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(broker, port)
    client.subscribe(topic)
    client.loop_start()
    
    while True:
        if data_queue:
            yield data_queue.pop(0)
        else:
            time.sleep(0.5)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PV Multimodal Forecast Dashboard", layout="wide")
st.title("PV Multimodal Forecast Dashboard (Real-time)")

# Sidebar controls
st.sidebar.header("Forecast Parameters")
plants = st.sidebar.multiselect(
    "Select PV Plant(s)", ["Plant 1", "Plant 2", "Plant 3"], default=["Plant 1"]
)
horizon = st.sidebar.slider("Forecast Horizon (hours)", 1, 48, 24)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 5)
data_source = st.sidebar.selectbox("Data Source", ["Simulated", "Kafka", "MQTT"])

# Load model
model = load_model()
if not model:
    st.stop()

# Placeholders
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

# -------------------------
# Real-time Stream Handling
# -------------------------
if data_source == "Simulated":
    def simulated_stream():
        while True:
            yield { "features": np.random.rand(horizon, 128) }
    stream = simulated_stream()
elif data_source == "Kafka":
    stream = kafka_stream()
elif data_source == "MQTT":
    stream = mqtt_stream()
else:
    st.error("Invalid data source")
    st.stop()

# -------------------------
# Main dashboard loop
# -------------------------
# Use Streamlit autorefresh
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=refresh_interval*1000, limit=None, key="forecast_refresh")

chart_data = pd.DataFrame()
metrics_text = ""

for plant in plants:
    try:
        raw_data = next(stream)
    except StopIteration:
        raw_data = { "features": np.random.rand(horizon, 128) }

    input_tensor = preprocess_input(raw_data, horizon)
    forecast = model.predict(input_tensor).numpy().flatten()
    
    # Simulate actual PV for demo
    actual = forecast + np.random.normal(0, 0.05, size=forecast.shape)
    
    # Confidence bands
    lower, upper = generate_confidence_band(forecast)
    
    # Metrics
    rmse, mae = compute_metrics(actual, forecast)
    metrics_text += f"**{plant}** → RMSE: {rmse:.3f}, MAE: {mae:.3f}  \n"
    
    # Prepare DataFrame
    df = pd.DataFrame({
        f"{plant} Predicted": forecast,
        f"{plant} Actual": actual,
        f"{plant} Lower": lower,
        f"{plant} Upper": upper
    }, index=np.arange(1, horizon+1))
    
    chart_data = pd.concat([chart_data, df], axis=1) if not chart_data.empty else df

# Display chart and metrics
chart_placeholder.line_chart(chart_data)
metrics_placeholder.markdown(metrics_text)
