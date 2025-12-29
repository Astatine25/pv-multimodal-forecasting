# dashboard/app.py

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.realtime_inference import RealtimePredictor

# Optional streaming
try:
    from kafka import KafkaConsumer
except:
    KafkaConsumer = None

try:
    import paho.mqtt.client as mqtt
except:
    mqtt = None

# ---------------------------
# CONFIG
# ---------------------------
HISTORY_STEPS = 12
IMG_EMB_DIM = 64
WEATHER_DIM = 3

# ---------------------------
# UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("Multimodal PV Forecast – Real Time")

source = st.sidebar.selectbox(
    "Data Source",
    ["Simulated", "Kafka", "MQTT"]
)

refresh = st.sidebar.slider("Refresh (seconds)", 1, 10, 5)

# ---------------------------
# Load model
# ---------------------------
model = RealtimePredictor()

# ---------------------------
# Fake real-time generator
# ---------------------------
def simulated_stream():
    while True:
        yield {
            "power_seq": np.random.rand(HISTORY_STEPS),
            "weather": np.random.rand(WEATHER_DIM),
            "image_emb": np.random.rand(IMG_EMB_DIM)
        }

# ---------------------------
# Kafka
# ---------------------------
def kafka_stream(topic="pv_stream", servers="localhost:9092"):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=servers,
        value_deserializer=lambda x: json.loads(x.decode())
    )
    for msg in consumer:
        yield msg.value

# ---------------------------
# Stream select
# ---------------------------
if source == "Simulated":
    stream = simulated_stream()
elif source == "Kafka" and KafkaConsumer:
    stream = kafka_stream()
else:
    st.error("Streaming backend not available")
    st.stop()

# ---------------------------
# Autorefresh
# ---------------------------
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=refresh * 1000, key="refresh")

# ---------------------------
# Inference
# ---------------------------
data = next(stream)

seq = np.concatenate(
    [data["power_seq"][:, None],
     np.repeat(data["weather"][None, :], HISTORY_STEPS, axis=0)],
    axis=1
)[None, :, :]

img = data["image_emb"][None, :]

forecast = model.predict(seq, img).flatten()

# uncertainty bands (temporary)
lower = forecast * 0.95
upper = forecast * 1.05

df = pd.DataFrame({
    "Forecast": forecast,
    "Lower": lower,
    "Upper": upper
})

st.line_chart(df)
