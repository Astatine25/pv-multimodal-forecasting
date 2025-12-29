# dashboard/app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from utils.data_utils import load_and_encode_images, load_weather_data
from models.multimodal_transformer import build_model
import paho.mqtt.client as mqtt
from kafka import KafkaConsumer

st.set_page_config(page_title="PV Power Forecasting", layout="wide")

# -----------------------------
# Load model
# -----------------------------
model = build_model()
model.load_weights("models/checkpoints/multimodal_model")  # load trained weights

# -----------------------------
# Load Weather Data
# -----------------------------
WEATHER_COLS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
weather_df = load_weather_data("data/Plant_2_Weather_Sensor_Data.csv", WEATHER_COLS)

# -----------------------------
# Load Images (ViT embeddings)
# -----------------------------
image_encoder = None
try:
    from timm import create_model
    import torch
    class ViTEncoder:
        def __init__(self):
            self.model = create_model('vit_base_patch16_224', pretrained=True)
            self.model.eval()

        def __call__(self, img_tensor):
            with torch.no_grad():
                return self.model.forward(img_tensor).numpy()

    image_encoder = ViTEncoder()
except:
    st.warning("timm / ViT not installed. Using placeholder embeddings.")

# -----------------------------
# MQTT / Kafka Streams (Optional)
# -----------------------------
KAFKA_TOPIC = "pv_forecast"
MQTT_BROKER = "localhost"

def on_message(client, userdata, msg):
    st.write(f"MQTT message received: {msg.payload}")

client = mqtt.Client()
client.on_message = on_message
# client.connect(MQTT_BROKER)
# client.loop_start()
# client.subscribe("pv/forecast")

# -----------------------------
# User Interface
# -----------------------------
st.title("Multimodal PV Power Forecasting")

uploaded_image = st.file_uploader("Upload sky image", type=["jpg", "png"])
weather_input = st.text_input("Enter weather CSV path (optional)", "data/Plant_2_Weather_Sensor_Data.csv")

if uploaded_image is not None:
    img_arr = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(uploaded_image, target_size=(224,224))
    ) / 255.0

    if image_encoder:
        img_tensor = np.expand_dims(img_arr, 0)
        img_emb = image_encoder(img_tensor)  # shape (1, IMG_EMB_DIM)
    else:
        img_emb = np.random.rand(1, 64)  # fallback

    # Dummy historical power + weather
    X_seq = np.random.rand(1, 12, 1 + len(WEATHER_COLS))

    # Predict
    pred = model.predict([X_seq, img_emb])[0]  # shape (3, FUTURE_STEPS)
    lower, median, upper = pred[0], pred[1], pred[2]

    st.line_chart(pd.DataFrame({
        "Lower": lower,
        "Median": median,
        "Upper": upper
    }))
