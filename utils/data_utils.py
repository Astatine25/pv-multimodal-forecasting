import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
import re

IMG_SIZE = 64
IMG_EMB_DIM = 64

def load_pv_data(csv_path):
    pv_df = pd.read_csv(csv_path)
    pv_df.columns = ["timestamp", "power"]
    pv_df["timestamp"] = pd.to_datetime(pv_df["timestamp"])
    pv_df = pv_df.set_index("timestamp").sort_index()
    pv_df = pv_df.resample("15min").mean().interpolate()
    scaler = StandardScaler()
    pv_df["power_scaled"] = scaler.fit_transform(pv_df[["power"]])
    return pv_df, scaler

def load_weather_data(csv_path, cols):
    weather_df = pd.read_csv(csv_path)
    weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])
    weather_df = weather_df.set_index("DATE_TIME")[cols]
    weather_df = weather_df.resample("1min").mean().interpolate().ffill().bfill()
    return weather_df

def load_and_encode_images(img_dir, image_encoder, limit=20000):
    images, times = [], []
    files = sorted(Path(img_dir).rglob("*.jpg"))[:limit]
    for f in files:
        match = re.search(r"\d{14}", f.stem)
        if not match:
            continue
        ts = pd.to_datetime(match.group(), format="%Y%m%d%H%M%S").floor("1min")
        img = Image.open(f).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
        images.append(np.asarray(img)/255.0)
        times.append(ts)
    images = np.array(images)
    emb = image_encoder.predict(images, batch_size=32, verbose=1)
    return pd.DataFrame(emb, index=pd.to_datetime(times)).sort_index()
