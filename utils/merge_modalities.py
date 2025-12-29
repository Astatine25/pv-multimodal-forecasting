"""
Merge PV power, weather data, and ViT image embeddings
into a single time-aligned multimodal dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
PV_PATH = "data/2019_pv_raw.csv"
WEATHER_PATH = "data/Plant_2_Weather_Sensor_Data.csv"

VIT_EMB_PATH = "data/processed/vit_embeddings.npy"
VIT_TS_PATH = "data/processed/vit_timestamps.csv"

OUTPUT_PATH = "data/processed/merged_multimodal.csv"

WEATHER_COLS = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION"
]

# =====================================================
# LOAD PV DATA
# =====================================================
print("Loading PV data...")
pv_df = pd.read_csv(PV_PATH)
pv_df.columns = ["timestamp", "power"]
pv_df["timestamp"] = pd.to_datetime(pv_df["timestamp"])
pv_df = pv_df.set_index("timestamp").sort_index()

# =====================================================
# LOAD WEATHER DATA
# =====================================================
print("Loading weather data...")
weather_df = pd.read_csv(WEATHER_PATH)
weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])
weather_df = (
    weather_df
    .set_index("DATE_TIME")[WEATHER_COLS]
    .resample("1min")
    .mean()
    .interpolate()
    .ffill()
    .bfill()
)

# =====================================================
# LOAD ViT EMBEDDINGS
# =====================================================
print("Loading ViT embeddings...")
vit_embeddings = np.load(VIT_EMB_PATH)
vit_timestamps = pd.read_csv(VIT_TS_PATH)

vit_timestamps["timestamp"] = pd.to_datetime(vit_timestamps["timestamp"])

img_df = pd.DataFrame(
    vit_embeddings,
    index=vit_timestamps["timestamp"],
    columns=[f"vit_{i}" for i in range(vit_embeddings.shape[1])]
).sort_index()

# =====================================================
# MERGE PV + IMAGE EMBEDDINGS
# =====================================================
print("Merging PV and images...")
merged_df = pd.merge_asof(
    pv_df.sort_index(),
    img_df.sort_index(),
    left_index=True,
    right_index=True,
    direction="backward",
    tolerance=pd.Timedelta("2min")
)

# =====================================================
# MERGE WEATHER
# =====================================================
print("Merging weather data...")
merged_df = pd.merge_asof(
    merged_df.sort_index(),
    weather_df.sort_index(),
    left_index=True,
    right_index=True,
    direction="nearest"
)

merged_df = merged_df.dropna()

# =====================================================
# SAVE OUTPUT
# =====================================================
Path("data/processed").mkdir(parents=True, exist_ok=True)
merged_df.to_csv(OUTPUT_PATH)

print("=" * 50)
print("Multimodal dataset created successfully")
print("Final shape:", merged_df.shape)
print("Saved to:", OUTPUT_PATH)
print("=" * 50)
