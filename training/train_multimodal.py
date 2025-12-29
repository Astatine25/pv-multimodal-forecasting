from utils.data_utils import load_pv_data, load_weather_data, load_and_encode_images
from models/multimodal_transformer import build_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

HISTORY_STEPS = 12
FUTURE_STEPS = 6
WEATHER_COLS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]

# Load data
pv_df, scaler = load_pv_data("data/2019_pv_raw.csv")
weather_df = load_weather_data("data/Plant_2_Weather_Sensor_Data.csv", WEATHER_COLS)

# Load image encoder
from tensorflow.keras import layers, models, Input
def build_image_encoder():
    inp = Input(shape=(64,64,3))
    x = layers.Conv2D(32,3,activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64,3,activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(64, activation="relu")(x)
    return models.Model(inp, out)

image_encoder = build_image_encoder()
img_emb_df = load_and_encode_images("https://github.com/Astatine25/pv-multimodal-forecasting/releases/tag/images", image_encoder)

# Merge modalities
full_df = pd.merge_asof(pv_df.sort_index(), img_emb_df.sort_index(), direction="backward")
full_df = pd.merge_asof(full_df.sort_index(), weather_df.sort_index(), direction="backward")
full_df = full_df.dropna()

# Build sequences
def build_sequences(df):
    X_p, X_i, X_w, y = [], [], [], []
    power = df["power_scaled"].values
    img_vals = df.iloc[:,1:1+64].values
    weather_vals = df[WEATHER_COLS].values
    for i in range(len(df)-HISTORY_STEPS-FUTURE_STEPS):
        X_p.append(power[i:i+HISTORY_STEPS])
        X_i.append(img_vals[i+HISTORY_STEPS])
        X_w.append(weather_vals[i+HISTORY_STEPS])
        y.append(power[i+HISTORY_STEPS:i+HISTORY_STEPS+FUTURE_STEPS])
    return np.array(X_p)[...,None], np.array(X_i), np.array(X_w), np.array(y)

X_p, X_i, X_w, y = build_sequences(full_df)

def expand_weather(X_p, X_w):
    w_seq = np.repeat(X_w[:,None,:], HISTORY_STEPS, axis=1)
    return np.concatenate([X_p, w_seq], axis=-1)

X_pw = expand_weather(X_p, X_w)

# Train-test split
split = int(0.8*len(X_pw))
X_tr, X_te = X_pw[:split], X_pw[split:]
X_i_tr, X_i_te = X_i[:split], X_i[split:]
y_tr, y_te = y[:split], y[split:]

# Build model
model = build_model(HISTORY_STEPS, len(WEATHER_COLS))

# Train
model.fit([X_tr, X_i_tr], y_tr,
          epochs=30, batch_size=32,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
          verbose=1)

# Save model
model.save("models/checkpoints/multimodal_model")
print("✅ Model training complete and saved.")
