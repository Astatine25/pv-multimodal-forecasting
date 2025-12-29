# training/train_multimodal.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping

# ==========================
# CONFIG
# ==========================
DATA_PATH = "data/processed/merged_multimodal.csv"
MODEL_OUT = "models/checkpoints/multimodal_model"

HISTORY_STEPS = 12
FUTURE_STEPS = 6
N_QUANTILES = 3
IMG_EMB_DIM = 768

# ==========================
# QUANTILE LOSS
# ==========================
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss

# ==========================
# LOAD DATA
# ==========================
print("Loading merged multimodal dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Identify columns
power_col = "POWER_AC"
weather_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
img_cols = [c for c in df.columns if c.startswith("vit_")]

assert len(img_cols) == IMG_EMB_DIM, "ViT embedding dimension mismatch"

# Normalize power
power = df[power_col].values.astype("float32")
power = (power - power.mean()) / (power.std() + 1e-6)

weather = df[weather_cols].values.astype("float32")
images = df[img_cols].values.astype("float32")

# ==========================
# BUILD SEQUENCES
# ==========================
def build_sequences(power, weather, images):
    X_pw, X_img, y = [], [], []

    for i in range(len(power) - HISTORY_STEPS - FUTURE_STEPS):
        pw_seq = np.concatenate(
            [
                power[i:i+HISTORY_STEPS, None],
                weather[i:i+HISTORY_STEPS]
            ],
            axis=1
        )

        X_pw.append(pw_seq)
        X_img.append(images[i+HISTORY_STEPS])
        y.append(power[i+HISTORY_STEPS:i+HISTORY_STEPS+FUTURE_STEPS])

    return (
        np.array(X_pw),
        np.array(X_img),
        np.array(y)
    )

X_pw, X_img, y = build_sequences(power, weather, images)

# ==========================
# TRAIN / TEST SPLIT
# ==========================
split = int(0.8 * len(X_pw))
X_pw_tr, X_pw_te = X_pw[:split], X_pw[split:]
X_img_tr, X_img_te = X_img[:split], X_img[split:]
y_tr, y_te = y[:split], y[split:]

# ==========================
# MODEL
# ==========================
def build_model():
    pw_in = Input(shape=(HISTORY_STEPS, 1 + len(weather_cols)))
    x = layers.Dense(128)(pw_in)
    x = layers.MultiHeadAttention(4, 32)(x, x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    img_in = Input(shape=(IMG_EMB_DIM,))
    img_x = layers.Dense(128, activation="relu")(img_in)

    fusion = layers.Concatenate()([x, img_x])
    fusion = layers.Dense(256, activation="relu")(fusion)
    fusion = layers.Dense(128, activation="relu")(fusion)

    out = layers.Dense(FUTURE_STEPS * N_QUANTILES)(fusion)
    out = layers.Reshape((N_QUANTILES, FUTURE_STEPS))(out)

    model = models.Model([pw_in, img_in], out)

    model.compile(
        optimizer="adam",
        loss=[
            quantile_loss(0.1),
            quantile_loss(0.5),
            quantile_loss(0.9),
        ],
    )

    return model

model = build_model()
model.summary()

# ==========================
# TRAIN
# ==========================
model.fit(
    [X_pw_tr, X_img_tr],
    [y_tr, y_tr, y_tr],
    validation_data=([X_pw_te, X_img_te], [y_te, y_te, y_te]),
    epochs=30,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
)

# ==========================
# SAVE
# ==========================
model.save(MODEL_OUT)
print("Model training complete and saved")
