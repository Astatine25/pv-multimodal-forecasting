# ==============================================================================
# Real-Time Inference Script
# Author: Vivek Palsutkar
# ==============================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

MODEL_PATH = "models/multimodal_pv_model.keras"
SCALER_PATH = "models/power_scaler.save"
SEQ_LEN = 12

# ---------------- LOAD ----------------
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------- REALTIME PREDICT ----------------
def predict_next(power_window):
    """
    power_window: list or np.array of last SEQ_LEN raw power values
    """
    power_window = np.array(power_window).reshape(-1, 1)
    power_scaled = scaler.transform(power_window)

    X = power_scaled.reshape(1, SEQ_LEN, 1)
    pred_scaled = model.predict(X, verbose=0)

    pred = scaler.inverse_transform(pred_scaled)
    return float(pred[0, 0])


# ---------------- TEST ----------------
if __name__ == "__main__":
    dummy_window = [120, 118, 117, 119, 122, 125, 130, 132, 129, 128, 126, 127]
    prediction = predict_next(dummy_window)
    print(f"🔮 Next-step PV Power Forecast: {prediction:.2f} kW")
