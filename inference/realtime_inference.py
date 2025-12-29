# inference/realtime_inference.py

import tensorflow as tf
import numpy as np
from pathlib import Path

class RealtimePredictor:
    def __init__(self, model_path="models/checkpoints/multimodal_model.keras"):
        self.model = tf.keras.models.load_model(model_path)

    @tf.function
    def _predict(self, x_seq, x_img):
        return self.model([x_seq, x_img], training=False)

    def predict(self, seq_tensor, img_tensor):
        """
        seq_tensor: (1, HISTORY_STEPS, 1 + weather_dim)
        img_tensor: (1, IMG_EMB_DIM)
        """
        seq_tensor = tf.convert_to_tensor(seq_tensor, dtype=tf.float32)
        img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

        preds = self._predict(seq_tensor, img_tensor)
        return preds.numpy()
