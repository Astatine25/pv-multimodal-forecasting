import tensorflow as tf
import numpy as np

class RealtimePredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, X_seq, X_img):
        return self.model.predict([X_seq, X_img])
