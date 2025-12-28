import torch
from models.transformer import MultimodalTransformer

class RealtimePredictor:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def predict(self, x):
        with torch.no_grad():
            return self.model(x)
