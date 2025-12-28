import torch
from models.multimodal_forecast import MultimodalForecastModel

class RealtimePredictor:
    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model = MultimodalForecastModel()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.quantiles = ckpt["quantiles"]

    @torch.no_grad()
    def predict(self, weather_feats, image_feats):
        preds = self.model(weather_feats, image_feats)
        return preds.squeeze(0)
