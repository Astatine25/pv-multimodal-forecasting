import torch
import torch.nn as nn

class MultimodalForecastModel(nn.Module):
    def __init__(self, feature_dim=128, image_dim=128, horizon=24, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        self.horizon = horizon

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + image_dim, 256),
            nn.ReLU(),
            nn.Linear(256, horizon * len(quantiles))
        )

    def forward(self, weather_feats, image_feats):
        x = torch.cat([weather_feats, image_feats], dim=-1)
        out = self.fusion(x)
        return out.view(-1, self.horizon, len(self.quantiles))
