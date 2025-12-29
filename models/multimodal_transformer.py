import torch
import torch.nn as nn

class MultimodalTransformer(nn.Module):
    def __init__(
        self,
        weather_dim,
        img_embed_dim=128,
        d_model=128,
        nhead=4,
        num_layers=2
    ):
        super().__init__()

        self.weather_fc = nn.Linear(weather_dim, d_model)
        self.img_fc = nn.Linear(img_embed_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.regressor = nn.Linear(d_model, 1)

    def forward(self, weather, image_embed):
        """
        weather: [B, T, weather_dim]
        image_embed: [B, T, img_embed_dim]
        """
        w = self.weather_fc(weather)
        i = self.img_fc(image_embed)

        x = w + i
        x = self.transformer(x)

        return self.regressor(x[:, -1])
