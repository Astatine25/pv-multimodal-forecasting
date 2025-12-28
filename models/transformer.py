import torch
import torch.nn as nn

class MultimodalTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.head(x[:, -1])
