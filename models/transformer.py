import torch
import torch.nn as nn

class MultimodalTransformer(nn.Module):
    def __init__(self, ts_dim, embed_dim=128, heads=4, layers=3):
        super().__init__()
        self.ts_proj = nn.Linear(ts_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, ts_seq, img_emb):
        ts_emb = self.ts_proj(ts_seq)
        fused = ts_emb + img_emb.unsqueeze(1)
        enc_out = self.encoder(fused)
        return self.regressor(enc_out[:, -1])
