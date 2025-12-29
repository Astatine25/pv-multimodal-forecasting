import timm
import torch
import torch.nn as nn

class ViTEncoder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )
        self.proj = nn.Linear(self.vit.num_features, emb_dim)

    def forward(self, x):
        feats = self.vit(x)
        return self.proj(feats)
