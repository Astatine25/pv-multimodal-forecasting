import torch
import torch.nn as nn
from torchvision import models

class CNNImageEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.proj(x)


class ViTImageEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        vit = models.vit_b_16(pretrained=True)
        vit.heads = nn.Identity()
        self.vit = vit
        self.proj = nn.Linear(768, out_dim)

    def forward(self, x):
        x = self.vit(x)
        return self.proj(x)
