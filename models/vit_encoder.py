import timm
import torch.nn as nn

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.vit = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True
        )
        self.vit.reset_classifier(0)
        self.fc = nn.Linear(self.vit.num_features, embed_dim)

    def forward(self, x):
        return self.fc(self.vit(x))
