from .multimodal_transformer import MultimodalTransformer

import torch
import torch.nn as nn

HISTORY_STEPS = 24
FEATURE_DIM = 4  # Power + weather features
IMG_EMB_DIM = 64
FUTURE_STEPS = 6

class MultimodalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Sequence encoder
        self.seq_fc = nn.Linear(HISTORY_STEPS*FEATURE_DIM, 128)
        # Image encoder
        self.img_fc = nn.Linear(IMG_EMB_DIM, 64)
        # Fusion
        self.fusion_fc = nn.Linear(128 + 64, 64)
        self.out_fc = nn.Linear(64, FUTURE_STEPS)

    def forward(self, seq_x, img_x):
        # Flatten sequence
        seq_x = seq_x.view(seq_x.size(0), -1)
        seq_feat = torch.relu(self.seq_fc(seq_x))
        img_feat = torch.relu(self.img_fc(img_x))
        fused = torch.cat([seq_feat, img_feat], dim=1)
        fused = torch.relu(self.fusion_fc(fused))
        out = self.out_fc(fused)
        return out

def build_model():
    return MultimodalTransformer()
