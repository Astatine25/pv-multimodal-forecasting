# -------------------------
# inference/realtime_inference.py
# -------------------------

import torch
from pathlib import Path
import sys

# Add repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the SAME model used in training
from models.multimodal_forecast import MultimodalForecastModel


class RealtimePredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1️⃣ Build model architecture
        self.model = MultimodalForecastModel().to(self.device)

        # 2️⃣ Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        # 3️⃣ Eval mode
        self.model.eval()

    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        return self.model(x)
