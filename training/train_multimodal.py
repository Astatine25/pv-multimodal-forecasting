# training/train_multimodal.py

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

# -------------------------
# Config
# -------------------------
EPOCHS = 10
LR = 1e-3
HORIZON = 24
FEATURE_DIM = 128
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Dummy Multimodal Model
# Replace later with real transformer
# -------------------------
class MultimodalForecastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(HORIZON * FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, HORIZON)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Dummy Dataset (replace later)
# -------------------------
def get_batch(batch_size=16):
    x = torch.rand(batch_size, HORIZON, FEATURE_DIM)
    y = torch.rand(batch_size, HORIZON)
    return x, y

# -------------------------
# Training
# -------------------------
def train():
    model = MultimodalForecastModel()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        x, y = get_batch()
        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        # -------------------------
        # Save checkpoint
        # -------------------------
        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": loss.item(),
            },
            ckpt_path,
        )

        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": loss.item(),
                },
                CHECKPOINT_DIR / "best_model.pt",
            )

    print("✅ Training complete. Checkpoints saved.")

if __name__ == "__main__":
    train()
