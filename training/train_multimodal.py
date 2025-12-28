# -------------------------
# training/train_multimodal.py
# -------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

# -------------------------
# Add repo root to PYTHONPATH
# -------------------------
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# -------------------------
# Import shared model
# -------------------------
from models.multimodal_forecast import MultimodalForecastModel

# -------------------------
# Config
# -------------------------
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 16
HORIZON = 24
FEATURE_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Dummy Dataset (replace later)
# -------------------------
def get_batch(batch_size=BATCH_SIZE):
    x = torch.rand(batch_size, HORIZON, FEATURE_DIM)
    y = torch.rand(batch_size, HORIZON)
    return x, y

# -------------------------
# Training Loop
# -------------------------
def train():
    print(f"Training on device: {DEVICE}")

    model = MultimodalForecastModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()

        x, y = get_batch()
        x, y = x.to(DEVICE), y.to(DEVICE)

        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:02d} | Loss: {loss.item():.6f}")

        # -------------------------
        # Save checkpoint
        # -------------------------
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }

        torch.save(ckpt, CHECKPOINT_DIR / f"epoch_{epoch}.pt")

        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(ckpt, CHECKPOINT_DIR / "best_model.pt")

    print("Training complete. Checkpoints saved.")

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    train()
