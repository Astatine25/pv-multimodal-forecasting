import torch
import torch.optim as optim
from pathlib import Path
import numpy as np

from models.multimodal_forecast import MultimodalForecastModel
from training.quantile_loss import quantile_loss

EPOCHS = 10
HORIZON = 24
FEATURE_DIM = 128
IMAGE_DIM = 128
QUANTILES = (0.1, 0.5, 0.9)

CKPT_DIR = Path("models/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def get_batch(batch_size=8):
    weather = torch.rand(batch_size, FEATURE_DIM)
    image = torch.rand(batch_size, IMAGE_DIM)
    target = torch.rand(batch_size, HORIZON)
    return weather, image, target

def train():
    model = MultimodalForecastModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        weather, image, target = get_batch()
        preds = model(weather, image)

        loss = quantile_loss(preds, target, QUANTILES)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        ckpt = {
            "model_state": model.state_dict(),
            "quantiles": QUANTILES
        }
        torch.save(ckpt, CKPT_DIR / f"epoch_{epoch}.pt")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(ckpt, CKPT_DIR / "best_model.pt")

    print("Training finished.")

if __name__ == "__main__":
    train()
