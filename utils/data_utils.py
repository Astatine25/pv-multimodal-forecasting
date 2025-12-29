import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms

from models.vit_encoder import ViTEncoder

# -------------------------
# ViT setup (CPU-safe)
# -------------------------
device = "cpu"
vit = ViTEncoder(emb_dim=64).to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# IMAGE LOADER + ENCODER
# -------------------------
def load_and_encode_images(img_dir, batch_size=16):
    images, timestamps = [], []

    files = sorted(Path(img_dir).rglob("*.jpg"))

    for f in files:
        # Extract timestamp from filename
        ts = pd.to_datetime(f.stem, format="%Y%m%d%H%M%S", errors="coerce")
        if pd.isna(ts):
            continue

        img = Image.open(f).convert("RGB")
        images.append(transform(img))
        timestamps.append(ts)

    if len(images) == 0:
        raise ValueError("No valid images found")

    images = torch.stack(images).to(device)

    # 🔑 YOUR SNIPPET RUNS HERE
    with torch.no_grad():
        emb = vit(images).cpu().numpy()

    emb_df = pd.DataFrame(
        emb,
        index=pd.to_datetime(timestamps)
    ).sort_index()

    return emb_df
