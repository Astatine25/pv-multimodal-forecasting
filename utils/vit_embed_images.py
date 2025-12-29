import torch
import timm
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================
IMAGE_DIR = "data/images"
OUT_DIR = "data/processed"

MODEL_NAME = "vit_base_patch16_224"
IMG_SIZE = 224
BATCH_SIZE = 16

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD ViT
# =====================================================
print("Loading ViT model...")
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

# =====================================================
# LOAD IMAGES + TIMESTAMPS
# =====================================================
image_paths = sorted(Path(IMAGE_DIR).rglob("*.jpg"))
print("Found", len(image_paths), "images")

embeddings = []
timestamps = []

# =====================================================
# ENCODING LOOP
# =====================================================
with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Encoding images"):
        # Extract timestamp from filename
        # Example: 20190131185630.jpg
        try:
            ts = pd.to_datetime(img_path.stem, format="%Y%m%d%H%M%S")
        except Exception:
            continue

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        emb = model(img_tensor).cpu().numpy()[0]
        embeddings.append(emb)
        timestamps.append(ts)

# =====================================================
# SAVE OUTPUTS
# =====================================================
embeddings = np.array(embeddings)
timestamps = pd.DataFrame({"timestamp": timestamps})

np.save(f"{OUT_DIR}/vit_embeddings.npy", embeddings)
timestamps.to_csv(f"{OUT_DIR}/vit_timestamps.csv", index=False)

print("=" * 50)
print("ViT embedding complete")
print("Embeddings shape:", embeddings.shape)
print("Saved to:", OUT_DIR)
print("=" * 50)
