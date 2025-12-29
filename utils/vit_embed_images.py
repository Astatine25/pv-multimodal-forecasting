import torch
import timm
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

# ===============================
# CONFIG
# ===============================
IMAGE_DIR = Path("data/images")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "vit_base_patch16_224"
BATCH_SIZE = 32        # Safe for CPU (increase to 64 if RAM allows)
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# IMAGE TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

# ===============================
# LOAD ViT MODEL
# ===============================
print("Loading ViT model...")
vit = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0   # IMPORTANT → embeddings only
)
vit.eval()
vit.to(DEVICE)

# ===============================
# LOAD IMAGES
# ===============================
image_files = sorted(IMAGE_DIR.rglob("*.jpg"))
assert len(image_files) > 0, "No images found!"

print(f"Found {len(image_files)} images")

# ===============================
# BATCHED EMBEDDING
# ===============================
all_embeddings = []
all_timestamps = []

batch_images = []
batch_times = []

with torch.no_grad():
    for img_path in tqdm(image_files, desc="Encoding images"):
        try:
            # Parse timestamp from filename
            ts = pd.to_datetime(img_path.stem, format="%Y%m%d%H%M%S", errors="coerce")
            if pd.isna(ts):
                continue

            img = Image.open(img_path).convert("RGB")
            img = transform(img)

            batch_images.append(img)
            batch_times.append(ts)

            # Run batch
            if len(batch_images) == BATCH_SIZE:
                batch_tensor = torch.stack(batch_images).to(DEVICE)
                emb = vit(batch_tensor).cpu().numpy()

                all_embeddings.append(emb)
                all_timestamps.extend(batch_times)

                batch_images.clear()
                batch_times.clear()

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    # Final batch
    if batch_images:
        batch_tensor = torch.stack(batch_images).to(DEVICE)
        emb = vit(batch_tensor).cpu().numpy()
        all_embeddings.append(emb)
        all_timestamps.extend(batch_times)

# ===============================
# SAVE OUTPUTS
# ===============================
embeddings = np.vstack(all_embeddings)
timestamps = np.array(all_timestamps, dtype="datetime64[s]")

np.save(OUTPUT_DIR / "vit_embeddings.npy", embeddings)
np.save(OUTPUT_DIR / "vit_timestamps.npy", timestamps)

print("===================================")
print("ViT embedding complete")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Saved to: {OUTPUT_DIR}")
print("===================================")
