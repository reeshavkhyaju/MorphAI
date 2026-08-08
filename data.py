import os
from pathlib import Path
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw" / "img_align_celeba"

RAW_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading/loading dataset...")
ds = load_dataset("korexyz/celeba-hq-256x256")
full_data = ds['train']

print(f"Saving {len(full_data)} raw images to {RAW_DIR}...")
for i, item in enumerate(full_data):
    img = item['image']
    file_path = RAW_DIR / f"{i:05d}.jpg"
    img.save(file_path, "JPEG", quality=100)

print("Done! All raw JPEG images are saved in raw/img_align_celeba/")