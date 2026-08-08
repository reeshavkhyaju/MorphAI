import os
from pathlib import Path
from PIL import Image

# 1. Path Definitions
BASE_DIR = Path(__file__).resolve().parent.parent

# Detect raw directory (checks data/raw/img_align_celeba first, then falls back to data/raw)
RAW_DIR_SUB = BASE_DIR / "data" / "raw" / "img_align_celeba"
RAW_DIR_BASE = BASE_DIR / "data" / "raw"

if RAW_DIR_SUB.exists() and any(RAW_DIR_SUB.iterdir()):
    RAW_DIR = RAW_DIR_SUB
elif RAW_DIR_BASE.exists() and any(RAW_DIR_BASE.iterdir()):
    RAW_DIR = RAW_DIR_BASE
else:
    RAW_DIR = RAW_DIR_SUB  # Target default path if initializing

# FIXED: Processed folder is now nested under 'data/processed'
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Pipeline subdirectories inside data/processed/
PROCESSED_SUBDIRS = [
    "train",
    "val",
    "test",
    "masks",
    "heatmaps",
    "landmarks",
    "manifests"
]

def setup_processed_directories():
    """Creates the processed directory structure inside data/processed/."""
    print(f"Creating processed directories under: {PROCESSED_DIR}")
    for subdir in PROCESSED_SUBDIRS:
        (PROCESSED_DIR / subdir).mkdir(parents=True, exist_ok=True)
    print("Directories initialized successfully.")

def process_raw_to_splits():
    """Reads raw images from disk, converts to 256x256 RGB, and creates 80/10/10 splits."""
    print(f"Scanning raw images from: {RAW_DIR}")
    
    valid_exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([
        p for p in RAW_DIR.glob("*") if p.suffix.lower() in valid_exts
    ])

    total_images = len(image_paths)
    if total_images == 0:
        raise FileNotFoundError(
            f"No image files found in '{RAW_DIR}'. "
            "Please confirm your raw .jpg files are placed in 'data/raw/' or 'data/raw/img_align_celeba/'."
        )

    print(f"Found {total_images} raw images.")

    # Calculate exact splits: 80% Train, 10% Val, 10% Test
    train_end = int(total_images * 0.8)
    val_end = train_end + int(total_images * 0.1)

    splits = {
        "train": image_paths[:train_end],
        "val": image_paths[train_end:val_end],
        "test": image_paths[val_end:]
    }

    for split_name, paths in splits.items():
        split_dir = PROCESSED_DIR / split_name
        print(f"Processing {split_name} split ({len(paths)} images) -> {split_dir}")

        for img_path in paths:
            with Image.open(img_path) as img:
                # Ensure standard 3-channel RGB format
                img = img.convert("RGB")
                
                # Resize to 256x256 for LGNet
                if img.size != (256, 256):
                    img = img.resize((256, 256), Image.Resampling.LANCZOS)
                
                # Save to target split folder inside data/processed/
                img.save(split_dir / img_path.name, "JPEG", quality=95)

    print("\nAll raw images processed and saved into data/processed/ (train, val, test) successfully!")

if __name__ == "__main__":
    setup_processed_directories()
    process_raw_to_splits()