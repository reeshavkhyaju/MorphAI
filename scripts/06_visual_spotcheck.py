import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "data" / "processed" / "manifests" / "train_manifest.json"

def spotcheck():
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest file not found at '{MANIFEST_PATH}'. Please run 05_build_manifest.py first.")
        return

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    if not manifest:
        print("Error: Train manifest is empty.")
        return

    # Grab the first manifest triplet
    sample = manifest[0]
    print(f"Inspecting sample mask type: {sample['mask_type']}")

    img_p = BASE_DIR / sample["image_path"]
    mask_p = BASE_DIR / sample["mask_path"]
    heat_p = BASE_DIR / sample["heatmap_path"]

    # Read RGB image, Mask, and Heatmap using OpenCV
    img = cv2.imread(str(img_p))
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    heatmap = cv2.imread(str(heat_p), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None or heatmap is None:
        print("Error loading sample image, mask, or heatmap files.")
        return

    # Highlight masked region in bright red over the image
    masked_img = img.copy()
    masked_img[mask > 0] = [0, 0, 255]

    # Convert grayscale landmark heatmap to JET colormap visualization
    heat_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Combine images side-by-side: [Original Image | Masked Image | Landmark Heatmap]
    canvas = np.hstack([img, masked_img, heat_color])

    out_path = BASE_DIR / "data" / "processed" / "manifests" / "spotcheck_canvas.png"
    cv2.imwrite(str(out_path), canvas)

    print(f"Spotcheck canvas saved to: {out_path}")
    
    # Automatically pop up visual preview
    Image.open(out_path).show()

if __name__ == "__main__":
    spotcheck()