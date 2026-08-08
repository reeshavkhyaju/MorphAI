import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.landmarks import LandmarkExtractor

def check_landmarks():
    train_dir = BASE_DIR / "data" / "processed" / "train"
    output_dir = BASE_DIR / "data" / "processed" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find first available processed training image
    sample_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    if not sample_images:
        print(f"Error: No images found in '{train_dir}'.")
        return

    sample_path = sample_images[0]
    print(f"Testing landmark extraction on image: {sample_path.name}")

    extractor = LandmarkExtractor()
    landmarks = extractor.extract_landmarks(sample_path)

    if landmarks is None:
        print(f"Warning: No face detected in {sample_path.name}.")
        extractor.close()
        return

    print(f"Successfully detected {len(landmarks)} landmarks!")

    # Load image for drawing
    img = cv2.imread(str(sample_path))

    # Draw all landmarks as small green dots
    for (x, y) in landmarks:
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)

    # Get semantic regions and draw bounding convex hulls
    regions = extractor.get_semantic_regions(landmarks)
    colors = {
        "left_eye": (255, 0, 0),
        "right_eye": (255, 0, 0),
        "left_eyebrow": (0, 255, 255),
        "right_eyebrow": (0, 255, 255),
        "nose": (0, 165, 255),
        "mouth": (0, 0, 255)
    }

    for region_name, pts in regions.items():
        pts_int = pts.astype(np.int32)
        hull = cv2.convexHull(pts_int)
        cv2.polylines(img, [hull], True, colors.get(region_name, (255, 255, 255)), 1)

    # Save output visualization
    out_file = output_dir / "landmark_verification.jpg"
    cv2.imwrite(str(out_file), img)
    print(f"Verification image saved to: {out_file}")

    extractor.close()

    # Automatically launch the image preview on screen
    print("Opening visual preview window...")
    Image.open(out_file).show()

if __name__ == "__main__":
    check_landmarks()