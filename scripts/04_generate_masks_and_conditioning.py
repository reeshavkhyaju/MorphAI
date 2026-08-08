import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.landmarks import LandmarkExtractor
from src.masks import MaskGenerator
from src.conditioning import ConditioningGenerator

def process_dataset_splits():
    processed_dir = BASE_DIR / "data" / "processed"
    masks_base_dir = processed_dir / "masks"
    heatmaps_base_dir = processed_dir / "heatmaps"

    extractor = LandmarkExtractor()
    mask_gen = MaskGenerator()
    cond_gen = ConditioningGenerator()

    splits = ["train", "val", "test"]

    for split in splits:
        split_img_dir = processed_dir / split
        if not split_img_dir.exists():
            continue

        images = sorted(list(split_img_dir.glob("*.jpg")) + list(split_img_dir.glob("*.png")))
        print(f"\nProcessing {split} split ({len(images)} images)...")

        # Create output directories per split
        split_mask_dir = masks_base_dir / split
        split_heatmap_dir = heatmaps_base_dir / split
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        split_heatmap_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(images):
            landmarks = extractor.extract_landmarks(img_path)
            regions = extractor.get_semantic_regions(landmarks)

            base_name = img_path.stem

            # Define the 9 mask types
            mask_variants = {}

            # 1-6. Semantic masks
            if regions is not None:
                for region_name, region_pts in regions.items():
                    mask_variants[f"semantic_{region_name}"] = mask_gen.generate_semantic_mask(region_pts)
            else:
                # Fallback if no landmarks detected
                for reg in ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "nose", "mouth"]:
                    mask_variants[f"semantic_{reg}"] = mask_gen.generate_random_box_mask(0.1)

            # 7-9. Random coverage masks
            mask_variants["box_10"] = mask_gen.generate_random_box_mask(coverage=0.10)
            mask_variants["box_25"] = mask_gen.generate_random_box_mask(coverage=0.25)
            mask_variants["box_50"] = mask_gen.generate_random_box_mask(coverage=0.50)

            # Save each mask variant and its corresponding landmark conditioning map
            for variant_name, mask in mask_variants.items():
                heatmap = cond_gen.generate_heatmap(landmarks, mask=mask)

                mask_file = split_mask_dir / f"{base_name}_{variant_name}.png"
                heatmap_file = split_heatmap_dir / f"{base_name}_{variant_name}.png"

                cv2.imwrite(str(mask_file), mask)
                cv2.imwrite(str(heatmap_file), heatmap)

            if (i + 1) % 500 == 0 or (i + 1) == len(images):
                print(f"[{split}] Processed {i + 1}/{len(images)} images...")

    extractor.close()
    print("\nMask and Landmark Conditioning Map generation complete!")

if __name__ == "__main__":
    process_dataset_splits()