import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MANIFEST_DIR = PROCESSED_DIR / "manifests"

# Known 9 mask variant suffixes generated in Step 04
MASK_VARIANTS = [
    "semantic_left_eye",
    "semantic_right_eye",
    "semantic_left_eyebrow",
    "semantic_right_eyebrow",
    "semantic_nose",
    "semantic_mouth",
    "box_10",
    "box_25",
    "box_50",
]

def build_manifests():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]

    for split in splits:
        img_dir = PROCESSED_DIR / split
        mask_dir = PROCESSED_DIR / "masks" / split
        heatmap_dir = PROCESSED_DIR / "heatmaps" / split

        if not img_dir.exists():
            continue

        print(f"Building manifest for {split} split...")
        manifest_entries = []
        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

        for i, img_path in enumerate(images):
            base_name = img_path.stem

            # Direct lookup instead of expensive folder-wide globbing
            for variant in MASK_VARIANTS:
                mask_path = mask_dir / f"{base_name}_{variant}.png"
                heatmap_path = heatmap_dir / f"{base_name}_{variant}.png"

                if mask_path.exists() and heatmap_path.exists():
                    manifest_entries.append({
                        "image_path": str(img_path.relative_to(BASE_DIR)),
                        "mask_path": str(mask_path.relative_to(BASE_DIR)),
                        "heatmap_path": str(heatmap_path.relative_to(BASE_DIR)),
                        "mask_type": variant
                    })

            if (i + 1) % 5000 == 0 or (i + 1) == len(images):
                print(f"[{split}] Indexed {i + 1}/{len(images)} images...")

        out_json = MANIFEST_DIR / f"{split}_manifest.json"
        with open(out_json, "w") as f:
            json.dump(manifest_entries, f, indent=2)

        print(f"Saved {len(manifest_entries)} entries to {out_json}\n")

if __name__ == "__main__":
    build_manifests()