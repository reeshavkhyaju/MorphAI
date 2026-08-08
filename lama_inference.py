import argparse
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from simple_lama_inpainting import SimpleLama
    HAS_LAMA = True
except ImportError:
    HAS_LAMA = False


def extract_id(p: Path) -> str:
    """Extracts numeric ID or clean stem from a filename."""
    s = p.stem.lower()
    s = re.sub(r'(_mask|_gt|_img|_image|_masked|_input|_result|_pred)', '', s)
    digits = re.findall(r'\d+', s)
    return digits[0] if digits else s


def find_image_mask_pairs(input_dir: Path, mask_dir: Path):
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    img_files = [f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]
    mask_files = [f for f in mask_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]

    print(f"📊 Discovered {len(img_files)} images in '{input_dir}'")
    print(f"📊 Discovered {len(mask_files)} masks in '{mask_dir}'")

    if not img_files or not mask_files:
        return []

    # Index images by stem and numerical ID
    img_map = {img.stem.lower(): img for img in img_files}
    img_id_map = {extract_id(img): img for img in img_files if extract_id(img)}

    pairs = []
    for mask in mask_files:
        mask_stem = mask.stem.lower()
        mask_id = extract_id(mask)

        # Match mask to corresponding ground truth image
        matched_img = (
            img_map.get(mask_stem) or
            img_map.get(re.sub(r'(_mask|mask_)', '', mask_stem)) or
            img_id_map.get(mask_id)
        )

        if matched_img:
            # Preserve mask relative subfolder path so prediction files retain category metadata
            rel_mask_path = mask.relative_to(mask_dir)
            pairs.append((matched_img, mask, rel_mask_path))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="LaMa Batch Inference Engine")
    parser.add_argument("--input_dir", type=str, default="data/processed/test", help="Path to ground truth images")
    parser.add_argument("--mask_dir", type=str, default="data/processed/masks", help="Path to mask directory")
    parser.add_argument("--output_dir", type=str, default="samples/lama_comparison", help="Output prediction path")
    args = parser.parse_args()

    if not HAS_LAMA:
        print("❌ Package 'simple-lama-inpainting' missing. Install via: pip install simple-lama-inpainting")
        return

    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Input Path:  '{input_dir}'")
    print(f"📁 Mask Path:   '{mask_dir}'")
    print(f"📁 Output Path: '{output_dir}'")

    pairs = find_image_mask_pairs(input_dir, mask_dir)
    print(f"✅ Successfully paired {len(pairs)} image-mask samples.")

    if not pairs:
        print("\n❌ Could not match image and mask files. Verify file ID numbers align between directories.")
        return

    print("\n📦 Loading LaMa Model Weights...")
    simple_lama = SimpleLama()

    print(f"\n⚡ Running Inpainting Inference across {len(pairs)} samples...\n")
    for img_path, mask_path, rel_mask_path in tqdm(pairs, desc="Inpainting"):
        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            result = simple_lama(img, mask)

            out_file = output_dir / rel_mask_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

            result.save(out_file)
        except Exception as e:
            print(f"⚠️ Error processing {img_path.name}: {e}")

    print(f"\n✅ Finished! Generated predictions saved to: '{output_dir}'")


if __name__ == "__main__":
    main()