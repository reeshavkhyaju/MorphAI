import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models import LGNetGenerator
from src.landmarks import LandmarkExtractor
from src.masks import MaskGenerator
from src.conditioning import ConditioningGenerator

# All 6 Semantic Regions + 3 Box Mask Variants
SEMANTIC_VARIANTS = [
    "semantic_left_eye",
    "semantic_right_eye",
    "semantic_left_eyebrow",
    "semantic_right_eyebrow",
    "semantic_nose",
    "semantic_mouth",
]

BOX_VARIANTS = ["box_10", "box_25", "box_50"]
ALL_VARIANTS = SEMANTIC_VARIANTS + BOX_VARIANTS


def load_model(checkpoint_path: str, device: torch.device) -> LGNetGenerator:
    """Loads trained LGNetGenerator checkpoint cleanly."""
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    model = LGNetGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("generator_state_dict")
            or checkpoint.get("generator")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    cleaned_state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model


def apply_seamless_blend(orig_np: np.ndarray, gen_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """Seamless Poisson blending of generated area onto original frame."""
    y_indices, x_indices = np.where(mask_np > 127)
    if len(y_indices) == 0:
        return gen_np

    center_x = int((np.min(x_indices) + np.max(x_indices)) / 2)
    center_y = int((np.min(y_indices) + np.max(y_indices)) / 2)

    if len(mask_np.shape) == 3:
        mask_np = mask_np.squeeze()

    try:
        return cv2.seamlessClone(gen_np, orig_np, mask_np, (center_x, center_y), cv2.NORMAL_CLONE)
    except Exception:
        feathered_mask = (cv2.GaussianBlur(mask_np.astype(np.float32), (21, 21), 5.0) / 255.0)[:, :, None]
        blended = (orig_np.astype(np.float32) * (1.0 - feathered_mask)) + (gen_np.astype(np.float32) * feathered_mask)
        return np.clip(blended, 0, 255).astype(np.uint8)


def load_tensor_from_file(img_path: Path, mode: str = "RGB", target_size=(256, 256)) -> tuple[np.ndarray, torch.Tensor]:
    """Loads image/mask/heatmap file and converts to torch tensor."""
    pil_img = Image.open(img_path).convert(mode).resize(target_size)
    arr_np = np.array(pil_img)

    if mode == "RGB":
        tensor = transforms.ToTensor()(pil_img).unsqueeze(0)  # [1, 3, H, W]
    else:
        tensor = (torch.from_numpy(arr_np).float() / 255.0).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    return arr_np, tensor


@torch.inference_mode()
def run_dataset_inference(model, image_path: Path, mask_path: Path, heatmap_path: Path, device: torch.device):
    """Loads pre-computed dataset files directly from disk."""
    orig_np, img_tensor = load_tensor_from_file(image_path, mode="RGB")
    mask_np, mask_tensor = load_tensor_from_file(mask_path, mode="L")
    _, heatmap_tensor = load_tensor_from_file(heatmap_path, mode="L")

    img_tensor = img_tensor.to(device)
    mask_tensor = (mask_tensor > 0.5).float().to(device)
    heatmap_tensor = heatmap_tensor.to(device)

    norm_img = (img_tensor * 2.0) - 1.0
    masked_img = norm_img * (1.0 - mask_tensor) + (-1.0) * mask_tensor

    input_5ch = torch.cat([masked_img, mask_tensor, heatmap_tensor], dim=1).float()

    raw_output = model(input_5ch)

    gen_tensor = (raw_output.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    gen_np = (gen_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    masked_viz_tensor = (masked_img.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    masked_viz_np = (masked_viz_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    blended_np = apply_seamless_blend(orig_np, gen_np, mask_np)
    return orig_np, masked_viz_np, blended_np


@torch.inference_mode()
def run_dynamic_inference(model, image_path: Path, variant_name: str, device: torch.device, extractor, mask_gen, cond_gen):
    """Generates mask and heatmap on-the-fly for custom raw photos."""
    orig_np, img_tensor = load_tensor_from_file(image_path, mode="RGB")
    img_tensor = img_tensor.to(device)
    norm_img = (img_tensor * 2.0) - 1.0

    landmarks = extractor.extract_landmarks(image_path)
    regions = extractor.get_semantic_regions(landmarks)

    if variant_name.startswith("semantic_") and regions is not None:
        region_key = variant_name.replace("semantic_", "")
        region_pts = regions.get(region_key, None)
        mask_np = mask_gen.generate_semantic_mask(region_pts)
    elif "box_10" in variant_name:
        mask_np = mask_gen.generate_random_box_mask(0.10)
    elif "box_25" in variant_name:
        mask_np = mask_gen.generate_random_box_mask(0.25)
    elif "box_50" in variant_name:
        mask_np = mask_gen.generate_random_box_mask(0.50)
    else:
        mask_np = mask_gen.generate_random_box_mask(0.25)

    heatmap_np = cond_gen.generate_heatmap(landmarks, mask=mask_np)

    mask_tensor = (torch.from_numpy(mask_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    heatmap_tensor = (torch.from_numpy(heatmap_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    masked_img = norm_img * (1.0 - mask_tensor) + (-1.0) * mask_tensor
    input_5ch = torch.cat([masked_img, mask_tensor, heatmap_tensor], dim=1).float()

    raw_output = model(input_5ch)

    gen_tensor = (raw_output.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    gen_np = (gen_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    masked_viz_tensor = (masked_img.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    masked_viz_np = (masked_viz_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    blended_np = apply_seamless_blend(orig_np, gen_np, mask_np)
    return orig_np, masked_viz_np, blended_np


def save_grid_dashboard(orig, results_dict, output_path):
    """Saves a multi-row visual dashboard comparing original face against all semantic reconstructions."""
    rows = []
    for var_name, (masked_img, reconstructed) in results_dict.items():
        # Label header
        clean_label = var_name.replace("semantic_", "").replace("_", " ").upper()
        row = np.hstack([orig, masked_img, reconstructed])
        
        # Add text overlay for clear visualization
        cv2.putText(row, clean_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        rows.append(row)

    grid_dashboard = np.vstack(rows)
    Image.fromarray(grid_dashboard).save(output_path)
    print(f"📊 Saved complete All-Semantic Dashboard: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MorphAI Full-Semantic Facial Inpainting Pipeline")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/morphai_epoch_05.pt", help="Path to checkpoint")

    # Dataset Mode Options
    parser.add_argument("--image_id", type=str, default=None, help="Dataset Image ID (e.g., 000001)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    
    # Semantic Variant Options
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "all_semantic"] + ALL_VARIANTS,
        help="Select a specific feature OR 'all' / 'all_semantic' to run all semantic features at once",
    )

    # Custom Raw Image Mode Options
    parser.add_argument("--image_path", type=str, default=None, help="Path to custom new image")
    parser.add_argument("--out", type=str, default="result.png", help="Output filename")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device)

    # Determine which variants to process
    if args.variant in ["all", "all_semantic"]:
        variants_to_run = SEMANTIC_VARIANTS if args.variant == "all_semantic" else ALL_VARIANTS
    else:
        variants_to_run = [args.variant]

    results_dict = {}
    orig_image = None

    # 1. Dataset Split Mode
    if args.image_id is not None:
        processed_dir = BASE_DIR / "data" / "processed"
        img_dir = processed_dir / args.split
        mask_dir = processed_dir / "masks" / args.split
        heatmap_dir = processed_dir / "heatmaps" / args.split

        img_path = img_dir / f"{args.image_id}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{args.image_id}.png"

        if not img_path.exists():
            print(f"❌ Error: Image for ID '{args.image_id}' not found in {img_dir}")
            return

        print(f"\n🚀 Running Inference for Image ID '{args.image_id}' across {len(variants_to_run)} variants...")

        for var in variants_to_run:
            mask_path = mask_dir / f"{args.image_id}_{var}.png"
            heatmap_path = heatmap_dir / f"{args.image_id}_{var}.png"

            if not mask_path.exists() or not heatmap_path.exists():
                print(f"⚠️ Skipping variant '{var}': Precomputed files missing.")
                continue

            orig, masked_input, reconstructed = run_dataset_inference(
                model, img_path, mask_path, heatmap_path, device
            )
            orig_image = orig
            results_dict[var] = (masked_input, reconstructed)
            print(f"  ✅ Completed: {var}")

    # 2. Custom Raw Image Mode
    elif args.image_path is not None:
        img_path = Path(args.image_path)
        if not img_path.exists():
            print(f"❌ Error: Image path '{args.image_path}' does not exist.")
            return

        print(f"\n🚀 Running On-The-Fly Dynamic Inference for '{img_path.name}' across {len(variants_to_run)} variants...")
        extractor = LandmarkExtractor()
        mask_gen = MaskGenerator()
        cond_gen = ConditioningGenerator()

        for var in variants_to_run:
            orig, masked_input, reconstructed = run_dynamic_inference(
                model, img_path, var, device, extractor, mask_gen, cond_gen
            )
            orig_image = orig
            results_dict[var] = (masked_input, reconstructed)
            print(f"  ✅ Completed: {var}")

        extractor.close()

    else:
        print("❌ Error: Please specify either --image_id (for dataset) or --image_path (for custom photo).")
        return

    # Save Output Results
    if len(results_dict) == 1:
        var_name = list(results_dict.keys())[0]
        masked_viz, reconstructed = results_dict[var_name]
        Image.fromarray(reconstructed).save(args.out)
        print(f"\n💾 Saved reconstructed image: {args.out}")

        grid_np = np.hstack([orig_image, masked_viz, reconstructed])
        grid_path = f"grid_{Path(args.out).name}"
        Image.fromarray(grid_np).save(grid_path)
        print(f"🖼️ Saved comparison grid: {grid_path}")

    elif len(results_dict) > 1:
        dashboard_path = f"dashboard_{Path(args.out).name}"
        save_grid_dashboard(orig_image, results_dict, dashboard_path)


if __name__ == "__main__":
    main()