import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

# --- Dynamic Path Resolution (Works in project root OR inside scripts/) ---
FILE_DIR = Path(__file__).resolve().parent
if (FILE_DIR / "src").exists():
    BASE_DIR = FILE_DIR
elif (FILE_DIR.parent / "src").exists():
    BASE_DIR = FILE_DIR.parent
else:
    BASE_DIR = FILE_DIR

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models import LGNetGenerator

# Optional ArcFace import
try:
    from facenet_pytorch import InceptionResnetV1
    HAS_ARCFACE = True
except ImportError:
    HAS_ARCFACE = False


# ==========================================
# 1. ArcFace Identity Similarity Extractor
# ==========================================

class ArcFaceIdentityExtractor(nn.Module):
    """Extracts 512-dim facial identity embeddings to calculate Cosine Similarity."""
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        if HAS_ARCFACE:
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            self.model = None

    @torch.inference_mode()
    def compute_similarity(self, gt_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> float:
        """
        Calculates Cosine Similarity between Ground Truth and Prediction.
        Expects Tensors in range [-1, 1] or [0, 1]. Returns score in range [0, 1].
        """
        if self.model is None:
            return 0.0

        # Ensure range is [-1, 1]
        gt_in = (gt_tensor * 2.0) - 1.0 if gt_tensor.min() >= 0.0 else gt_tensor
        pred_in = (pred_tensor * 2.0) - 1.0 if pred_tensor.min() >= 0.0 else pred_tensor

        # Resize to 160x160 expected by InceptionResnetV1
        gt_160 = F.interpolate(gt_in, size=(160, 160), mode='bilinear', align_corners=False)
        pred_160 = F.interpolate(pred_in, size=(160, 160), mode='bilinear', align_corners=False)

        emb_gt = self.model(gt_160)
        emb_pred = self.model(pred_160)

        cos_sim = F.cosine_similarity(emb_gt, emb_pred, dim=1)
        return float(cos_sim.item())


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


def load_model(checkpoint_path: Path, device: torch.device) -> LGNetGenerator:
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


def load_tensor_from_file(img_path: Path, mode: str = "RGB", target_size=(256, 256)):
    """Loads image/mask/heatmap file and converts to torch tensor."""
    pil_img = Image.open(img_path).convert(mode).resize(target_size)
    arr_np = np.array(pil_img)

    if mode == "RGB":
        tensor = transforms.ToTensor()(pil_img).unsqueeze(0)  # [1, 3, H, W]
    else:
        tensor = (torch.from_numpy(arr_np).float() / 255.0).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    return arr_np, tensor


@torch.inference_mode()
def run_dataset_inference(
    model: LGNetGenerator,
    image_path: Path,
    mask_path: Path,
    heatmap_path: Path,
    device: torch.device,
    arcface_net: ArcFaceIdentityExtractor = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Loads pre-computed dataset files, runs model, and evaluates ArcFace Identity similarity."""
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

    # ArcFace Identity Metric Calculation
    arcface_sim = 0.0
    if arcface_net is not None:
        arcface_sim = arcface_net.compute_similarity(norm_img, raw_output)

    gen_tensor = (raw_output.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    gen_np = (gen_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    masked_viz_tensor = (masked_img.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    masked_viz_np = (masked_viz_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    blended_np = apply_seamless_blend(orig_np, gen_np, mask_np)
    return orig_np, masked_viz_np, blended_np, mask_np, arcface_sim


def main():
    parser = argparse.ArgumentParser(description="Dataset Inference Pipeline with ArcFace Identity Metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/morphai_epoch_05.pt", help="Path to checkpoint")
    parser.add_argument("--image_id", type=str, default="all", help="Dataset Image ID or 'all' for full test dataset")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split (val or test)")
    parser.add_argument(
        "--variant",
        type=str,
        default="semantic_mouth",
        choices=["all", "all_semantic"] + ALL_VARIANTS,
        help="Select mask variant or 'all'/'all_semantic'",
    )
    parser.add_argument("--out_dir", type=str, default="samples/inference_results", help="Directory to save predictions")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = BASE_DIR / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    model = load_model(checkpoint_path, device)
    
    # Initialize ArcFace Identity Extractor
    arcface_net = ArcFaceIdentityExtractor(device) if HAS_ARCFACE else None
    if not HAS_ARCFACE:
        print(" Warning: 'facenet-pytorch' not found. ArcFace metric computation will be skipped.")
        print("   To enable ArcFace identity similarity, run: pip install facenet-pytorch")

    processed_dir = BASE_DIR / "data" / "processed"
    img_dir = processed_dir / args.split
    mask_dir = processed_dir / "masks" / args.split
    heatmap_dir = processed_dir / "heatmaps" / args.split

    if args.variant == "all":
        variants_to_run = ALL_VARIANTS
    elif args.variant == "all_semantic":
        variants_to_run = SEMANTIC_VARIANTS
    else:
        variants_to_run = [args.variant]

    # --- Mode 1: Batch Dataset Inference ---
    if args.image_id.lower() == "all":
        valid_exts = {".jpg", ".png", ".jpeg"}
        image_files = sorted([f for f in img_dir.glob("*") if f.suffix.lower() in valid_exts])

        if not image_files:
            print(f" Error: No image files found in '{img_dir}'.")
            return

        out_base = BASE_DIR / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
        gt_out = out_base / "gt"
        pred_out = out_base / "pred"
        mask_out = out_base / "mask"
        grid_out = out_base / "grid"

        gt_out.mkdir(parents=True, exist_ok=True)
        pred_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        grid_out.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning BATCH inference on {len(image_files)} images from [{args.split.upper()}] split...")

        saved_count = 0
        arcface_tracker: Dict[str, List[float]] = {v: [] for v in variants_to_run}

        for img_path in tqdm(image_files, desc="Generating Predictions"):
            img_id = img_path.stem

            for var in variants_to_run:
                mask_path = mask_dir / f"{img_id}_{var}.png"
                heatmap_path = heatmap_dir / f"{img_id}_{var}.png"

                if not mask_path.exists() or not heatmap_path.exists():
                    continue

                orig, masked_viz, reconstructed, mask_np, arcface_sim = run_dataset_inference(
                    model, img_path, mask_path, heatmap_path, device, arcface_net
                )

                if HAS_ARCFACE:
                    arcface_tracker[var].append(arcface_sim)

                save_filename = f"{img_id}_{var}.png"
                Image.fromarray(orig).save(gt_out / save_filename)
                Image.fromarray(reconstructed).save(pred_out / save_filename)
                Image.fromarray(mask_np).save(mask_out / save_filename)

                # Save 3-Panel Grid [ Ground Truth | Masked Input | Prediction ]
                grid_3panel = np.hstack([orig, masked_viz, reconstructed])
                Image.fromarray(grid_3panel).save(grid_out / save_filename)

                saved_count += 1

        print(f"\n Completed batch inference! Saved {saved_count} triplets & grids to '{out_base}'.")

        # Display ArcFace Summary Table if available
        if HAS_ARCFACE and any(len(scores) > 0 for scores in arcface_tracker.values()):
            print("\n" + "=" * 50)
            print("       ARCFACE IDENTITY SIMILARITY SUMMARY      ")
            print("=" * 50)
            print(f"| {'Variant':<25} | {'Mean ArcFace ID ↑':<18} |")
            print("|" + "-"*27 + "|" + "-"*20 + "|")
            for var, scores in arcface_tracker.items():
                if scores:
                    mean_score = sum(scores) / len(scores)
                    print(f"| {var:<25} | {mean_score:<18.4f} |")
            print("=" * 50 + "\n")

    # --- Mode 2: Single Image Inference ---
    else:
        img_path = img_dir / f"{args.image_id}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{args.image_id}.png"

        if not img_path.exists():
            print(f" Error: Image ID '{args.image_id}' not found in '{img_dir}'.")
            return

        print(f"\n Running Inference on Image ID '{args.image_id}'...")

        out_base = Path(args.out_dir)

        for var in variants_to_run:
            mask_path = mask_dir / f"{args.image_id}_{var}.png"
            heatmap_path = heatmap_dir / f"{args.image_id}_{var}.png"

            if not mask_path.exists() or not heatmap_path.exists():
                print(f" Skipping variant '{var}': Files missing.")
                continue

            orig, masked_viz, reconstructed, _, arcface_sim = run_dataset_inference(
                model, img_path, mask_path, heatmap_path, device, arcface_net
            )

            # Save individual reconstruction
            pred_path = out_base / f"{args.image_id}_{var}_pred.png"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(reconstructed).save(pred_path)

            # Save 3-Panel Grid [ Original | Masked Input | Reconstructed ]
            grid_3panel = np.hstack([orig, masked_viz, reconstructed])
            grid_path = out_base / f"{args.image_id}_{var}_grid.png"
            Image.fromarray(grid_3panel).save(grid_path)

            print(f" Pipeline Complete for variant: '{var}'")
            print(f"   ├─  Saved prediction: {pred_path}")
            print(f"   └─  Saved 3-Panel Grid [Original | Masked | Reconstructed]: {grid_path}")
            if HAS_ARCFACE:
                print(f"   👤 ArcFace Identity Similarity [{var}]: {arcface_sim:.4f}")


if __name__ == "__main__":
    main()