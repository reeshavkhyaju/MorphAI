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

# --- Dynamic Path Resolution (Works in project root OR inside scripts/) ---
BASE_DIR = Path(__file__).resolve().parent
if (BASE_DIR / "src").exists():
    pass
elif (BASE_DIR.parent / "src").exists():
    BASE_DIR = BASE_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models import LGNetGenerator
from src.landmarks import LandmarkExtractor
from src.masks import MaskGenerator
from src.conditioning import ConditioningGenerator

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


SEMANTIC_VARIANTS = [
    "semantic_left_eye",
    "semantic_right_eye",
    "semantic_left_eyebrow",
    "semantic_right_eyebrow",
    "semantic_nose",
    "semantic_mouth",
]

IRREGULAR_VARIANTS = [
    "irregular_shape",
    "irregular_shape_left",
    "irregular_shape_right",
]

ALL_VARIANTS = SEMANTIC_VARIANTS + IRREGULAR_VARIANTS


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
    """Seamless Poisson blending of generated area onto base frame."""
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


def generate_irregular_mask(landmarks, location="center", img_size=(256, 256)) -> np.ndarray:
    """Generates a small organic irregular polygon shape relative to face landmarks."""
    h, w = img_size
    mask = np.zeros((h, w), dtype=np.uint8)

    # Fallback default position
    cx, cy = w // 2, int(h * 0.22)

    # Calculate exact position if landmarks exist
    if landmarks is not None and len(landmarks) >= 27:
        try:
            eyebrows = landmarks[17:27]
            eb_x_min = np.min(eyebrows[:, 0])
            eb_x_max = np.max(eyebrows[:, 0])
            eb_x_mid = np.mean(eyebrows[:, 0])
            eb_y_mean = np.mean(eyebrows[:, 1])

            face_width = eb_x_max - eb_x_min
            target_y = eb_y_mean - (face_width * 0.28)  # Directly above eyebrow line

            if location == "left":
                target_x = eb_x_min + (face_width * 0.20)
            elif location == "right":
                target_x = eb_x_max - (face_width * 0.20)
            else:  # Center
                target_x = eb_x_mid

            cx = int(np.clip(target_x, 15, w - 15))
            cy = int(np.clip(target_y, 15, h - 15))
        except Exception:
            pass

    # Generate an organic irregular polygon (blob) around target center
    num_points = np.random.randint(7, 11)
    base_radius = np.random.uniform(14, 20)  # Small natural patch (~30-40px diameter)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))

    pts = []
    for angle in angles:
        r = base_radius * np.random.uniform(0.75, 1.25)
        px = int(cx + r * np.cos(angle))
        py = int(cy + r * np.sin(angle))
        pts.append([px, py])

    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)

    return mask


@torch.inference_mode()
def process_custom_image(
    model: LGNetGenerator,
    image_path: Path,
    variant_name: str,
    output_dir: Path,
    device: torch.device,
    extractor: LandmarkExtractor,
    mask_gen: MaskGenerator,
    cond_gen: ConditioningGenerator,
    arcface_net: ArcFaceIdentityExtractor = None
) -> float:
    """Processes custom photo following dataset pipeline steps and calculates ArcFace score."""
    
    # 1. Load Original High-Res Image
    orig_highres_pil = Image.open(image_path).convert("RGB")
    orig_highres_np = np.array(orig_highres_pil)
    orig_h, orig_w = orig_highres_np.shape[:2]

    # 2. Resize to 256x256 for Model Processing
    resample_method = getattr(Image, "Resampling", Image).LANCZOS
    pil_256 = orig_highres_pil.resize((256, 256), resample_method)
    orig_256_np = np.array(pil_256)

    # Temporary file for landmark extraction on 256x256 image
    temp_256_path = output_dir / f"_temp_256_{image_path.name}"
    pil_256.save(temp_256_path)

    # 3. Extract Landmarks for this photo at 256x256
    landmarks = extractor.extract_landmarks(temp_256_path)
    regions = extractor.get_semantic_regions(landmarks)

    if temp_256_path.exists():
        temp_256_path.unlink()

    # 4. Generate Mask (Semantic or Irregular Blob)
    if variant_name.startswith("semantic_") and regions is not None:
        region_key = variant_name.replace("semantic_", "")
        region_pts = regions.get(region_key, None)
        mask_256_np = mask_gen.generate_semantic_mask(region_pts)
    elif "irregular" in variant_name:
        loc = "center"
        if "left" in variant_name:
            loc = "left"
        elif "right" in variant_name:
            loc = "right"
        mask_256_np = generate_irregular_mask(landmarks, location=loc)
    else:
        mask_256_np = generate_irregular_mask(landmarks, location="center")

    # 5. Generate Conditioning Heatmap
    heatmap_256_np = cond_gen.generate_heatmap(landmarks, mask=mask_256_np)

    # 6. Save Intermediate Mask & Heatmap Files
    stem = image_path.stem
    mask_file = output_dir / f"{stem}_{variant_name}_mask.png"
    heatmap_file = output_dir / f"{stem}_{variant_name}_heatmap.png"
    
    Image.fromarray(mask_256_np).save(mask_file)
    Image.fromarray(heatmap_256_np).save(heatmap_file)

    # 7. Prepare Tensors normalized to [-1, 1]
    img_tensor = transforms.ToTensor()(pil_256).unsqueeze(0).to(device)
    norm_img = (img_tensor * 2.0) - 1.0
    
    mask_tensor = (torch.from_numpy(mask_256_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    heatmap_tensor = (torch.from_numpy(heatmap_256_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    # Apply mask (-1 where masked out)
    masked_img = norm_img * (1.0 - mask_tensor) + (-1.0) * mask_tensor

    # 8. Concatenate into 5-channel Input Tensor [1, 5, 256, 256]
    input_5ch = torch.cat([masked_img, mask_tensor, heatmap_tensor], dim=1).float()

    # 9. Model Generator Pass
    raw_output = model(input_5ch)

    # 10. ArcFace Identity Metric Calculation
    arcface_sim = 0.0
    if arcface_net is not None:
        arcface_sim = arcface_net.compute_similarity(norm_img, raw_output)

    # 11. Post-process Reconstruction Output
    gen_tensor = (raw_output.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    gen_256_np = (gen_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    masked_viz_tensor = (masked_img.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    masked_viz_256_np = (masked_viz_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    # Seamless Poisson blending on 256x256 frame
    reconstructed_256_np = apply_seamless_blend(orig_256_np, gen_256_np, mask_256_np)

    reconstructed_256_file = output_dir / f"{stem}_{variant_name}_256x256.png"
    Image.fromarray(reconstructed_256_np).save(reconstructed_256_file)

    # 12. Scale up generated feature back onto high-res original frame if needed
    if (orig_h, orig_w) != (256, 256):
        gen_highres_np = cv2.resize(gen_256_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        mask_highres_np = cv2.resize(mask_256_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        highres_blended_np = apply_seamless_blend(orig_highres_np, gen_highres_np, mask_highres_np)
        
        highres_file = output_dir / f"{stem}_{variant_name}_highres.png"
        Image.fromarray(highres_blended_np).save(highres_file)
        print(f"   ├─ 🔍 Scaled reconstruction to Original Resolution ({orig_w}x{orig_h}): {highres_file.name}")

    # 13. Save 3-Panel Grid [ Original | Masked Input | Reconstructed ]
    grid_3panel = np.hstack([orig_256_np, masked_viz_256_np, reconstructed_256_np])
    grid_file = output_dir / f"{stem}_{variant_name}_grid.png"
    Image.fromarray(grid_3panel).save(grid_file)

    print(f"\n✅ Pipeline Complete for variant: '{variant_name}'")
    if HAS_ARCFACE:
        print(f"   ├─ 👤 ArcFace Identity Similarity: {arcface_sim:.4f} (1.0 = Perfect Match)")
    print(f"   ├─ 🎭 Mask: {mask_file.name}")
    print(f"   ├─ 🔥 Heatmap: {heatmap_file.name}")
    print(f"   ├─ ✨ Reconstructed Output: {reconstructed_256_file.name}")
    print(f"   └─ 🖼️ 3-Panel Grid [Original | Masked | Reconstructed]: {grid_file.name}")

    return arcface_sim


def main():
    parser = argparse.ArgumentParser(description="Custom Photo Inference Pipeline with ArcFace Identity Metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/morphai_epoch_05.pt", help="Path to model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to raw custom input image")
    parser.add_argument(
        "--variant",
        type=str,
        default="irregular_shape",
        help="Select variant (e.g. 'irregular_shape', 'irregular_shape_left', 'irregular_shape_right', 'semantic_nose', 'all')",
    )
    parser.add_argument(
        "--out_dir", "--out",
        dest="out_dir",
        type=str,
        default="custom_outputs",
        help="Output directory folder for generated files",
    )

    args = parser.parse_args()
    img_path = Path(args.image_path)

    if not img_path.exists():
        print(f"❌ Error: Image path '{args.image_path}' does not exist.")
        return

    # Safe directory creation
    out_path = Path(args.out_dir)
    if out_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
        output_dir = out_path.parent if out_path.parent != Path(".") else Path("custom_outputs")
    else:
        output_dir = out_path

    if output_dir.exists() and output_dir.is_file():
        output_dir.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Initialize ArcFace Identity Extractor
    arcface_net = ArcFaceIdentityExtractor(device) if HAS_ARCFACE else None
    if not HAS_ARCFACE:
        print("⚠️ Warning: 'facenet-pytorch' not found. ArcFace metric computation will be skipped.")
        print("   To enable ArcFace identity similarity, run: pip install facenet-pytorch")

    print("\n🔍 Initializing landmark extractor & mask generators...")
    extractor = LandmarkExtractor()
    mask_gen = MaskGenerator()
    cond_gen = ConditioningGenerator()

    if args.variant == "all":
        variants_to_run = ALL_VARIANTS
    elif args.variant == "all_semantic":
        variants_to_run = SEMANTIC_VARIANTS
    else:
        variants_to_run = [args.variant]

    scores_tracker: Dict[str, float] = {}

    for var in variants_to_run:
        score = process_custom_image(
            model, img_path, var, output_dir, device, extractor, mask_gen, cond_gen, arcface_net
        )
        scores_tracker[var] = score

    extractor.close()

    # Summary table if multiple variants ran
    if HAS_ARCFACE and len(variants_to_run) > 1:
        print("\n" + "=" * 50)
        print("       ARCFACE IDENTITY SIMILARITY SUMMARY      ")
        print("=" * 50)
        print(f"| {'Variant':<25} | {'ArcFace ID Score ↑':<18} |")
        print("|" + "-"*27 + "|" + "-"*20 + "|")
        for var, score in scores_tracker.items():
            print(f"| {var:<25} | {score:<18.4f} |")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()