import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Setup project root path
FILE_DIR = Path(__file__).resolve().parent
BASE_DIR = FILE_DIR.parent if (FILE_DIR.parent / "src").exists() else FILE_DIR
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models import LGNetGenerator


MASK_TYPES = [
    "semantic_left_eyebrow",
    "semantic_right_eyebrow",
    "semantic_left_eye",
    "semantic_right_eye",
    "semantic_nose",
    "semantic_mouth",
    "rect_10",
    "rect_25",
    "rect_50",
]


def load_generator(checkpoint_path: Path, device: torch.device):
    model = LGNetGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("generator_state_dict") or checkpoint.get("generator") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Full Batch Inference on device: {device}")

    # Paths
    ckpt_path = BASE_DIR / "checkpoints" / "morphai_epoch_05.pt"
    img_dir = BASE_DIR / "data" / "processed" / "test"
    mask_dir = BASE_DIR / "data" / "processed" / "masks" / "test"
    hm_dir = BASE_DIR / "data" / "processed" / "heatmaps" / "test"

    gt_out_dir = BASE_DIR / "samples" / "inference_results" / "gt"
    pred_out_dir = BASE_DIR / "samples" / "inference_results" / "pred"

    gt_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_dir.mkdir(parents=True, exist_ok=True)

    # Load Model
    model = load_generator(ckpt_path, device)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    gt_images = sorted([f for f in img_dir.glob("*") if f.suffix.lower() in valid_exts])

    print(f"📸 Found {len(gt_images)} ground truth images in test set.")
    print(f"🎭 Generating predictions across {len(MASK_TYPES)} mask categories...\n")

    for img_path in tqdm(gt_images, desc="Generating Inferences"):
        img_id = img_path.stem

        # Save GT image to gt folder if missing
        gt_save_path = gt_out_dir / f"{img_id}.png"
        if not gt_save_path.exists():
            pil_gt = Image.open(img_path).convert("RGB").resize((256, 256))
            pil_gt.save(gt_save_path)
        else:
            pil_gt = Image.open(gt_save_path).convert("RGB")

        gt_tensor = transforms.ToTensor()(pil_gt).unsqueeze(0).to(device)  # [1, 3, 256, 256] in [0, 1]
        norm_gt = (gt_tensor * 2.0) - 1.0                                   # [-1, 1]

        # Generate output for each mask type
        for mask_key in MASK_TYPES:
            pred_save_path = pred_out_dir / f"{img_id}_{mask_key}.png"
            if pred_save_path.exists():
                continue

            mask_path = mask_dir / f"{img_id}_{mask_key}.png"
            hm_path = hm_dir / f"{img_id}_{mask_key}.png"

            if not mask_path.exists() or not hm_path.exists():
                continue

            # Load Mask & Heatmap
            pil_mask = Image.open(mask_path).convert("L").resize((256, 256))
            mask_t = (transforms.ToTensor()(pil_mask) > 0.5).float().unsqueeze(0).to(device)

            pil_hm = Image.open(hm_path).convert("L").resize((256, 256))
            hm_t = transforms.ToTensor()(pil_hm).unsqueeze(0).to(device)

            # Build 5-channel model input: [Masked_RGB (3ch), Mask (1ch), Heatmap (1ch)]
            masked_img = norm_gt * (1.0 - mask_t) + (-1.0) * mask_t
            input_5ch = torch.cat([masked_img, mask_t, hm_t], dim=1)

            with torch.inference_mode():
                out_tensor = model(input_5ch).clamp(-1.0, 1.0)
                pred_t = ((out_tensor + 1.0) / 2.0).squeeze(0).cpu()

            pred_pil = transforms.ToPILImage()(pred_t)
            pred_pil.save(pred_save_path)

    print(f"\n✅ All inference predictions saved to: {pred_out_dir.resolve()}\n")


if __name__ == "__main__":
    main()