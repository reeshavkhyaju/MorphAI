import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from skimage.metrics import structural_similarity as compute_ssim_skimage

# --- Optional Imports ---
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from facenet_pytorch import InceptionResnetV1
    HAS_ARCFACE = True
except ImportError:
    HAS_ARCFACE = False


# ==========================================
# 1. ArcFace Identity Extractor
# ==========================================

class ArcFaceIdentityExtractor(nn.Module):
    """Extracts 512-dim facial identity embeddings for Cosine Similarity."""
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.model = None
        if HAS_ARCFACE:
            try:
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
                for p in self.model.parameters():
                    p.requires_grad = False
            except Exception:
                self.model = None

    @torch.inference_mode()
    def compute_similarity(self, gt: torch.Tensor, pred: torch.Tensor) -> List[float]:
        if self.model is None:
            return [0.0] * gt.size(0)

        gt_in = F.interpolate((gt * 2.0) - 1.0, size=(160, 160), mode='bilinear', align_corners=False)
        pred_in = F.interpolate((pred * 2.0) - 1.0, size=(160, 160), mode='bilinear', align_corners=False)

        emb_gt = self.model(gt_in)      # [B, 512]
        emb_pred = self.model(pred_in)  # [B, 512]

        cos_sim = F.cosine_similarity(emb_gt, emb_pred, dim=1)
        return cos_sim.cpu().tolist()


# ==========================================
# 2. Category Rules & Dataset
# ==========================================

PATTERN_RULES = [
    ("Semantic", "Left Eyebrow",  re.compile(r"(left_eyebrow|l_eyebrow|l_brow)", re.IGNORECASE)),
    ("Semantic", "Right Eyebrow", re.compile(r"(right_eyebrow|r_eyebrow|r_brow)", re.IGNORECASE)),
    ("Semantic", "Left Eye",     re.compile(r"(left_eye|l_eye)", re.IGNORECASE)),
    ("Semantic", "Right Eye",    re.compile(r"(right_eye|r_eye)", re.IGNORECASE)),
    ("Semantic", "Nose",         re.compile(r"(nose)", re.IGNORECASE)),
    ("Semantic", "Mouth",        re.compile(r"(mouth|lip|lips)", re.IGNORECASE)),
    ("Box",      "Box 10%",      re.compile(r"(rect_10|box_10|box10|rect10|10_pct|10pct|10%)", re.IGNORECASE)),
    ("Box",      "Box 25%",      re.compile(r"(rect_25|box_25|box25|rect25|25_pct|25pct|25%)", re.IGNORECASE)),
    ("Box",      "Box 50%",      re.compile(r"(rect_50|box_50|box50|rect50|50_pct|50pct|50%)", re.IGNORECASE)),
]


def extract_id(path_obj: Path) -> str:
    s = path_obj.stem.lower()
    s = re.sub(r'(_lama|_gt|_pred|_gen|_inpainted|_out|_result|_mask|_cropped)', '', s)
    digits = re.findall(r'\d+', s)
    return digits[0] if digits else s.strip('_')


class LaMaEvaluationDataset(Dataset):
    """Dataset pairing Ground Truth test images with predicted inpainting outputs."""
    def __init__(self, gt_dir: Path, pred_dir: Path, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.samples = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        gt_files = [f for f in gt_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]
        pred_files = [f for f in pred_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]

        if not gt_files:
            raise FileNotFoundError(f"No ground truth images found in '{gt_dir}'.")
        if not pred_files:
            raise FileNotFoundError(f"No prediction images found in '{pred_dir}'.")

        # Index Ground Truth Files by Stem and Numerical ID
        gt_stem_map = {f.stem.lower(): f for f in gt_files}
        gt_id_map = {extract_id(f): f for f in gt_files if extract_id(f)}

        for pf in pred_files:
            p_stem = pf.stem.lower()
            p_id = extract_id(pf)

            matched_gt = gt_stem_map.get(p_stem) or gt_id_map.get(p_id)

            if matched_gt:
                full_path_str = pf.as_posix()
                matched_group, matched_cat = "Uncategorized", "General"

                for group, cat_name, pattern in PATTERN_RULES:
                    if pattern.search(full_path_str):
                        matched_group, matched_cat = group, cat_name
                        break

                self.samples.append((matched_gt, pf, matched_group, matched_cat))

        print(f"✅ Successfully paired {len(self.samples)} sample pairs for evaluation.")

        if len(self.samples) == 0:
            raise RuntimeError("Failed to pair files. Ensure ground truth and prediction IDs match.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt_path, pred_path, group, cat_name = self.samples[idx]
        gt_pil = Image.open(gt_path).convert("RGB").resize(self.image_size)
        pred_pil = Image.open(pred_path).convert("RGB").resize(self.image_size)
        return transforms.ToTensor()(gt_pil), transforms.ToTensor()(pred_pil), group, cat_name


# ==========================================
# 3. LaMa Evaluation Core
# ==========================================

def run_evaluation(gt_dir: Path, pred_dir: Path, device: torch.device, batch_size: int = 32) -> Dict:
    dataset = LaMaEvaluationDataset(gt_dir, pred_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    arcface_net = ArcFaceIdentityExtractor(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval() if HAS_LPIPS else None

    records = {"ALL": {"psnr": [], "ssim": [], "lpips": [], "arcface": []}, "Semantic": {}, "Box": {}}

    print(f"📊 Running Evaluation across {len(dataset)} samples on device '{device}'...")

    for gt_b, pred_b, groups, cat_names in tqdm(dataloader, desc="Evaluating Batches"):
        gt_b = gt_b.to(device)
        pred_b = pred_b.to(device)

        # ArcFace Cosine Similarity
        arcface_scores = arcface_net.compute_similarity(gt_b, pred_b)

        # LPIPS & PSNR
        gt_norm = (gt_b * 2.0) - 1.0
        pred_norm = (pred_b * 2.0) - 1.0
        lp_scores = lpips_fn(gt_norm, pred_norm).flatten().cpu().tolist() if lpips_fn else [0.0] * gt_b.size(0)

        mse = torch.mean((gt_b - pred_b) ** 2, dim=[1, 2, 3]).clamp(min=1e-10)
        psnr_scores = (10.0 * torch.log10(1.0 / mse)).cpu().tolist()

        gt_np = (gt_b.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pred_np = (pred_b.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

        for i in range(gt_b.size(0)):
            ssim_val = float(compute_ssim_skimage(gt_np[i], pred_np[i], channel_axis=2, data_range=255))
            grp, cat = groups[i], cat_names[i]

            # Collect into "ALL" exclusively for Semantic categories (6 categories x 2,800 = 16,800 total count)
            if grp == "Semantic":
                records["ALL"]["psnr"].append(psnr_scores[i])
                records["ALL"]["ssim"].append(ssim_val)
                records["ALL"]["lpips"].append(lp_scores[i])
                records["ALL"]["arcface"].append(arcface_scores[i])

            if grp in ("Semantic", "Box"):
                if cat not in records[grp]:
                    records[grp][cat] = {"psnr": [], "ssim": [], "lpips": [], "arcface": []}
                records[grp][cat]["psnr"].append(psnr_scores[i])
                records[grp][cat]["ssim"].append(ssim_val)
                records[grp][cat]["lpips"].append(lp_scores[i])
                records[grp][cat]["arcface"].append(arcface_scores[i])

    def compile_stats(d):
        return {
            "PSNR ↑": float(np.mean(d["psnr"])) if d["psnr"] else 0.0,
            "SSIM ↑": float(np.mean(d["ssim"])) if d["ssim"] else 0.0,
            "LPIPS ↓": float(np.mean(d["lpips"])) if d["lpips"] else 0.0,
            "ArcFace ID ↑": float(np.mean(d["arcface"])) if d["arcface"] else 0.0,
            "Count": len(d["psnr"])
        }

    results = {"ALL": compile_stats(records["ALL"]), "Semantic": {}, "Box": {}}
    for grp in ("Semantic", "Box"):
        for cat_name, ddict in records[grp].items():
            results[grp][cat_name] = compile_stats(ddict)

    return results


def main():
    parser = argparse.ArgumentParser(description="LaMa Full Metric Evaluation Engine")
    parser.add_argument("--gt_dir", type=str, default="data/processed/test")
    parser.add_argument("--pred_dir", type=str, default="samples/lama_comparison")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_json", type=str, default="samples/lama_comparison/lama_full_facial_metrics.json")
    args = parser.parse_args()

    gt_path = Path(args.gt_dir)
    pred_path = Path(args.pred_dir)

    print(f"📁 Ground Truth Directory: '{gt_path}'")
    print(f"📁 Prediction Directory:   '{pred_path}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_evaluation(gt_path, pred_path, device, args.batch_size)

    print("\n" + "=" * 80)
    print("           LAMA EVALUATION RESULTS WITH ARCFACE IDENTITY MATCHING        ")
    print("=" * 80)
    print(f"| {'Category':<16} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'LPIPS ↓':<8} | {'ArcFace ID ↑':<12} | {'Count':<5} |")
    print("|" + "-"*18 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*14 + "|" + "-"*7 + "|")
    print(f"| {'ALL':<16} | {results['ALL']['PSNR ↑']:<8.4f} | {results['ALL']['SSIM ↑']:<8.4f} | {results['ALL']['LPIPS ↓']:<8.4f} | {results['ALL']['ArcFace ID ↑']:<12.4f} | {results['ALL']['Count']:<5} |")

    for grp in ("Semantic", "Box"):
        if results[grp]:
            print("|" + "-"*18 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*14 + "|" + "-"*7 + "|")
            for cat, m in results[grp].items():
                print(f"| {cat:<16} | {m['PSNR ↑']:<8.4f} | {m['SSIM ↑']:<8.4f} | {m['LPIPS ↓']:<8.4f} | {m['ArcFace ID ↑']:<12.4f} | {m['Count']:<5} |")
    print("=" * 80 + "\n")

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Evaluation complete. Full metrics saved to: '{out_json}'")


if __name__ == "__main__":
    main()