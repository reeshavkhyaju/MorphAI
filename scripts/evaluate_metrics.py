# import argparse
# import json
# import re
# import sys
# from pathlib import Path
# from typing import Dict, List, Tuple

# import numpy as np
# from PIL import Image
# from scipy.linalg import sqrtm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.models import inception_v3, Inception_V3_Weights
# from tqdm import tqdm

# from skimage.metrics import structural_similarity as compute_ssim_skimage

# # --- Optional Imports ---
# try:
#     import lpips
#     HAS_LPIPS = True
# except ImportError:
#     HAS_LPIPS = False

# try:
#     from facenet_pytorch import InceptionResnetV1
#     HAS_ARCFACE = True
# except ImportError:
#     HAS_ARCFACE = False


# # ==========================================
# # 1. ArcFace Identity Extractor
# # ==========================================

# class ArcFaceIdentityExtractor(nn.Module):
#     """Extracts 512-dim facial identity embeddings for Cosine Similarity."""
#     def __init__(self, device: torch.device):
#         super().__init__()
#         self.device = device
#         if HAS_ARCFACE:
#             self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#             for p in self.model.parameters():
#                 p.requires_grad = False
#         else:
#             self.model = None

#     @torch.inference_mode()
#     def compute_similarity(self, gt: torch.Tensor, pred: torch.Tensor) -> List[float]:
#         """Calculates Cosine Similarity between GT and Pred identity embeddings in range [0, 1]."""
#         if self.model is None:
#             return [0.0] * gt.size(0)

#         # Rescale [0, 1] -> [-1, 1] for backbone
#         gt_in = F.interpolate((gt * 2.0) - 1.0, size=(160, 160), mode='bilinear', align_corners=False)
#         pred_in = F.interpolate((pred * 2.0) - 1.0, size=(160, 160), mode='bilinear', align_corners=False)

#         emb_gt = self.model(gt_in)      # [B, 512]
#         emb_pred = self.model(pred_in)  # [B, 512]

#         # Cosine Similarity
#         cos_sim = F.cosine_similarity(emb_gt, emb_pred, dim=1)
#         return cos_sim.cpu().tolist()


# # ==========================================
# # 2. Pattern Rules & Dataset
# # ==========================================

# PATTERN_RULES = [
#     ("Semantic", "Left Eyebrow",  re.compile(r"(left_eyebrow|semantic_left_eyebrow)", re.IGNORECASE)),
#     ("Semantic", "Right Eyebrow", re.compile(r"(right_eyebrow|semantic_right_eyebrow)", re.IGNORECASE)),
#     ("Semantic", "Left Eye",     re.compile(r"(left_eye|semantic_left_eye)", re.IGNORECASE)),
#     ("Semantic", "Right Eye",    re.compile(r"(right_eye|semantic_right_eye)", re.IGNORECASE)),
#     ("Semantic", "Nose",         re.compile(r"(nose|semantic_nose)", re.IGNORECASE)),
#     ("Semantic", "Mouth",        re.compile(r"(mouth|semantic_mouth)", re.IGNORECASE)),
#     ("Box",      "Box 10%",      re.compile(r"(rect_10|box_10|box10|rect10)", re.IGNORECASE)),
#     ("Box",      "Box 25%",      re.compile(r"(rect_25|box_25|box25|rect25)", re.IGNORECASE)),
#     ("Box",      "Box 50%",      re.compile(r"(rect_50|box_50|box50|rect50)", re.IGNORECASE)),
# ]

# class MultiCategoryDataset(Dataset):
#     def __init__(self, gt_dir: Path, pred_dir: Path, image_size: Tuple[int, int] = (256, 256)):
#         self.image_size = image_size
#         self.samples = []
#         valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

#         gt_files = sorted([f for f in gt_dir.glob("*") if f.suffix.lower() in valid_exts])
#         pred_files = sorted([f for f in pred_dir.glob("*") if f.suffix.lower() in valid_exts])

#         if not gt_files or not pred_files:
#             raise FileNotFoundError("GT or Pred folder is empty.")

#         gt_map = {f.stem: f for f in gt_files}

#         for pf in pred_files:
#             p_stem = pf.stem
#             matched_group, matched_cat = "Uncategorized", "General"
#             clean_stem = p_stem

#             for group, cat_name, pattern in PATTERN_RULES:
#                 if pattern.search(p_stem):
#                     matched_group, matched_cat = group, cat_name
#                     clean_stem = pattern.sub("", clean_stem)
#                     break

#             clean_stem = re.sub(r'(_pred|_gen|_inpainted|_out|_result|_gt|_mask)$', '', clean_stem, flags=re.IGNORECASE).strip('_')
#             matched_gt = gt_map.get(clean_stem) or gt_map.get(p_stem)

#             if not matched_gt:
#                 for gf in gt_files:
#                     if p_stem.startswith(gf.stem) or clean_stem.startswith(gf.stem):
#                         matched_gt = gf
#                         break

#             if matched_gt:
#                 self.samples.append((matched_gt, pf, matched_group, matched_cat))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         gt_path, pred_path, group, cat_name = self.samples[idx]
#         gt_pil = Image.open(gt_path).convert("RGB").resize(self.image_size)
#         pred_pil = Image.open(pred_path).convert("RGB").resize(self.image_size)
#         return transforms.ToTensor()(gt_pil), transforms.ToTensor()(pred_pil), group, cat_name


# # ==========================================
# # 3. Main Evaluation Core
# # ==========================================

# def run_evaluation(gt_dir: Path, pred_dir: Path, device: torch.device, batch_size: int = 16) -> Dict:
#     dataset = MultiCategoryDataset(gt_dir, pred_dir)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

#     arcface_net = ArcFaceIdentityExtractor(device)
#     lpips_fn = lpips.LPIPS(net="vgg").to(device).eval() if HAS_LPIPS else None

#     records = {"ALL": {"psnr": [], "ssim": [], "lpips": [], "arcface": []}, "Semantic": {}, "Box": {}}

#     print(f"📊 Running Evaluation (+ ArcFace Identity Similarity) over {len(dataset)} samples...")

#     for gt_b, pred_b, groups, cat_names in tqdm(dataloader, desc="Evaluating"):
#         gt_b = gt_b.to(device)
#         pred_b = pred_b.to(device)

#         # ArcFace Cosine Similarity
#         arcface_scores = arcface_net.compute_similarity(gt_b, pred_b)

#         # LPIPS & PSNR
#         gt_norm = (gt_b * 2.0) - 1.0
#         pred_norm = (pred_b * 2.0) - 1.0
#         lp_scores = lpips_fn(gt_norm, pred_norm).flatten().cpu().tolist() if lpips_fn else [0.0] * gt_b.size(0)

#         mse = torch.mean((gt_b - pred_b) ** 2, dim=[1, 2, 3]).clamp(min=1e-10)
#         psnr_scores = (10.0 * torch.log10(1.0 / mse)).cpu().tolist()

#         gt_np = (gt_b.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
#         pred_np = (pred_b.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

#         for i in range(gt_b.size(0)):
#             ssim_val = float(compute_ssim_skimage(gt_np[i], pred_np[i], channel_axis=2, data_range=255))
#             grp, cat = groups[i], cat_names[i]

#             records["ALL"]["psnr"].append(psnr_scores[i])
#             records["ALL"]["ssim"].append(ssim_val)
#             records["ALL"]["lpips"].append(lp_scores[i])
#             records["ALL"]["arcface"].append(arcface_scores[i])

#             if grp in ("Semantic", "Box"):
#                 if cat not in records[grp]:
#                     records[grp][cat] = {"psnr": [], "ssim": [], "lpips": [], "arcface": []}
#                 records[grp][cat]["psnr"].append(psnr_scores[i])
#                 records[grp][cat]["ssim"].append(ssim_val)
#                 records[grp][cat]["lpips"].append(lp_scores[i])
#                 records[grp][cat]["arcface"].append(arcface_scores[i])

#     def compile_stats(d):
#         return {
#             "PSNR ↑": float(np.mean(d["psnr"])) if d["psnr"] else 0.0,
#             "SSIM ↑": float(np.mean(d["ssim"])) if d["ssim"] else 0.0,
#             "LPIPS ↓": float(np.mean(d["lpips"])) if d["lpips"] else 0.0,
#             "ArcFace ID ↑": float(np.mean(d["arcface"])) if d["arcface"] else 0.0,
#             "Count": len(d["psnr"])
#         }

#     results = {"ALL": compile_stats(records["ALL"]), "Semantic": {}, "Box": {}}
#     for grp in ("Semantic", "Box"):
#         for cat_name, ddict in records[grp].items():
#             results[grp][cat_name] = compile_stats(ddict)

#     return results


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gt_dir", type=str, default="samples/inference_results/gt")
#     parser.add_argument("--pred_dir", type=str, default="samples/inference_results/pred")
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--output_json", type=str, default="samples/inference_results/metrics.json")
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     results = run_evaluation(Path(args.gt_dir), Path(args.pred_dir), device, args.batch_size)

#     print("\n" + "=" * 80)
#     print("               EVALUATION RESULTS WITH ARCFACE IDENTITY MATCHING              ")
#     print("=" * 80)
#     print(f"| {'Category':<16} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'LPIPS ↓':<8} | {'ArcFace ID ↑':<12} |")
#     print("|" + "-"*18 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*14 + "|")
#     print(f"| {'ALL':<16} | {results['ALL']['PSNR ↑']:<8.4f} | {results['ALL']['SSIM ↑']:<8.4f} | {results['ALL']['LPIPS ↓']:<8.4f} | {results['ALL']['ArcFace ID ↑']:<12.4f} |")
    
#     for grp in ("Semantic", "Box"):
#         for cat, m in results[grp].items():
#             print(f"| {cat:<16} | {m['PSNR ↑']:<8.4f} | {m['SSIM ↑']:<8.4f} | {m['LPIPS ↓']:<8.4f} | {m['ArcFace ID ↑']:<12.4f} |")
#     print("=" * 80 + "\n")

#     with open(args.output_json, "w") as f:
#         json.dump(results, f, indent=4)


# if __name__ == "__main__":
#     main()

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
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
        if HAS_ARCFACE:
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            self.model = None

    @torch.inference_mode()
    def compute_similarity(self, gt: torch.Tensor, pred: torch.Tensor) -> List[float]:
        """Calculates Cosine Similarity between GT and Pred identity embeddings in range [0, 1]."""
        if self.model is None:
            return [0.0] * gt.size(0)

        # Rescale [0, 1] -> [-1, 1] for backbone
        gt_in = F.interpolate((gt * 2.0) - 1.0, size=(160, 160), mode='bilinear', align_corners=False)
        pred_in = F.interpolate((pred * 2.0) - 1.0, size=(160, 160), mode='bilinear', align_corners=False)

        emb_gt = self.model(gt_in)      # [B, 512]
        emb_pred = self.model(pred_in)  # [B, 512]

        # Cosine Similarity
        cos_sim = F.cosine_similarity(emb_gt, emb_pred, dim=1)
        return cos_sim.cpu().tolist()


# ==========================================
# 2. Pattern Rules & Dataset
# ==========================================

PATTERN_RULES = [
    ("Semantic", "Left Eyebrow",  re.compile(r"(left_eyebrow|semantic_left_eyebrow|l_eyebrow|l_brow)", re.IGNORECASE)),
    ("Semantic", "Right Eyebrow", re.compile(r"(right_eyebrow|semantic_right_eyebrow|r_eyebrow|r_brow)", re.IGNORECASE)),
    ("Semantic", "Left Eye",     re.compile(r"(left_eye|semantic_left_eye|l_eye)", re.IGNORECASE)),
    ("Semantic", "Right Eye",    re.compile(r"(right_eye|semantic_right_eye|r_eye)", re.IGNORECASE)),
    ("Semantic", "Nose",         re.compile(r"(nose|semantic_nose)", re.IGNORECASE)),
    ("Semantic", "Mouth",        re.compile(r"(mouth|semantic_mouth|lip|lips)", re.IGNORECASE)),
    ("Box",      "Box 10%",      re.compile(r"(rect_10|box_10|box10|rect10|10_pct|10pct|10%)", re.IGNORECASE)),
    ("Box",      "Box 25%",      re.compile(r"(rect_25|box_25|box25|rect25|25_pct|25pct|25%)", re.IGNORECASE)),
    ("Box",      "Box 50%",      re.compile(r"(rect_50|box_50|box50|rect50|50_pct|50pct|50%)", re.IGNORECASE)),
]


def extract_id(p: Path) -> str:
    s = p.stem.lower()
    s = re.sub(r'(_pred|_gen|_inpainted|_out|_result|_gt|_mask|_lama)', '', s)
    digits = re.findall(r'\d+', s)
    return digits[0] if digits else s.strip('_')


class MultiCategoryDataset(Dataset):
    def __init__(self, gt_dir: Path, pred_dir: Path, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.samples = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        # Use rglob to recursively discover images in nested folders
        gt_files = sorted([f for f in gt_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts])
        pred_files = sorted([f for f in pred_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts])

        if not gt_files or not pred_files:
            raise FileNotFoundError("GT or Pred folder is empty.")

        gt_stem_map = {f.stem.lower(): f for f in gt_files}
        gt_id_map = {extract_id(f): f for f in gt_files if extract_id(f)}

        for pf in pred_files:
            p_stem = pf.stem
            full_path_str = pf.as_posix()
            matched_group, matched_cat = "Uncategorized", "General"

            # Check full path string to capture subfolder names like 'Box 10%' or 'box_25'
            for group, cat_name, pattern in PATTERN_RULES:
                if pattern.search(full_path_str):
                    matched_group, matched_cat = group, cat_name
                    break

            p_id = extract_id(pf)
            matched_gt = gt_stem_map.get(p_stem.lower()) or gt_id_map.get(p_id)

            if not matched_gt:
                for gf in gt_files:
                    if p_stem.lower().startswith(gf.stem.lower()) or p_id == extract_id(gf):
                        matched_gt = gf
                        break

            if matched_gt:
                self.samples.append((matched_gt, pf, matched_group, matched_cat))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt_path, pred_path, group, cat_name = self.samples[idx]
        gt_pil = Image.open(gt_path).convert("RGB").resize(self.image_size)
        pred_pil = Image.open(pred_path).convert("RGB").resize(self.image_size)
        return transforms.ToTensor()(gt_pil), transforms.ToTensor()(pred_pil), group, cat_name


# ==========================================
# 3. Main Evaluation Core
# ==========================================

def run_evaluation(gt_dir: Path, pred_dir: Path, device: torch.device, batch_size: int = 16) -> Dict:
    dataset = MultiCategoryDataset(gt_dir, pred_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    arcface_net = ArcFaceIdentityExtractor(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval() if HAS_LPIPS else None

    records = {"ALL": {"psnr": [], "ssim": [], "lpips": [], "arcface": []}, "Semantic": {}, "Box": {}}

    print(f"📊 Running Evaluation (+ ArcFace Identity Similarity) over {len(dataset)} samples...")

    for gt_b, pred_b, groups, cat_names in tqdm(dataloader, desc="Evaluating"):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="samples/inference_results/gt")
    parser.add_argument("--pred_dir", type=str, default="samples/inference_results/pred")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, default="samples/inference_results/metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_evaluation(Path(args.gt_dir), Path(args.pred_dir), device, args.batch_size)

    print("\n" + "=" * 80)
    print("               EVALUATION RESULTS WITH ARCFACE IDENTITY MATCHING              ")
    print("=" * 80)
    print(f"| {'Category':<16} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'LPIPS ↓':<8} | {'ArcFace ID ↑':<12} |")
    print("|" + "-"*18 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*14 + "|")
    print(f"| {'ALL':<16} | {results['ALL']['PSNR ↑']:<8.4f} | {results['ALL']['SSIM ↑']:<8.4f} | {results['ALL']['LPIPS ↓']:<8.4f} | {results['ALL']['ArcFace ID ↑']:<12.4f} |")
    
    for grp in ("Semantic", "Box"):
        if results[grp]:
            print("|" + "-"*18 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*14 + "|")
            for cat, m in results[grp].items():
                print(f"| {cat:<16} | {m['PSNR ↑']:<8.4f} | {m['SSIM ↑']:<8.4f} | {m['LPIPS ↓']:<8.4f} | {m['ArcFace ID ↑']:<12.4f} |")
    print("=" * 80 + "\n")

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()