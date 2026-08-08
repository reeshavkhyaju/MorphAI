import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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

        emb_gt = self.model(gt_in)
        emb_pred = self.model(pred_in)

        cos_sim = F.cosine_similarity(emb_gt, emb_pred, dim=1)
        return cos_sim.cpu().tolist()


# ==========================================
# 2. Pattern Matching Rules & Flexible Dataset
# ==========================================

PATTERN_RULES = [
    ("Semantic", "Left Eyebrow",  re.compile(r"(left_eyebrow|semantic_left_eyebrow)", re.IGNORECASE)),
    ("Semantic", "Right Eyebrow", re.compile(r"(right_eyebrow|semantic_right_eyebrow)", re.IGNORECASE)),
    ("Semantic", "Left Eye",     re.compile(r"(left_eye|semantic_left_eye)", re.IGNORECASE)),
    ("Semantic", "Right Eye",    re.compile(r"(right_eye|semantic_right_eye)", re.IGNORECASE)),
    ("Semantic", "Nose",         re.compile(r"(nose|semantic_nose)", re.IGNORECASE)),
    ("Semantic", "Mouth",        re.compile(r"(mouth|semantic_mouth)", re.IGNORECASE)),
    ("Box",      "Box 10%",      re.compile(r"(rect_10|box_10|box10|rect10)", re.IGNORECASE)),
    ("Box",      "Box 25%",      re.compile(r"(rect_25|box_25|box25|rect25)", re.IGNORECASE)),
    ("Box",      "Box 50%",      re.compile(r"(rect_50|box_50|box50|rect50)", re.IGNORECASE)),
]


class FlexibleMultiModelDataset(Dataset):
    """Dataset supporting flat, pred, or subfolder structures inside samples/inference_results."""
    def __init__(self, base_dir: Path, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.samples = []
        self.model_names = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        gt_dir, model_dirs = self._discover_directories(base_dir, valid_exts)

        if gt_dir and model_dirs:
            self._load_from_subfolders(gt_dir, model_dirs, valid_exts)
        else:
            self._load_from_flat_folder(base_dir, valid_exts)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"❌ No matching image files found inside: '{base_dir}'.\n"
                f"Please ensure ground truth images are in 'gt/' and predictions are in model subfolders or 'pred/'."
            )

    def _discover_directories(self, base_dir: Path, valid_exts: set) -> Tuple[Path, Dict[str, Path]]:
        gt_dir = None
        for candidate in ["gt", "ground_truth", "target", "gts"]:
            if (base_dir / candidate).is_dir():
                gt_dir = base_dir / candidate
                break

        model_dirs = {}
        ignore_dirs = {"gt", "ground_truth", "target", "gts", "masks", "mask", "metrics", "__pycache__", ".git"}

        if gt_dir:
            for sub in base_dir.iterdir():
                if sub.is_dir() and sub.name.lower() not in ignore_dirs:
                    if sub.name.lower() in ("pred", "preds", "predictions", "results"):
                        # Check if 'pred' contains subfolders for each model
                        sub_models = [d for d in sub.iterdir() if d.is_dir() and d.name.lower() not in ignore_dirs]
                        if sub_models:
                            for m_dir in sub_models:
                                if any(f.suffix.lower() in valid_exts for f in m_dir.glob("*")):
                                    model_dirs[m_dir.name] = m_dir
                        else:
                            # 'pred' directly contains image outputs
                            if any(f.suffix.lower() in valid_exts for f in sub.glob("*")):
                                model_dirs["Prediction"] = sub
                    else:
                        if any(f.suffix.lower() in valid_exts for f in sub.glob("*")):
                            name = sub.name
                            if name.lower() in ("our", "our_model", "our_net"):
                                name = "Our Model"
                            elif name.lower() == "lama":
                                name = "LaMa"
                            model_dirs[name] = sub

        return gt_dir, model_dirs

    def _load_from_subfolders(self, gt_dir: Path, model_dirs: Dict[str, Path], valid_exts: set):
        gt_files = {f.stem: f for f in gt_dir.glob("*") if f.suffix.lower() in valid_exts}
        self.model_names = list(model_dirs.keys())

        model_files = {}
        for m_name, m_path in model_dirs.items():
            model_files[m_name] = {f.stem: f for f in m_path.glob("*") if f.suffix.lower() in valid_exts}

        for gt_stem, gt_file in gt_files.items():
            matched_group, matched_cat = "Uncategorized", "General"
            
            for group, cat_name, pattern in PATTERN_RULES:
                if pattern.search(gt_stem):
                    matched_group, matched_cat = group, cat_name
                    break

            gt_stem_clean = re.sub(r'(_gt|_mask)$', '', gt_stem, flags=re.IGNORECASE).strip('_')

            pred_paths = {}
            for m_name, m_dict in model_files.items():
                # 1. Direct match
                matched = m_dict.get(gt_stem) or m_dict.get(gt_stem_clean)
                
                # 2. Try common prediction suffixes
                if not matched:
                    for suf in ["_pred", "_gen", "_inpainted", "_out", "_result", f"_{m_name.lower().replace(' ', '_')}"]:
                        if (gt_stem + suf) in m_dict:
                            matched = m_dict[gt_stem + suf]
                            break
                        elif (gt_stem_clean + suf) in m_dict:
                            matched = m_dict[gt_stem_clean + suf]
                            break

                # 3. Fuzzy stem cleaning match
                if not matched:
                    for m_stem, m_file in m_dict.items():
                        m_stem_clean = re.sub(r'(_pred|_gen|_inpainted|_out|_result|_gt|_mask|_our|_lama)$', '', m_stem, flags=re.IGNORECASE).strip('_')
                        if m_stem_clean == gt_stem or m_stem_clean == gt_stem_clean:
                            matched = m_file
                            break

                if matched:
                    pred_paths[m_name] = matched

            if pred_paths:
                self.samples.append((gt_file, pred_paths, matched_group, matched_cat))

    def _load_from_flat_folder(self, base_dir: Path, valid_exts: set):
        all_files = sorted([f for f in base_dir.rglob("*") if f.suffix.lower() in valid_exts])
        
        gt_files, our_files, lama_files = {}, {}, {}
        
        for f in all_files:
            stem = f.stem.lower()
            if "_gt" in stem:
                gt_files[re.sub(r'_gt$', '', stem)] = f
            elif "_our" in stem:
                our_files[re.sub(r'_our(model)?$', '', stem)] = f
            elif "_lama" in stem:
                lama_files[re.sub(r'_lama$', '', stem)] = f

        self.model_names = []
        if our_files: self.model_names.append("Our Model")
        if lama_files: self.model_names.append("LaMa")

        for key, gt_path in gt_files.items():
            pred_paths = {}
            if "Our Model" in self.model_names and key in our_files:
                pred_paths["Our Model"] = our_files[key]
            if "LaMa" in self.model_names and key in lama_files:
                pred_paths["LaMa"] = lama_files[key]

            if pred_paths:
                matched_group, matched_cat = "Uncategorized", "General"
                for group, cat_name, pattern in PATTERN_RULES:
                    if pattern.search(key):
                        matched_group, matched_cat = group, cat_name
                        break
                self.samples.append((gt_path, pred_paths, matched_group, matched_cat))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt_path, pred_paths, group, cat_name = self.samples[idx]
        gt_pil = Image.open(gt_path).convert("RGB").resize(self.image_size)

        pred_tensors = {}
        for m_name, p_path in pred_paths.items():
            p_pil = Image.open(p_path).convert("RGB").resize(self.image_size)
            pred_tensors[m_name] = transforms.ToTensor()(p_pil)

        return transforms.ToTensor()(gt_pil), pred_tensors, group, cat_name


# ==========================================
# 3. Evaluation Core
# ==========================================

def run_evaluation(base_dir: Path, device: torch.device, batch_size: int = 16) -> Tuple[Dict, List[str]]:
    dataset = FlexibleMultiModelDataset(base_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    arcface_net = ArcFaceIdentityExtractor(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval() if HAS_LPIPS else None

    model_names = dataset.model_names
    records = {
        m_name: {"ALL": {"psnr": [], "ssim": [], "lpips": [], "arcface": []}, "Semantic": {}, "Box": {}}
        for m_name in model_names
    }

    print(f"\n📊 Evaluated {len(dataset)} matched sample pairs from '{base_dir}'")
    print(f"   Detected Models: {', '.join(model_names)}\n")

    for gt_b, pred_dict_b, groups, cat_names in tqdm(dataloader, desc="Evaluating Batches"):
        gt_b = gt_b.to(device)
        gt_norm = (gt_b * 2.0) - 1.0
        gt_np = (gt_b.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

        for m_name, pred_b in pred_dict_b.items():
            pred_b = pred_b.to(device)
            pred_norm = (pred_b * 2.0) - 1.0

            arcface_scores = arcface_net.compute_similarity(gt_b, pred_b)
            lp_scores = lpips_fn(gt_norm, pred_norm).flatten().cpu().tolist() if lpips_fn else [0.0] * gt_b.size(0)

            mse = torch.mean((gt_b - pred_b) ** 2, dim=[1, 2, 3]).clamp(min=1e-10)
            psnr_scores = (10.0 * torch.log10(1.0 / mse)).cpu().tolist()

            pred_np = (pred_b.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

            for i in range(gt_b.size(0)):
                ssim_val = float(compute_ssim_skimage(gt_np[i], pred_np[i], channel_axis=2, data_range=255))
                grp, cat = groups[i], cat_names[i]

                records[m_name]["ALL"]["psnr"].append(psnr_scores[i])
                records[m_name]["ALL"]["ssim"].append(ssim_val)
                records[m_name]["ALL"]["lpips"].append(lp_scores[i])
                records[m_name]["ALL"]["arcface"].append(arcface_scores[i])

                if grp in ("Semantic", "Box"):
                    if cat not in records[m_name][grp]:
                        records[m_name][grp][cat] = {"psnr": [], "ssim": [], "lpips": [], "arcface": []}
                    records[m_name][grp][cat]["psnr"].append(psnr_scores[i])
                    records[m_name][grp][cat]["ssim"].append(ssim_val)
                    records[m_name][grp][cat]["lpips"].append(lp_scores[i])
                    records[m_name][grp][cat]["arcface"].append(arcface_scores[i])

    def compile_stats(d):
        return {
            "PSNR ↑": float(np.mean(d["psnr"])) if d["psnr"] else 0.0,
            "SSIM ↑": float(np.mean(d["ssim"])) if d["ssim"] else 0.0,
            "LPIPS ↓": float(np.mean(d["lpips"])) if d["lpips"] else 0.0,
            "ArcFace ID ↑": float(np.mean(d["arcface"])) if d["arcface"] else 0.0,
            "Count": len(d["psnr"])
        }

    results = {}
    for m_name in model_names:
        results[m_name] = {"ALL": compile_stats(records[m_name]["ALL"]), "Semantic": {}, "Box": {}}
        for grp in ("Semantic", "Box"):
            for cat_name, ddict in records[m_name][grp].items():
                results[m_name][grp][cat_name] = compile_stats(ddict)

    return results, model_names


def main():
    parser = argparse.ArgumentParser(description="Inference Results Evaluator")
    parser.add_argument("--results_dir", type=str, default="samples/inference_results", help="Path to results directory")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, default="samples/inference_results/multi_model_metrics.json")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"❌ Error: Results directory '{results_path}' does not exist.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, model_names = run_evaluation(results_path, device, args.batch_size)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Metric computations finished! Detailed JSON saved to: {out_json}")


if __name__ == "__main__":
    main()