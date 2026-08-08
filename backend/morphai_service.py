"""
MorphAI inference service.

Wraps the trained LGNet generator that lives in ../MorphAI. The pipeline here is
a direct port of MorphAI/app.py + MorphAI/inference_custom.py, so the API returns
exactly the same reconstructions as the original Gradio demo:

    image -> 256x256 -> MediaPipe landmarks -> mask -> landmark heatmap
          -> 5-channel tensor (RGB + mask + heatmap) -> LGNetGenerator -> blend
"""

from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config

# --- Make the original project importable (src.models, src.landmarks, ...) ---
if str(config.MORPHAI_DIR) not in sys.path:
    sys.path.insert(0, str(config.MORPHAI_DIR))

from src.conditioning import ConditioningGenerator  # noqa: E402
from src.landmarks import SEMANTIC_LANDMARK_INDICES, LandmarkExtractor  # noqa: E402
from src.masks import MaskGenerator  # noqa: E402
from src.models import LGNetGenerator  # noqa: E402

try:
    from facenet_pytorch import InceptionResnetV1

    HAS_ARCFACE = True
except ImportError:
    HAS_ARCFACE = False

try:
    from skimage.metrics import structural_similarity as _skimage_ssim

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


IMAGE_SIZE = (256, 256)

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

VARIANT_LABELS = {
    "semantic_left_eye": "Left Eye",
    "semantic_right_eye": "Right Eye",
    "semantic_left_eyebrow": "Left Eyebrow",
    "semantic_right_eyebrow": "Right Eyebrow",
    "semantic_nose": "Nose",
    "semantic_mouth": "Mouth",
    "irregular_shape": "Irregular Blob (center)",
    "irregular_shape_left": "Irregular Blob (left)",
    "irregular_shape_right": "Irregular Blob (right)",
}

# Colours used by the landmark overlay, matching the legend in the UI.
REGION_COLORS = {
    "left_eye": (220, 53, 69),
    "right_eye": (220, 53, 69),
    "left_eyebrow": (255, 149, 0),
    "right_eyebrow": (255, 149, 0),
    "nose": (25, 160, 90),
    "mouth": (37, 99, 235),
}


# ==========================================
# ArcFace identity metric
# ==========================================
class ArcFaceIdentityExtractor(nn.Module):
    """512-d facial embeddings used for cosine identity similarity."""

    def __init__(self, device: torch.device, enabled: bool = True):
        super().__init__()
        self.device = device
        self.model = None
        self.error: Optional[str] = None

        if not enabled:
            self.error = "disabled by configuration"
            return
        if not HAS_ARCFACE:
            self.error = "facenet-pytorch is not installed"
            return

        try:
            self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
            for p in self.model.parameters():
                p.requires_grad = False
            print("[MorphAI] ArcFace (InceptionResnetV1/vggface2) loaded.")
        except Exception as exc:  # weights download can fail offline
            self.error = str(exc)
            self.model = None
            print(f"[MorphAI] ArcFace unavailable: {exc}")

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.inference_mode()
    def compute_similarity(self, gt: torch.Tensor, pred: torch.Tensor) -> Optional[float]:
        if self.model is None:
            return None
        try:
            gt_in = gt.detach().to(device=self.device, dtype=torch.float32)
            pred_in = pred.detach().to(device=self.device, dtype=torch.float32)

            # Both branches expect [-1, 1]
            if gt_in.min() >= 0.0:
                gt_in = (gt_in * 2.0) - 1.0
            if pred_in.min() >= 0.0:
                pred_in = (pred_in * 2.0) - 1.0

            gt_160 = F.interpolate(gt_in, size=(160, 160), mode="bilinear", align_corners=False)
            pred_160 = F.interpolate(pred_in, size=(160, 160), mode="bilinear", align_corners=False)

            cos = F.cosine_similarity(self.model(gt_160), self.model(pred_160), dim=1)
            return float(cos.item())
        except Exception as exc:
            print(f"[MorphAI] ArcFace similarity failed: {exc}")
            return None


# ==========================================
# Blending helpers (ported from MorphAI/app.py)
# ==========================================
def apply_fast_feather_blend(orig_np: np.ndarray, gen_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """Gaussian-feathered alpha blend."""
    mask_3ch = mask_np[:, :, None] if mask_np.ndim == 2 else mask_np
    feathered = cv2.GaussianBlur(mask_3ch.astype(np.float32), (15, 15), 3.0) / 255.0
    if feathered.ndim == 2:
        feathered = feathered[:, :, None]

    blended = orig_np.astype(np.float32) * (1.0 - feathered) + gen_np.astype(np.float32) * feathered
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_seamless_blend(orig_np: np.ndarray, gen_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """Poisson (seamless clone) blend, falling back to feathering on failure."""
    ys, xs = np.where(mask_np > 127)
    if len(ys) == 0:
        return gen_np

    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    flat_mask = mask_np.squeeze() if mask_np.ndim == 3 else mask_np

    try:
        return cv2.seamlessClone(gen_np, orig_np, flat_mask, center, cv2.NORMAL_CLONE)
    except Exception:
        return apply_fast_feather_blend(orig_np, gen_np, mask_np)


def generate_irregular_mask(landmarks, location: str = "center", img_size=IMAGE_SIZE) -> np.ndarray:
    """Organic polygon blob placed relative to the eyebrow line."""
    h, w = img_size
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, int(h * 0.22)

    if landmarks is not None and len(landmarks) >= 27:
        try:
            eyebrows = landmarks[17:27]
            eb_x_min = np.min(eyebrows[:, 0])
            eb_x_max = np.max(eyebrows[:, 0])
            eb_x_mid = np.mean(eyebrows[:, 0])
            eb_y_mean = np.mean(eyebrows[:, 1])

            face_width = eb_x_max - eb_x_min
            target_y = eb_y_mean - (face_width * 0.28)

            if location == "left":
                target_x = eb_x_min + (face_width * 0.20)
            elif location == "right":
                target_x = eb_x_max - (face_width * 0.20)
            else:
                target_x = eb_x_mid

            cx = int(np.clip(target_x, 15, w - 15))
            cy = int(np.clip(target_y, 15, h - 15))
        except Exception:
            pass

    num_points = np.random.randint(7, 11)
    base_radius = np.random.uniform(14, 20)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))

    pts = []
    for angle in angles:
        r = base_radius * np.random.uniform(0.75, 1.25)
        pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])

    cv2.fillPoly(mask, np.array([pts], dtype=np.int32), 255)
    return mask


# ==========================================
# Metrics
# ==========================================
def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-10:
        return 100.0
    return float(10.0 * np.log10((255.0**2) / mse))


def compute_masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Optional[float]:
    sel = mask > 127
    if not sel.any():
        return None
    diff = (a.astype(np.float64) - b.astype(np.float64))[sel]
    mse = float(np.mean(diff**2))
    if mse <= 1e-10:
        return 100.0
    return float(10.0 * np.log10((255.0**2) / mse))


def _fallback_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Gaussian-window SSIM, used when scikit-image is not installed."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a_f, b_f = a.astype(np.float64), b.astype(np.float64)

    mu_a = cv2.GaussianBlur(a_f, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b_f, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a_f * a_f, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b_f * b_f, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a_f * b_f, (11, 11), 1.5) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
    return float(ssim_map.mean())


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if HAS_SKIMAGE:
        return float(_skimage_ssim(a, b, channel_axis=2, data_range=255))
    return _fallback_ssim(a, b)


# ==========================================
# Encoding helpers
# ==========================================
def encode_png(array_or_pil) -> str:
    """Encode an image as a base64 data URI (PNG)."""
    pil = array_or_pil if isinstance(array_or_pil, Image.Image) else Image.fromarray(array_or_pil)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ==========================================
# Checkpoint loading
# ==========================================
def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[LGNetGenerator, Dict]:
    """Loads the generator weights, mirroring MorphAI/inference_custom.py."""
    print(f"[MorphAI] Loading checkpoint: {checkpoint_path}")
    model = LGNetGenerator().to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    meta: Dict = {}
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("generator_state_dict")
            or checkpoint.get("generator")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
        for key in ("epoch", "global_step", "step", "iteration"):
            value = checkpoint.get(key)
            if isinstance(value, (int, float)):
                meta[key] = value
    else:
        state_dict = checkpoint

    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()
    return model, meta


# ==========================================
# Service
# ==========================================
class MorphAIService:
    """Loads the pipeline once and serves inference requests."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = config.CHECKPOINT_PATH
        self.model: Optional[LGNetGenerator] = None
        self.load_error: Optional[str] = None
        self.checkpoint_meta: Dict = {}

        # MediaPipe's FaceMesh is not thread safe and Flask's dev server is
        # threaded, so every request serialises through this lock.
        self._lock = threading.Lock()

        print(f"[MorphAI] Initialising pipeline on device: {self.device}")
        self.extractor = LandmarkExtractor()
        self.mask_gen = MaskGenerator(image_size=IMAGE_SIZE)
        self.cond_gen = ConditioningGenerator(image_size=IMAGE_SIZE)
        self.arcface = ArcFaceIdentityExtractor(self.device, enabled=not config.DISABLE_ARCFACE)

        if self.checkpoint_path.exists():
            try:
                self.model, self.checkpoint_meta = load_model(self.checkpoint_path, self.device)
                print("[MorphAI] Generator ready.")
            except Exception as exc:
                self.load_error = f"Failed to load checkpoint: {exc}"
                print(f"[MorphAI] {self.load_error}")
        else:
            self.load_error = f"Checkpoint not found at {self.checkpoint_path}"
            print(f"[MorphAI] {self.load_error}")

    # ---------- status ----------
    @property
    def ready(self) -> bool:
        return self.model is not None

    def status(self) -> Dict:
        return {
            "status": "ok" if self.ready else "degraded",
            "model_loaded": self.ready,
            "load_error": self.load_error,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_name": self.checkpoint_path.name,
            "checkpoint_meta": self.checkpoint_meta,
            "arcface_available": self.arcface.available,
            "arcface_error": self.arcface.error,
            "ssim_backend": "scikit-image" if HAS_SKIMAGE else "opencv-fallback",
            "image_size": list(IMAGE_SIZE),
        }

    # ---------- dataset evaluation ----------
    @staticmethod
    def evaluation_report() -> Dict:
        path = config.EVAL_RESULTS_PATH
        try:
            source = path.relative_to(config.PROJECT_ROOT).as_posix()
        except ValueError:  # MORPHAI_DIR pointed outside the project
            source = path.as_posix()
        if not path.exists():
            return {"available": False, "source": source, "metrics": {}, "n_samples": None}

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        return {
            "available": True,
            "source": source,
            "n_samples": data.get("n_samples"),
            "metrics": data.get("metrics", data),
        }

    # ---------- helpers ----------
    @staticmethod
    def _open_rgb(image_bytes: bytes, what: str = "image") -> Image.Image:
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"The uploaded {what} could not be decoded: {exc}") from exc

    @classmethod
    def _to_pil_256(cls, image_bytes: bytes) -> Tuple[Image.Image, Image.Image]:
        original = cls._open_rgb(image_bytes)
        resample = getattr(Image, "Resampling", Image).LANCZOS
        return original, original.resize(IMAGE_SIZE, resample)

    def _build_mask(self, variant: str, landmarks, regions, custom_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, str]:
        if custom_mask is not None:
            return custom_mask, "custom"

        if variant.startswith("semantic_"):
            if regions is None:
                raise ValueError(
                    "No face detected, so a semantic mask cannot be built. "
                    "Upload a clearer frontal face image or pick an irregular variant."
                )
            region_key = variant.replace("semantic_", "")
            region_pts = regions.get(region_key)
            mask = self.mask_gen.generate_semantic_mask(region_pts)
            if mask.sum() == 0:
                raise ValueError(f"Region '{region_key}' could not be localised on this face.")
            return mask, "semantic"

        location = "center"
        if variant.endswith("_left"):
            location = "left"
        elif variant.endswith("_right"):
            location = "right"
        return generate_irregular_mask(landmarks, location=location), "irregular"

    # ---------- landmark visualisation ----------
    def detect_landmarks(self, image_bytes: bytes) -> Dict:
        """Returns the 256x256 input plus a colour-coded landmark overlay."""
        _, pil_256 = self._to_pil_256(image_bytes)
        base_np = np.array(pil_256)

        with self._lock:
            landmarks = self.extractor.extract_landmarks(pil_256)
            regions = self.extractor.get_semantic_regions(landmarks)

        overlay = base_np.copy()
        region_counts: Dict[str, int] = {}

        if landmarks is not None:
            # Faint mesh for all 468 (+refined) points.
            for x, y in landmarks:
                cv2.circle(overlay, (int(round(x)), int(round(y))), 1, (170, 180, 195), -1, cv2.LINE_AA)

            for region_name, indices in SEMANTIC_LANDMARK_INDICES.items():
                color = REGION_COLORS.get(region_name, (37, 99, 235))
                pts = [landmarks[i] for i in indices if i < len(landmarks)]
                region_counts[region_name] = len(pts)
                for x, y in pts:
                    cv2.circle(overlay, (int(round(x)), int(round(y))), 2, color, -1, cv2.LINE_AA)
                if len(pts) >= 3:
                    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                    cv2.polylines(overlay, [hull], True, color, 1, cv2.LINE_AA)

        heatmap = self.cond_gen.generate_heatmap(landmarks)

        return {
            "success": True,
            "face_detected": landmarks is not None,
            "landmark_count": 0 if landmarks is None else int(len(landmarks)),
            "region_counts": region_counts,
            "images": {
                "original": encode_png(base_np),
                "landmarks": encode_png(overlay),
                "heatmap": encode_png(heatmap),
            },
        }

    # ---------- main inference ----------
    @torch.inference_mode()
    def predict(
        self,
        image_bytes: bytes,
        variant: str = "semantic_nose",
        fast_blend: bool = True,
        compute_identity: bool = True,
        custom_mask_bytes: Optional[bytes] = None,
    ) -> Dict:
        if not self.ready:
            raise RuntimeError(self.load_error or "Model is not loaded.")

        started = time.perf_counter()

        original_pil, pil_256 = self._to_pil_256(image_bytes)
        orig_256_np = np.array(pil_256)

        custom_mask = None
        if custom_mask_bytes:
            mask_pil = self._open_rgb(custom_mask_bytes, "mask").convert("L").resize(IMAGE_SIZE)
            custom_mask = (np.array(mask_pil) > 10).astype(np.uint8) * 255
            if custom_mask.sum() == 0:
                raise ValueError("The supplied custom mask is empty.")

        with self._lock:
            landmarks = self.extractor.extract_landmarks(pil_256)
            regions = self.extractor.get_semantic_regions(landmarks)

            mask_256, mask_source = self._build_mask(variant, landmarks, regions, custom_mask)
            heatmap_256 = self.cond_gen.generate_heatmap(landmarks, mask=mask_256)

            img_tensor = transforms.ToTensor()(pil_256).unsqueeze(0).to(self.device)
            norm_img = (img_tensor * 2.0) - 1.0

            mask_tensor = (torch.from_numpy(mask_256).float() / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)
            heatmap_tensor = (torch.from_numpy(heatmap_256).float() / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)

            masked_img = norm_img * (1.0 - mask_tensor) + (-1.0) * mask_tensor
            input_5ch = torch.cat([masked_img, mask_tensor, heatmap_tensor], dim=1).float()

            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    raw_output = self.model(input_5ch)
            else:
                raw_output = self.model(input_5ch)

            identity = None
            if compute_identity:
                identity = self.arcface.compute_similarity(norm_img, raw_output)

        masked_vis_np = (((masked_img.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0) * 255.0).astype(np.uint8)

        gen_tensor = (raw_output.squeeze(0).float().clamp(-1.0, 1.0) + 1.0) / 2.0
        gen_256_np = (gen_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        blend_fn = apply_fast_feather_blend if fast_blend else apply_seamless_blend
        reconstructed_np = blend_fn(orig_256_np, gen_256_np, mask_256)

        elapsed = time.perf_counter() - started

        metrics = {
            "psnr": round(compute_psnr(orig_256_np, reconstructed_np), 4),
            "ssim": round(compute_ssim(orig_256_np, reconstructed_np), 4),
            "psnr_masked": None,
            "identity": None if identity is None else round(identity, 4),
            "mask_coverage": round(float((mask_256 > 127).mean() * 100.0), 2),
        }
        masked_psnr = compute_masked_psnr(orig_256_np, reconstructed_np, mask_256)
        if masked_psnr is not None:
            metrics["psnr_masked"] = round(masked_psnr, 4)

        return {
            "success": True,
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "mask_source": mask_source,
            "blend": "feather" if fast_blend else "seamless",
            "face_detected": landmarks is not None,
            "landmark_count": 0 if landmarks is None else int(len(landmarks)),
            "source_resolution": list(original_pil.size),
            "processing_time": round(elapsed, 3),
            "device": str(self.device),
            "metrics": metrics,
            "identity_available": self.arcface.available,
            "images": {
                "original": encode_png(orig_256_np),
                "masked": encode_png(masked_vis_np),
                "generated": encode_png(gen_256_np),
                "reconstructed": encode_png(reconstructed_np),
                "heatmap": encode_png(heatmap_256),
                "mask": encode_png(mask_256),
            },
        }


_service: Optional[MorphAIService] = None
_service_lock = threading.Lock()


def get_service() -> MorphAIService:
    """Process-wide singleton so the checkpoint is loaded exactly once."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = MorphAIService()
    return _service
