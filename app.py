import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr

# --- Dynamic Path Resolution ---
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

try:
    from facenet_pytorch import InceptionResnetV1
    HAS_ARCFACE = True
except ImportError:
    HAS_ARCFACE = False


# ==========================================
# 1. Fixed ArcFace Identity Extractor
# ==========================================
class ArcFaceIdentityExtractor(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.model = None

        if HAS_ARCFACE:
            try:
                # Load pretrained model safely
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
                for p in self.model.parameters():
                    p.requires_grad = False
                print("✅ ArcFace (InceptionResnetV1) model loaded successfully.")
            except Exception as e:
                print(f"⚠️ Warning: Failed to load ArcFace weights ({e}). ArcFace metric disabled.")
                self.model = None

    @torch.inference_mode()
    def compute_similarity(self, gt_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> float:
        if self.model is None:
            return 0.0

        try:
            # Force conversion to float32 on the correct device (Fixes AMP half-precision mismatch)
            gt_in = gt_tensor.detach().to(device=self.device, dtype=torch.float32)
            pred_in = pred_tensor.detach().to(device=self.device, dtype=torch.float32)

            if gt_in.min() >= 0.0:
                gt_in = (gt_in * 2.0) - 1.0
            if pred_in.min() >= 0.0:
                pred_in = (pred_in * 2.0) - 1.0

            gt_160 = F.interpolate(gt_in, size=(160, 160), mode='bilinear', align_corners=False)
            pred_160 = F.interpolate(pred_in, size=(160, 160), mode='bilinear', align_corners=False)

            emb_gt = self.model(gt_160)
            emb_pred = self.model(pred_160)

            cos_sim = F.cosine_similarity(emb_gt, emb_pred, dim=1)
            return float(cos_sim.item())
        except Exception as e:
            print(f"⚠️ Error computing ArcFace similarity: {e}")
            return 0.0


# ==========================================
# 2. Optimized Blending Helpers
# ==========================================
def apply_fast_feather_blend(orig_np: np.ndarray, gen_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """Fast Gaussian feathered alpha blending at 256x256 resolution."""
    if len(mask_np.shape) == 2:
        mask_3ch = mask_np[:, :, None]
    else:
        mask_3ch = mask_np

    feathered_mask = (cv2.GaussianBlur(mask_3ch.astype(np.float32), (15, 15), 3.0) / 255.0)
    if len(feathered_mask.shape) == 2:
        feathered_mask = feathered_mask[:, :, None]

    blended = (orig_np.astype(np.float32) * (1.0 - feathered_mask)) + (gen_np.astype(np.float32) * feathered_mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_seamless_blend(orig_np: np.ndarray, gen_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """Seamless Poisson blending at 256x256 resolution."""
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
        return apply_fast_feather_blend(orig_np, gen_np, mask_np)


def generate_irregular_mask(landmarks, location="center", img_size=(256, 256)) -> np.ndarray:
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
        px = int(cx + r * np.cos(angle))
        py = int(cy + r * np.sin(angle))
        pts.append([px, py])

    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return mask


def load_model(checkpoint_path: str, device: torch.device) -> LGNetGenerator:
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


# ==========================================
# 3. Pipeline Initialization
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/morphai_epoch_05.pt"

print(f"🚀 Initializing LGNet on device: {DEVICE}")
extractor = LandmarkExtractor()
mask_gen = MaskGenerator()
cond_gen = ConditioningGenerator()
arcface_net = ArcFaceIdentityExtractor(DEVICE)

model = None
if Path(CHECKPOINT_PATH).exists():
    model = load_model(CHECKPOINT_PATH, DEVICE)
else:
    print(f"⚠️ Checkpoint not found at '{CHECKPOINT_PATH}'. Please ensure the path is correct.")


# ==========================================
# 4. Interactive Inference Handler (3-Panel)
# ==========================================
@torch.inference_mode()
def run_inpaint_demo(editor_data, mask_source_mode, selected_variant, fast_blend_toggle, compute_arcface_toggle):
    if model is None:
        return None, None, None, None, None, "❌ Model checkpoint not loaded. Check CHECKPOINT_PATH."

    if editor_data is None or editor_data.get("background") is None:
        return None, None, None, None, None, "⚠️ Please upload an image first."

    # 1. Panel 1: Original Input Image (Resized to 256x256)
    orig_highres_pil = editor_data["background"].convert("RGB")
    resample_method = getattr(Image, "Resampling", Image).LANCZOS
    input_256_pil = orig_highres_pil.resize((256, 256), resample_method)
    orig_256_np = np.array(input_256_pil)

    # 2. Extract Landmarks
    temp_dir = Path("custom_outputs")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "_temp_app_input.png"
    input_256_pil.save(temp_path)

    landmarks = extractor.extract_landmarks(temp_path)
    regions = extractor.get_semantic_regions(landmarks)

    if temp_path.exists():
        temp_path.unlink()

    # 3. Determine Mask
    mask_256_np = None
    if mask_source_mode == "Draw Custom Mask on Canvas":
        layers = editor_data.get("layers", [])
        if layers and len(layers) > 0:
            drawn_mask_pil = layers[0].convert("L").resize((256, 256))
            mask_256_np = (np.array(drawn_mask_pil) > 10).astype(np.uint8) * 255

        if mask_256_np is None or mask_256_np.sum() == 0:
            return None, None, None, None, None, "⚠️ Custom mask mode selected, but no mask was drawn on canvas."
    else:
        if selected_variant.startswith("semantic_") and regions is not None:
            region_key = selected_variant.replace("semantic_", "")
            region_pts = regions.get(region_key, None)
            mask_256_np = mask_gen.generate_semantic_mask(region_pts)
        elif "irregular" in selected_variant:
            loc = "center"
            if "left" in selected_variant:
                loc = "left"
            elif "right" in selected_variant:
                loc = "right"
            mask_256_np = generate_irregular_mask(landmarks, location=loc)
        else:
            mask_256_np = generate_irregular_mask(landmarks, location="center")

    # 4. Generate Heatmap
    heatmap_256_np = cond_gen.generate_heatmap(landmarks, mask=mask_256_np)

    # 5. Build Tensors
    img_tensor = transforms.ToTensor()(input_256_pil).unsqueeze(0).to(DEVICE)
    norm_img = (img_tensor * 2.0) - 1.0

    mask_tensor = (torch.from_numpy(mask_256_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    heatmap_tensor = (torch.from_numpy(heatmap_256_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

    masked_img = norm_img * (1.0 - mask_tensor) + (-1.0) * mask_tensor
    input_5ch = torch.cat([masked_img, mask_tensor, heatmap_tensor], dim=1).float()

    # Panel 2: Masked Input Image (256x256)
    masked_vis_np = (((masked_img.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0) * 255.0).astype(np.uint8)
    masked_256_pil = Image.fromarray(masked_vis_np)

    # 6. Generator Pass
    if DEVICE.type == "cuda":
        with torch.amp.autocast("cuda"):
            raw_output = model(input_5ch)
    else:
        raw_output = model(input_5ch)

    # 7. Safe ArcFace Metric Score Computation
    arcface_score = None
    if compute_arcface_toggle:
        if arcface_net.model is not None:
            arcface_score = arcface_net.compute_similarity(norm_img, raw_output)

    # 8. Panel 3: Reconstructed Output (256x256)
    gen_tensor = (raw_output.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
    gen_256_np = (gen_tensor.permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)

    blend_fn = apply_fast_feather_blend if fast_blend_toggle else apply_seamless_blend
    reconstructed_256_np = blend_fn(orig_256_np, gen_256_np, mask_256_np)
    reconstructed_256_pil = Image.fromarray(reconstructed_256_np)

    # Status summary message
    status_msg = "✅ Inpainting complete!"
    if compute_arcface_toggle:
        if arcface_score is not None:
            status_msg += f" | 👤 ArcFace ID Similarity: {arcface_score:.4f}"
        else:
            status_msg += " | ⚠️ ArcFace Model not initialized."

    return (
        input_256_pil,
        masked_256_pil,
        reconstructed_256_pil,
        Image.fromarray(heatmap_256_np),
        Image.fromarray(mask_256_np),
        status_msg
    )


# ==========================================
# 5. Gradio Web Interface Layout (3-Panel)
# ==========================================
VARIANTS_LIST = [
    "semantic_left_eye",
    "semantic_right_eye",
    "semantic_left_eyebrow",
    "semantic_right_eyebrow",
    "semantic_nose",
    "semantic_mouth",
    "irregular_shape",
    "irregular_shape_left",
    "irregular_shape_right",
]

with gr.Blocks(title="MorphAI LGNet Custom Reconstruction") as demo:
    gr.Markdown("# 🎭 LGNet 3-Panel Facial Inpainting (256x256)")

    with gr.Row():
        with gr.Column(scale=1):
            editor_input = gr.ImageEditor(
                label="Input Face Canvas",
                type="pil",
                sources=["upload"],
                image_mode="RGB",
                layers=False,
                canvas_size=(256, 256)
            )

            mask_mode = gr.Radio(
                choices=["Draw Custom Mask on Canvas", "Use Auto-Preset Variant"],
                value="Use Auto-Preset Variant",
                label="Mask Selection Mode"
            )

            variant_dropdown = gr.Dropdown(
                choices=VARIANTS_LIST,
                value="semantic_nose",
                label="Preset Mask Variant"
            )

            with gr.Accordion("⚡ Performance Tweaks", open=False):
                fast_blend_chk = gr.Checkbox(
                    value=True,
                    label="Use Fast Alpha Feathering"
                )
                calc_arcface_chk = gr.Checkbox(
                    value=True,
                    label="Calculate ArcFace Identity Score"
                )

            run_btn = gr.Button("✨ Run Inpainting", variant="primary")
            status_output = gr.Textbox(label="Status & Metrics", interactive=False)

    # 3-Panel Horizontal Layout (256x256 display)
    gr.Markdown("### 📸 Inference Results")
    with gr.Row():
        p1_input = gr.Image(label="1. Original Input (256x256)", type="pil")
        p2_masked = gr.Image(label="2. Masked Image (256x256)", type="pil")
        p3_reconstructed = gr.Image(label="3. Reconstructed Output (256x256)", type="pil")

    with gr.Row():
        heatmap_preview = gr.Image(label="Landmark Heatmap (256x256)", type="pil")
        mask_preview = gr.Image(label="Binary Mask (256x256)", type="pil")

    run_btn.click(
        fn=run_inpaint_demo,
        inputs=[editor_input, mask_mode, variant_dropdown, fast_blend_chk, calc_arcface_chk],
        outputs=[p1_input, p2_masked, p3_reconstructed, heatmap_preview, mask_preview, status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=3000, share=False)
    