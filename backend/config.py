"""Runtime configuration for the MorphAI Flask backend."""

import os
from pathlib import Path

# backend/ -> plastic/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# # The original research project. Model code, weights and evaluation results all
# # live here; the backend imports from it instead of duplicating anything.
# MORPHAI_DIR = Path(os.environ.get("MORPHAI_DIR", PROJECT_ROOT / "MorphAI")).resolve()

# CHECKPOINT_PATH = Path(
#     os.environ.get("MORPHAI_CHECKPOINT", MORPHAI_DIR / "checkpoints" / "morphai_epoch_05.pt")
# ).resolve()

# The original research project. Model code, weights and evaluation results all
# live here; the backend imports from it instead of duplicating anything.
#
# NOTE: in this layout, checkpoints/, src/, external/, samples/ etc. live
# directly under the repo root (MorphAI-main), not in a nested "MorphAI"
# subfolder — so MORPHAI_DIR defaults to PROJECT_ROOT itself.
MORPHAI_DIR = Path(os.environ.get("MORPHAI_DIR", PROJECT_ROOT)).resolve()

CHECKPOINT_PATH = Path(
    os.environ.get("MORPHAI_CHECKPOINT", MORPHAI_DIR / "checkpoints" / "morphai_epoch_05.pt")
).resolve()

# Dataset-level evaluation produced by scripts/evaluate_metrics.py
EVAL_RESULTS_PATH = MORPHAI_DIR / "samples" / "final_eval_results.json"

# Built React bundle (frontend/dist). Served by Flask when it exists so the
# whole project can run from a single process in production.
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

HOST = os.environ.get("MORPHAI_HOST", "127.0.0.1")
PORT = int(os.environ.get("MORPHAI_PORT", "5000"))
DEBUG = os.environ.get("MORPHAI_DEBUG", "0") == "1"

MAX_UPLOAD_MB = 16
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Disable the ArcFace identity metric entirely (it downloads vggface2 weights
# on first use) by setting MORPHAI_DISABLE_ARCFACE=1.
DISABLE_ARCFACE = os.environ.get("MORPHAI_DISABLE_ARCFACE", "0") == "1"
