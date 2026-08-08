# MorphAI

MorphAI is a facial-image processing and prediction pipeline built around CelebA, MediaPipe Face Mesh, and an LGNet-based generator. The code in this repository prepares face images, detects landmarks, builds masks and conditioning heatmaps, and packages everything into manifests that can be loaded by PyTorch.

## What the project does

The pipeline takes CelebA images and turns them into training-ready samples:

1. Read CelebA split metadata from `data/raw/list_eval_partition.csv`.
2. Resize the images to `256 x 256`.
3. Detect facial landmarks with MediaPipe Face Mesh.
4. Build semantic masks for facial regions like eyes, eyebrows, nose, and mouth.
5. Build random rectangular masks for additional training cases.
6. Generate conditioning heatmaps from visible landmarks.
7. Write JSON manifests that point to each image, mask, and heatmap.
8. Load those manifests in a PyTorch dataset and feed them into the LGNet wrapper.

## Repository Layout

- `config.py` - central paths, image size, batch size, and output directories.
- `src/` - core dataset, landmark, mask, conditioning, and LGNet wrapper code.
- `scripts/` - one-step pipeline and validation scripts.
- `frontend/` - a static demo UI that can upload an image and call a local prediction endpoint.
- `external/lgnet/` - vendored LGNet model code and checkpoints.

## Data Flow

The data pipeline is split into small steps so each stage can be checked separately:

- `scripts/01_prepare_splits.py` reads the CelebA partition CSV and writes file lists for train, val, and test.
- `scripts/02_resize_images.py` resizes raw images into `data/processed/train`, `val`, and `test`.
- `scripts/03_check_landmarks.py` verifies landmark detection on the resized images.
- `scripts/04_generate_masks_and_conditioning.py` creates masks and heatmaps and writes `full_manifest.json`.
- `scripts/05_build_manifest.py` splits the full manifest into per-split JSON files.
- `scripts/06_visual_spotcheck.py` writes a quick visual sanity check image.
- `scripts/07_test_dataset_loader.py` checks the PyTorch dataset and dataloader.
- `scripts/08_test_lgnet_wrapper.py` checks the LGNet wrapper forward pass.

## Core Modules

- `src/dataset.py` loads a manifest entry and returns `masked_rgb`, `mask`, `heatmap`, and `gt` tensors.
- `src/landmarks.py` detects Face Mesh landmarks and defines the facial regions used for semantic masks.
- `src/masks.py` creates semantic hull masks and random rectangular masks.
- `src/conditioning.py` builds landmark-based conditioning maps.
- `src/lgnet_wrapper.py` loads pretrained LGNet stage 1 weights and expands the first convolution to accept an extra conditioning channel.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The project expects the CelebA files in these locations:

- `data/raw/list_eval_partition.csv`
- `data/raw/img_align_celeba/img_align_celeba/`

It also expects LGNet weights at:

- `external/lgnet/checkpoints/celebahq_LGNet/latest_net_G1.pth`

## Running the Pipeline

A typical run is:

```bash
python scripts/01_prepare_splits.py
python scripts/02_resize_images.py
python scripts/03_check_landmarks.py
python scripts/04_generate_masks_and_conditioning.py
python scripts/05_build_manifest.py
python scripts/06_visual_spotcheck.py
python scripts/07_test_dataset_loader.py
python scripts/08_test_lgnet_wrapper.py
```

## Outputs

The scripts write generated artifacts into `data/processed/`:

- `train/`, `val/`, `test/` - resized images.
- `masks/` - binary masks.
- `heatmaps/` - landmark conditioning maps.
- `manifests/` - JSON file lists for each split.

The spot check script writes:

- `spotcheck_masked.png`
- `spotcheck_heatmap.png`

## Frontend

The `frontend/` folder contains a static Bootstrap demo UI. Its JavaScript currently calls `http://127.0.0.1:5000/predict`, so you need a local backend that exposes that endpoint for the upload/predict flow to work.

## Notes

- `config.py` creates the output directories on import.
- `scripts/01_prepare_splits.py` currently uses a subset size of 8000 for train and proportionally smaller val/test splits.
- `src/face_align.py` and `src/identity_loss.py` are present but empty in this version of the repository.
