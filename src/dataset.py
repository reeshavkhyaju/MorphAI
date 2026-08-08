import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class MorphAIDataset(Dataset):
    def __init__(self, manifest_path, base_dir=None):
        self.manifest_path = Path(manifest_path)
        self.base_dir = Path(base_dir) if base_dir else self.manifest_path.resolve().parent.parent.parent.parent

        with open(self.manifest_path, "r") as f:
            self.entries = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Resolve paths
        img_p = self.base_dir / entry["image_path"]
        mask_p = self.base_dir / entry["mask_path"]
        heat_p = self.base_dir / entry["heatmap_path"]

        # Read RGB image, Mask, and Heatmap
        img = cv2.imread(str(img_p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.imread(str(heat_p), cv2.IMREAD_GRAYSCALE)

        # Convert to float numpy arrays normalized to [0, 1]
        img_float = img.astype(np.float32) / 255.0
        mask_float = (mask > 127).astype(np.float32)  # Binary 0.0 or 1.0
        heatmap_float = heatmap.astype(np.float32) / 255.0

        # Apply mask to image: set masked region pixels to 0
        masked_img = img_float * (1.0 - mask_float[:, :, None])

        # Normalize images from [0, 1] to [-1, 1] for GAN stability
        gt_tensor = torch.from_numpy(img_float).permute(2, 0, 1) * 2.0 - 1.0
        masked_img_tensor = torch.from_numpy(masked_img).permute(2, 0, 1) * 2.0 - 1.0

        # Mask and Heatmap tensors shape: (1, H, W) in range [0, 1]
        mask_tensor = torch.from_numpy(mask_float).unsqueeze(0)
        heatmap_tensor = torch.from_numpy(heatmap_float).unsqueeze(0)

        # Concatenate into 5-channel Input Tensor: (3 RGB + 1 Mask + 1 Heatmap)
        input_tensor = torch.cat([masked_img_tensor, mask_tensor, heatmap_tensor], dim=0)

        return {
            "input": input_tensor,        # Shape: (5, 256, 256)
            "gt": gt_tensor,              # Shape: (3, 256, 256)
            "mask": mask_tensor,          # Shape: (1, 256, 256)
            "heatmap": heatmap_tensor,    # Shape: (1, 256, 256)
            "mask_type": entry["mask_type"]
        }