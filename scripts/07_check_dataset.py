import sys
from pathlib import Path
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.dataset import MorphAIDataset

def check_dataset():
    manifest_path = BASE_DIR / "data" / "processed" / "manifests" / "train_manifest.json"

    if not manifest_path.exists():
        print("Error: train_manifest.json not found.")
        return

    print("Initializing MorphAIDataset...")
    dataset = MorphAIDataset(manifest_path=manifest_path, base_dir=BASE_DIR)
    print(f"Total samples in training set: {len(dataset):,}")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    print("\nTesting PyTorch DataLoader batch fetch...")
    for batch in dataloader:
        inputs = batch["input"]
        gt = batch["gt"]
        mask = batch["mask"]
        heatmap = batch["heatmap"]

        print(f"✓ Input tensor shape   : {inputs.shape}  | Min: {inputs.min():.2f}, Max: {inputs.max():.2f}")
        print(f"✓ GT image tensor shape : {gt.shape}      | Min: {gt.min():.2f}, Max: {gt.max():.2f}")
        print(f"✓ Mask tensor shape     : {mask.shape}    | Min: {mask.min():.2f}, Max: {mask.max():.2f}")
        print(f"✓ Heatmap tensor shape  : {heatmap.shape} | Min: {heatmap.min():.2f}, Max: {heatmap.max():.2f}")
        print(f"✓ Batch mask types      : {batch['mask_type'][:3]}...")
        break

    print("\nDataset pipeline verified successfully!")

if __name__ == "__main__":
    check_dataset()