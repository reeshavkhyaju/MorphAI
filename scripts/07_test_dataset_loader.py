import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from src.dataset import MorphAIDataset
from config import MANIFEST_DIR, BATCH_SIZE, NUM_WORKERS

def test_loader():
    manifest_path = os.path.join(MANIFEST_DIR, "manifest_train.json")
    ds = MorphAIDataset(manifest_path)
    print(f"Dataset size: {len(ds)}")

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    batch = next(iter(loader))

    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"{k}: {v.shape}, dtype={v.dtype}, range=({v.min().item():.2f}, {v.max().item():.2f})")
        else:
            print(f"{k}: {v[:3]}")  # print sample values for non-tensor fields

if __name__ == "__main__":
    test_loader()