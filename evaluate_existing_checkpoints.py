import sys
import glob
import re
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.dataset import MorphAIDataset
from src.models import LGNetGenerator
from src.losses import LGNetLoss

def evaluate_checkpoints():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluating Checkpoints for Training & Validation Losses on {device} ===")

    train_manifest = BASE_DIR / "data" / "processed" / "manifests" / "train_manifest.json"
    val_manifest = BASE_DIR / "data" / "processed" / "manifests" / "val_manifest.json"
    ckpt_pattern = str(BASE_DIR / "checkpoints" / "morphai_epoch_*.pt")
    
    ckpt_files = sorted(glob.glob(ckpt_pattern))
    if not ckpt_files:
        print("No checkpoint files found in 'checkpoints/' directory!")
        return

    # Prepare 500-sample subsets for fast evaluation
    train_dataset = MorphAIDataset(train_manifest, base_dir=BASE_DIR)
    val_dataset = MorphAIDataset(val_manifest, base_dir=BASE_DIR)

    train_subset = Subset(train_dataset, range(min(500, len(train_dataset))))
    val_subset = Subset(val_dataset, range(min(500, len(val_dataset))))

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=2)

    generator = LGNetGenerator().to(device)
    loss_fn = LGNetLoss().to(device)

    epochs = []
    train_losses = []
    val_losses = []

    def compute_dataset_loss(loader):
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch["input"].to(device)
                gt = batch["gt"].to(device)
                mask = batch["mask"].to(device)

                with torch.amp.autocast('cuda'):
                    fake_img = generator(inputs)
                    loss, _ = loss_fn(fake_img, gt, mask)

                total_loss += loss.item()
                num_batches += 1
        return total_loss / max(num_batches, 1)

    for filepath in ckpt_files:
        match = re.search(r"epoch_(\d+)", filepath)
        if not match:
            continue
        epoch_num = int(match.group(1))

        ckpt = torch.load(filepath, map_location=device)
        generator.load_state_dict(ckpt["generator_state_dict"])
        generator.eval()

        t_loss = compute_dataset_loss(train_loader)
        v_loss = compute_dataset_loss(val_loader)

        epochs.append(epoch_num)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        print(f"Epoch {epoch_num:02d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

    # Plot Training vs Validation Curves
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, marker='o', color='#1f77b4', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses, marker='s', color='#ff7f0e', linewidth=2, linestyle='--', label='Validation Loss')
    plt.title('LGNet Training vs Validation Performance', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_file = "model_performance_graph2.png"
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"\nDone! Updated graph saved as '{out_file}'.")

if __name__ == "__main__":
    evaluate_checkpoints()

# import sys
# import glob
# import re
# import torch
# from pathlib import Path
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

# BASE_DIR = Path(__file__).resolve().parent
# sys.path.append(str(BASE_DIR))

# from src.dataset import MorphAIDataset
# from src.models import LGNetGenerator
# from src.losses import LGNetLoss

# def evaluate_checkpoints():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"=== Evaluating Checkpoints on Full Datasets on {device} ===")

#     train_manifest = BASE_DIR / "data" / "processed" / "manifests" / "train_manifest.json"
#     val_manifest = BASE_DIR / "data" / "processed" / "manifests" / "val_manifest.json"
#     ckpt_pattern = str(BASE_DIR / "checkpoints" / "morphai_epoch_*.pt")
    
#     ckpt_files = sorted(glob.glob(ckpt_pattern))
#     if not ckpt_files:
#         print("No checkpoint files found in 'checkpoints/' directory!")
#         return

#     # Load Full Datasets
#     print("Loading full datasets into memory...")
#     train_dataset = MorphAIDataset(train_manifest, base_dir=BASE_DIR)
#     val_dataset = MorphAIDataset(val_manifest, base_dir=BASE_DIR)

#     # Note: If full train evaluation is too slow, you can wrap train_dataset back in Subset
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

#     generator = LGNetGenerator().to(device)
#     loss_fn = LGNetLoss().to(device)

#     epochs = []
#     train_losses = []
#     val_losses = []

#     def compute_dataset_loss(loader, name):
#         total_loss = 0.0
#         num_batches = len(loader)
#         with torch.no_grad():
#             for i, batch in enumerate(loader):
#                 inputs = batch["input"].to(device, non_blocking=True)
#                 gt = batch["gt"].to(device, non_blocking=True)
#                 mask = batch["mask"].to(device, non_blocking=True)

#                 with torch.amp.autocast('cuda'):
#                     fake_img = generator(inputs)
#                     loss, _ = loss_fn(fake_img, gt, mask)

#                 total_loss += loss.item()
#                 if (i + 1) % 500 == 0 or (i + 1) == num_batches:
#                     print(f"   [{name}] Processed batch {i+1}/{num_batches}", flush=True)

#         return total_loss / max(num_batches, 1)

#     for filepath in ckpt_files:
#         match = re.search(r"epoch_(\d+)", filepath)
#         if not match:
#             continue
#         epoch_num = int(match.group(1))

#         print(f"\n--- Evaluating Epoch {epoch_num:02d} ---", flush=True)
#         ckpt = torch.load(filepath, map_location=device)
#         generator.load_state_dict(ckpt["generator_state_dict"])
#         generator.eval()

#         t_loss = compute_dataset_loss(train_loader, "Train")
#         v_loss = compute_dataset_loss(val_loader, "Val")

#         epochs.append(epoch_num)
#         train_losses.append(t_loss)
#         val_losses.append(v_loss)

#         print(f"Epoch {epoch_num:02d} Complete | Full Train Loss: {t_loss:.4f} | Full Val Loss: {v_loss:.4f}")

#     # Plot Full Performance Graph
#     plt.figure(figsize=(9, 5))
#     plt.plot(epochs, train_losses, marker='o', color='#1f77b4', linewidth=2, label='Full Training Loss')
#     plt.plot(epochs, val_losses, marker='s', color='#ff7f0e', linewidth=2, linestyle='--', label='Full Validation Loss')
#     plt.title('LGNet Full Dataset Performance Across Epochs', fontsize=12, fontweight='bold')
#     plt.xlabel('Epoch', fontsize=10)
#     plt.ylabel('Loss', fontsize=10)
#     plt.xticks(epochs)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#     plt.tight_layout()

#     out_file = "model_performance_full_graph.png"
#     plt.savefig(out_file, dpi=300)
#     plt.show()
#     print(f"\nDone! Full graph saved as '{out_file}'.")

# if __name__ == "__main__":
#     evaluate_checkpoints()

# import sys
# import glob
# import re
# import torch
# from pathlib import Path
# from torch.utils.data import DataLoader, Subset
# import matplotlib.pyplot as plt

# BASE_DIR = Path(__file__).resolve().parent
# sys.path.append(str(BASE_DIR))

# from src.dataset import MorphAIDataset
# from src.models import LGNetGenerator
# from src.losses import LGNetLoss

# def evaluate_checkpoints():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"=== Fast Evaluation Running on {device} ===")

#     # Enable CUDA optimization flags
#     if device.type == "cuda":
#         torch.backends.cudnn.benchmark = True

#     train_manifest = BASE_DIR / "data" / "processed" / "manifests" / "train_manifest.json"
#     val_manifest = BASE_DIR / "data" / "processed" / "manifests" / "val_manifest.json"
#     ckpt_pattern = str(BASE_DIR / "checkpoints" / "morphai_epoch_*.pt")
    
#     ckpt_files = sorted(glob.glob(ckpt_pattern))
#     if not ckpt_files:
#         print("No checkpoint files found in 'checkpoints/' directory!")
#         return

#     full_train = MorphAIDataset(train_manifest, base_dir=BASE_DIR)
#     full_val = MorphAIDataset(val_manifest, base_dir=BASE_DIR)

#     # Subsample for 95x speedup (3,200 train & 1,600 val images are sufficient for exact trends)
#     train_subset = Subset(full_train, range(min(3200, len(full_train))))
#     val_subset = Subset(full_val, range(min(1600, len(full_val))))

#     # num_workers=0 avoids Windows multiprocessing overhead/deadlocks
#     # batch_size=64 doubles GPU throughput
#     train_loader = DataLoader(train_subset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
#     val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

#     generator = LGNetGenerator().to(device)
#     loss_fn = LGNetLoss().to(device)

#     epochs = []
#     train_losses = []
#     val_losses = []

#     # torch.inference_mode() is faster than torch.no_grad()
#     @torch.inference_mode()
#     def compute_dataset_loss(loader):
#         total_loss = 0.0
#         num_batches = len(loader)
        
#         for batch in loader:
#             inputs = batch["input"].to(device, non_blocking=True)
#             gt = batch["gt"].to(device, non_blocking=True)
#             mask = batch["mask"].to(device, non_blocking=True)

#             with torch.amp.autocast('cuda'):
#                 fake_img = generator(inputs)
#                 loss, _ = loss_fn(fake_img, gt, mask)

#             total_loss += loss.item()

#         return total_loss / max(num_batches, 1)

#     for filepath in ckpt_files:
#         match = re.search(r"epoch_(\d+)", filepath)
#         if not match:
#             continue
#         epoch_num = int(match.group(1))

#         ckpt = torch.load(filepath, map_location=device)
#         generator.load_state_dict(ckpt["generator_state_dict"])
#         generator.eval()

#         t_loss = compute_dataset_loss(train_loader)
#         v_loss = compute_dataset_loss(val_loader)

#         epochs.append(epoch_num)
#         train_losses.append(t_loss)
#         val_losses.append(v_loss)

#         print(f"Epoch {epoch_num:02d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

#     # Plot Performance
#     plt.figure(figsize=(9, 5))
#     plt.plot(epochs, train_losses, marker='o', color='#1f77b4', linewidth=2, label='Training Loss')
#     plt.plot(epochs, val_losses, marker='s', color='#ff7f0e', linewidth=2, linestyle='--', label='Validation Loss')
#     plt.title('LGNet Training vs Validation Performance', fontsize=12, fontweight='bold')
#     plt.xlabel('Epoch', fontsize=10)
#     plt.ylabel('Loss', fontsize=10)
#     plt.xticks(epochs)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#     plt.tight_layout()

#     out_file = "model_performance_fast.png"
#     plt.savefig(out_file, dpi=300)
#     plt.show()
#     print(f"\nDone! Graph saved as '{out_file}'.")

# if __name__ == "__main__":
#     evaluate_checkpoints()