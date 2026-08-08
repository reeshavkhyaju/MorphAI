import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.dataset import MorphAIDataset
from src.models import LGNetGenerator, PatchDiscriminator
from src.losses import LGNetLoss, DiscriminatorHingeLoss

# --- Configuration Hyperparameters (Optimized for 6GB VRAM) ---
BATCH_SIZE = 8           # Safe limit for 6GB VRAM
NUM_WORKERS = 2          # Prevents CPU thread hanging on Windows
NUM_EPOCHS = 5           # Optimized to 5 epochs (~1M face passes total)
LR_G = 1e-4              # Generator Learning Rate
LR_D = 4e-4              # Discriminator Learning Rate (TTUR)
BETA1 = 0.0
BETA2 = 0.9
SAVE_EVERY_STEPS = 1000  # Saves a checkpoint every 1000 batches so work is never lost!

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
SAMPLE_DIR = BASE_DIR / "samples"
TRAIN_MANIFEST = BASE_DIR / "data" / "processed" / "manifests" / "train_manifest.json"
VAL_MANIFEST = BASE_DIR / "data" / "processed" / "manifests" / "val_manifest.json"


def save_sample_visuals(epoch, step, generator, val_loader, device):
    """Generates visual previews [Input | Reconstruction | Ground Truth] for inspection."""
    generator.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        inputs = batch["input"].to(device)
        gt = batch["gt"].to(device)
        mask = batch["mask"].to(device)

        with torch.amp.autocast('cuda'):
            fake = generator(inputs)

        inpainted = inputs[:, :3] * (1.0 - mask) + fake * mask

        masked_view = (inputs[:, :3] + 1.0) / 2.0
        inpainted_view = (inpainted + 1.0) / 2.0
        gt_view = (gt + 1.0) / 2.0

        comparison = torch.cat([masked_view[:4], inpainted_view[:4], gt_view[:4]], dim=0)
        save_path = SAMPLE_DIR / f"epoch_{epoch+1:02d}_step_{step+1:05d}_sample.png"
        save_image(comparison, save_path, nrow=4, normalize=False)
        print(f" Saved visual sample preview -> {save_path}", flush=True)
    
    generator.train()


def save_checkpoint(path, epoch, step, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d):
    """Helper function to save model checkpoint to disk."""
    torch.save({
        "epoch": epoch + 1,
        "step": step + 1,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "opt_g_state_dict": opt_g.state_dict(),
        "opt_d_state_dict": opt_d.state_dict(),
        "scaler_g_state_dict": scaler_g.state_dict(),
        "scaler_d_state_dict": scaler_d.state_dict(),
    }, path)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting MorphAI AMP Training Engine on Device: {device} ===", flush=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & DataLoader Setup
    print("Loading datasets...", flush=True)
    train_dataset = MorphAIDataset(TRAIN_MANIFEST, base_dir=BASE_DIR)
    val_dataset = MorphAIDataset(VAL_MANIFEST, base_dir=BASE_DIR)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=2, pin_memory=True
    )

    print(f"Training Samples  : {len(train_dataset):,}")
    print(f"Validation Samples: {len(val_dataset):,}")
    print(f"Batches per Epoch : {len(train_loader):,}\n", flush=True)

    # 2. Model & Loss Instantiations
    generator = LGNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)

    gen_criterion = LGNetLoss().to(device)
    disc_criterion = DiscriminatorHingeLoss().to(device)

    # 3. Optimizers & AMP Gradient Scalers
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    # --- Auto-Resume Checkpoint Logic ---
    start_epoch = 0
    latest_ckpt_path = CHECKPOINT_DIR / "morphai_latest.pt"
    if latest_ckpt_path.exists():
        print(f" Resuming from existing checkpoint: {latest_ckpt_path}", flush=True)
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        generator.load_state_dict(ckpt["generator_state_dict"])
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        opt_g.load_state_dict(ckpt["opt_g_state_dict"])
        opt_d.load_state_dict(ckpt["opt_d_state_dict"])
        scaler_g.load_state_dict(ckpt["scaler_g_state_dict"])
        scaler_d.load_state_dict(ckpt["scaler_d_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"--> Successfully loaded checkpoint. Resuming from Epoch {start_epoch + 1}\n", flush=True)

    # 4. Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        generator.train()
        discriminator.train()
        
        epoch_start = time.time()
        running_g_loss = 0.0
        running_d_loss = 0.0

        for step, batch in enumerate(train_loader):
            inputs = batch["input"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # ---------------------------
            # Train Discriminator
            # ---------------------------
            opt_d.zero_grad()
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    fake_img = generator(inputs)

                disc_real = discriminator(gt)
                disc_fake = discriminator(fake_img.detach())
                loss_d = disc_criterion(disc_real, disc_fake)

            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ---------------------------
            # Train Generator
            # ---------------------------
            opt_g.zero_grad()
            with torch.amp.autocast('cuda'):
                fake_img_g = generator(inputs)
                disc_fake_g = discriminator(fake_img_g)
                loss_g, loss_dict = gen_criterion(fake_img_g, gt, mask, disc_fake_g)

            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            running_g_loss += loss_g.item()
            running_d_loss += loss_d.item()

            # Print Step Progress at Step 0, then every 50 steps
            if step == 0 or (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch+1:02d}/{NUM_EPOCHS:02d}] | "
                    f"Step [{step+1:05d}/{len(train_loader):05d}] | "
                    f"G Loss: {loss_g.item():.4f} (L1: {loss_dict['l1']:.3f}, Perc: {loss_dict['perceptual']:.3f}, Style: {loss_dict['style']:.3f}, ID: {loss_dict['id']:.3f}, Adv: {loss_dict['adv']:.3f}) | "
                    f"D Loss: {loss_d.item():.4f}",
                    flush=True
                )

            # --- Periodic Mid-Epoch Checkpoint Saving ---
            if (step + 1) % SAVE_EVERY_STEPS == 0:
                save_checkpoint(latest_ckpt_path, epoch, step, generator, discriminator, opt_g, opt_d, scaler_g, scaler_d)
                save_sample_visuals(epoch, step, generator, val_loader, device)
                print(f" Mid-epoch checkpoint saved at step {step+1:,} -> {latest_ckpt_path}", flush=True)

        elapsed = time.time() - epoch_start
        avg_g = running_g_loss / len(train_loader)
        avg_d = running_d_loss / len(train_loader)

        print(f"\n Epoch {epoch+1:02d} Summary [{elapsed:.1f}s] | Avg G Loss: {avg_g:.4f} | Avg D Loss: {avg_d:.4f}", flush=True)

        # Save Visual Sample & End-of-Epoch Checkpoint
        save_sample_visuals(epoch, len(train_loader), generator, val_loader, device)
        
        epoch_ckpt_path = CHECKPOINT_DIR / f"morphai_epoch_{epoch+1:02d}.pt"
        save_checkpoint(epoch_ckpt_path, epoch, len(train_loader), generator, discriminator, opt_g, opt_d, scaler_g, scaler_d)
        save_checkpoint(latest_ckpt_path, epoch, len(train_loader), generator, discriminator, opt_g, opt_d, scaler_g, scaler_d)
        
        print(f" End of Epoch Checkpoint saved -> {epoch_ckpt_path}\n" + "-"*80, flush=True)

if __name__ == "__main__":
    train()