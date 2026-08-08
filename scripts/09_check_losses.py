import sys
from pathlib import Path

print("[1/4] Starting loss module verification...", flush=True)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import torch
from src.models import LGNetGenerator, PatchDiscriminator
from src.losses import LGNetLoss, DiscriminatorHingeLoss

def check_losses():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2/4] Running check on compute device: {device}", flush=True)

    generator = LGNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)

    gen_criterion = LGNetLoss().to(device)
    disc_criterion = DiscriminatorHingeLoss().to(device)

    print("[3/4] Creating dummy inputs and computing loss forward pass...", flush=True)
    dummy_input = torch.randn(2, 5, 256, 256, device=device)
    gt_img = torch.randn(2, 3, 256, 256, device=device)
    mask = torch.ones(2, 1, 256, 256, device=device)

    # --- Step A: Generator Pass ---
    fake_img = generator(dummy_input)
    disc_fake_for_gen = discriminator(fake_img)
    total_gen_loss, loss_dict = gen_criterion(fake_img, gt_img, mask, disc_fake_for_gen)

    # --- Step B: Discriminator Pass (with .detach() to separate graph) ---
    disc_fake_for_disc = discriminator(fake_img.detach())
    disc_real = discriminator(gt_img)
    disc_loss = disc_criterion(disc_real, disc_fake_for_disc)

    print(f"✓ Total Generator Loss  : {total_gen_loss.item():.4f}")
    print(f"  └─ Breakdown -> L1: {loss_dict['l1']:.4f} | Perceptual: {loss_dict['perceptual']:.4f} | Style: {loss_dict['style']:.4f} | Adv: {loss_dict['adv']:.4f}")
    print(f"✓ Discriminator Loss   : {disc_loss.item():.4f}\n")

    print("[4/4] Testing backward pass gradient computation...", flush=True)
    
    # Backward pass for Generator
    total_gen_loss.backward()
    print("✓ Generator gradients computed successfully!")

    # Backward pass for Discriminator
    disc_loss.backward()
    print("✓ Discriminator gradients computed successfully!")

    print("\nLoss Objective Pipeline Verified Successfully!\n", flush=True)

if __name__ == "__main__":
    check_losses()