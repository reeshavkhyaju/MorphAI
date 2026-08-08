import sys
from pathlib import Path

print("[1/5] Initializing script...", flush=True)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

print("[2/5] Importing PyTorch...", flush=True)
import torch

print("[3/5] Importing Generator & Discriminator...", flush=True)
from src.models import LGNetGenerator, PatchDiscriminator

def check_models():
    print("\n--- Testing MorphAI Models ---", flush=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[4/5] Running on device: {device}", flush=True)

    generator = LGNetGenerator(in_channels=5, out_channels=3).to(device)
    discriminator = PatchDiscriminator(in_channels=3).to(device)

    print("[5/5] Passing dummy batch through LGNet & Discriminator...", flush=True)
    dummy_input = torch.randn(2, 5, 256, 256, device=device)

    fake_img = generator(dummy_input)
    print(f"✓ Generator Output Tensor Shape: {fake_img.shape}", flush=True)

    disc_pred = discriminator(fake_img)
    print(f"✓ Discriminator Output Patch Shape: {disc_pred.shape}\n", flush=True)

    gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)

    print("--- Model Parameter Summary ---", flush=True)
    print(f"LGNet Generator trainable parameters: {gen_params:,}", flush=True)
    print(f"Patch Discriminator trainable parameters: {disc_params:,}", flush=True)
    print("--------------------------------", flush=True)
    print("\nModel Architecture Check Completed Successfully!\n", flush=True)

if __name__ == "__main__":
    check_models()