import sys, os, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LGNET_WEIGHTS_PATH
from src.lgnet_wrapper import LGNetStage1Conditioned

def test_forward_pass():
    model = LGNetStage1Conditioned(LGNET_WEIGHTS_PATH)
    model.eval()

    masked_rgb = torch.randn(2, 3, 256, 256)
    mask = torch.randint(0, 2, (2, 1, 256, 256)).float()
    heatmap = torch.rand(2, 1, 256, 256)

    with torch.no_grad():
        out = model(masked_rgb, mask, heatmap)
    print(f"Output shape: {out.shape}")          # expect (2, 3, 256, 256)
    print(f"First conv now: {model.net.model.model[0]}")  # confirm 5 input channels

if __name__ == "__main__":
    test_forward_pass()