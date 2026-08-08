import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class GatedConv2d(nn.Module):
    """
    Gated Convolution Layer: Dynamically masks out feature channels 
    based on spatial context and missing region boundaries.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.sigmoid = nn.Sigmoid()
        self.activation = activation

    def forward(self, x):
        features = self.feature_conv(x)
        gating = self.sigmoid(self.gating_conv(x))
        if self.activation is not None:
            features = self.activation(features)
        return features * gating


class DilatedGatedBlock(nn.Module):
    """
    Dilated Bottleneck Block to expand the receptive field 
    without losing fine-grained spatial resolution.
    """
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.conv1 = GatedConv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = GatedConv2d(channels, channels, kernel_size=3, padding=1, dilation=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class LGNetGenerator(nn.Module):
    """
    Local-Global Net (LGNet) Inpainting Generator.
    Input:  (B, 5, 256, 256) -> 3 RGB + 1 Mask + 1 Landmark Heatmap
    Output: (B, 3, 256, 256) -> Reconstructed RGB Image in range [-1, 1]
    """
    def __init__(self, in_channels=5, out_channels=3, base_channels=64):
        super().__init__()

        # --- Encoder ---
        self.enc1 = GatedConv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3)  # (B, 64, 256, 256)
        self.enc2 = GatedConv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)  # (B, 128, 128, 128)
        self.enc3 = GatedConv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)  # (B, 256, 64, 64)

        # --- Dilated Bottleneck (Receptive Field Expansion) ---
        self.bottleneck = nn.Sequential(
            DilatedGatedBlock(base_channels * 4, dilation=2),
            DilatedGatedBlock(base_channels * 4, dilation=4),
            DilatedGatedBlock(base_channels * 4, dilation=8),
            DilatedGatedBlock(base_channels * 4, dilation=16)
        )

        # --- Decoder with Skip Connections ---
        # Interp(b) [256] + Skip e2 [128] = 384 channels (base_channels * 6)
        self.dec1 = GatedConv2d(base_channels * 6, base_channels * 2, kernel_size=3, padding=1)  # (B, 128, 128, 128)

        # Interp(d1) [128] + Skip e1 [64] = 192 channels (base_channels * 3)
        self.dec2 = GatedConv2d(base_channels * 3, base_channels, kernel_size=3, padding=1)  # (B, 64, 256, 256)

        # d2 [64] + Skip e1 [64] = 128 channels (base_channels * 2)
        self.out_conv = nn.Conv2d(base_channels * 2, out_channels, kernel_size=7, padding=3)  # (B, 3, 256, 256)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                  # (B, 64, 256, 256)
        e2 = self.enc2(e1)                 # (B, 128, 128, 128)
        e3 = self.enc3(e2)                 # (B, 256, 64, 64)

        # Bottleneck
        b = self.bottleneck(e3)            # (B, 256, 64, 64)

        # Decoder + Skip Connections
        d1 = F.interpolate(b, scale_factor=2, mode='nearest')  # (B, 256, 128, 128)
        d1 = torch.cat([d1, e2], dim=1)    # Concatenate skip (256 + 128 = 384 channels)
        d1 = self.dec1(d1)                 # (B, 128, 128, 128)

        d2 = F.interpolate(d1, scale_factor=2, mode='nearest')  # (B, 128, 256, 256)
        d2 = torch.cat([d2, e1], dim=1)    # Concatenate skip (128 + 64 = 192 channels)
        d2 = self.dec2(d2)                 # (B, 64, 256, 256)

        out = self.out_conv(torch.cat([d2, e1], dim=1))  # Concatenate skip (64 + 64 = 128 channels) -> (B, 3, 256, 256)
        return torch.tanh(out)


class PatchDiscriminator(nn.Module):
    """
    Spectral Normalized PatchGAN Discriminator.
    Evaluates real vs. generated image patches at fine spatial scales.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        def conv_block(in_c, out_c, stride=2):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            conv_block(in_channels, base_channels, stride=2),      # (B, 64, 128, 128)
            conv_block(base_channels, base_channels * 2, stride=2),  # (B, 128, 64, 64)
            conv_block(base_channels * 2, base_channels * 4, stride=2),# (B, 256, 32, 32)
            conv_block(base_channels * 4, base_channels * 8, stride=1),# (B, 512, 31, 31)
            spectral_norm(nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)) # (B, 1, 30, 30)
        )

    def forward(self, x):
        return self.net(x)