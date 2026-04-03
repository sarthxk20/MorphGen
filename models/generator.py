# models/generator.py

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import LATENT_DIM, NUM_ATTRIBUTES, CHANNELS, FEATURES_G


class GeneratorBlock(nn.Module):
    """
    A single upsampling block used repeatedly in the Generator.
    Each block doubles the spatial resolution of the feature map.

    ConvTranspose2d → BatchNorm → ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size, stride, padding,
                bias=False           # bias is redundant when using BatchNorm
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    Conditional DCGAN Generator.

    Takes a noise vector z and a one-hot attribute vector,
    concatenates them, and progressively upsamples to produce
    a 3 x 64 x 64 RGB image.

    Architecture:
        (LATENT_DIM + NUM_ATTRIBUTES) → 4x4 → 8x8 → 16x16 → 32x32 → 64x64

    Input:
        z    : [batch, LATENT_DIM]       — random noise
        attrs: [batch, NUM_ATTRIBUTES]   — binary attribute vector (e.g., smiling=1, glasses=0)

    Output:
        image: [batch, 3, 64, 64]        — pixel values in [-1, 1]
    """
    def __init__(self):
        super().__init__()

        self.input_dim = LATENT_DIM + NUM_ATTRIBUTES  # 128 + 40 = 168

        fg = FEATURES_G  # 64

        # Project and reshape: (168,) → (fg*16, 4, 4)
        self.project = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, fg * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(fg * 16),
            nn.ReLU(inplace=True)
        )
        # 4x4 → 8x8
        self.block1 = GeneratorBlock(fg * 16, fg * 8)
        # 8x8 → 16x16
        self.block2 = GeneratorBlock(fg * 8,  fg * 4)
        # 16x16 → 32x32
        self.block3 = GeneratorBlock(fg * 4,  fg * 2)
        # 32x32 → 64x64
        self.block4 = GeneratorBlock(fg * 2,  fg)

        # Final layer: no BatchNorm, Tanh to produce [-1, 1] output
        self.output = nn.Sequential(
            nn.ConvTranspose2d(fg, CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self, z, attrs):
        """
        Args:
            z     : [batch, LATENT_DIM]
            attrs : [batch, NUM_ATTRIBUTES] — values are 0 or 1

        Returns:
            [batch, 3, 64, 64]
        """
        # Concatenate noise and attribute vector
        x = torch.cat([z, attrs.float()], dim=1)       # [batch, 168]

        # Reshape to 4D for ConvTranspose: [batch, 168, 1, 1]
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = self.project(x)   # [batch, fg*16, 4, 4]
        x = self.block1(x)    # [batch, fg*8,  8, 8]
        x = self.block2(x)    # [batch, fg*4,  16, 16]
        x = self.block3(x)    # [batch, fg*2,  32, 32]
        x = self.block4(x)    # [batch, fg,    64, 64]  (note: block4 doesn't double — see output)
        x = self.output(x)    # [batch, 3,     64, 64]
        return x

    def _initialize_weights(self):
        """
        DCGAN paper recommends initializing conv weights from
        N(0, 0.02) and BatchNorm weights from N(1, 0.02).
        This stabilizes early training.
        """
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import DEVICE

    G = Generator().to(DEVICE)
    print(G)
    print(f"\nTotal parameters: {sum(p.numel() for p in G.parameters()):,}")

    z     = torch.randn(4, LATENT_DIM).to(DEVICE)
    attrs = torch.randint(0, 2, (4, NUM_ATTRIBUTES)).to(DEVICE)
    out   = G(z, attrs)
    print(f"\nInput  z     : {z.shape}")
    print(f"Input  attrs : {attrs.shape}")
    print(f"Output image : {out.shape}")    # expect [4, 3, 64, 64]
    print(f"Pixel range  : [{out.min():.2f}, {out.max():.2f}]")  # expect [-1, 1]
