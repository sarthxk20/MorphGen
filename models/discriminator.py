# models/discriminator.py

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import NUM_ATTRIBUTES, CHANNELS, FEATURES_D, IMAGE_SIZE


class DiscriminatorBlock(nn.Module):
    """
    A single downsampling block used repeatedly in the Discriminator.
    Each block halves the spatial resolution of the feature map.

    Conv2d → InstanceNorm → LeakyReLU

    Note: We use InstanceNorm instead of BatchNorm here.
    WGAN-GP's gradient penalty requires per-sample gradients,
    which BatchNorm breaks by coupling samples in a batch together.
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """
    Conditional DCGAN Discriminator (Critic in WGAN terminology).

    Takes an image and an attribute vector, and outputs a scalar score.
    Higher score = more likely to be real.
    No sigmoid — WGAN-GP uses raw scores (Wasserstein distance estimate).

    Conditioning: the attribute vector is projected to a spatial map
    and concatenated channel-wise with the input image.

    Architecture:
        64x64 → 32x32 → 16x16 → 8x8 → 4x4 → scalar

    Input:
        image : [batch, 3, 64, 64]
        attrs : [batch, NUM_ATTRIBUTES]

    Output:
        score : [batch, 1]
    """
    def __init__(self):
        super().__init__()

        fd = FEATURES_D  # 64

        # Project attribute vector → [batch, NUM_ATTRIBUTES, 64, 64]
        # so it can be concatenated channel-wise with the image
        self.attr_embed = nn.Sequential(
            nn.Linear(NUM_ATTRIBUTES, IMAGE_SIZE * IMAGE_SIZE),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Input channels = image channels + 1 (projected attribute map)
        in_channels = CHANNELS + 1

        # 64x64 → 32x32  (no norm on first layer — standard practice)
        self.block1 = DiscriminatorBlock(in_channels, fd,      normalize=False)
        # 32x32 → 16x16
        self.block2 = DiscriminatorBlock(fd,          fd * 2)
        # 16x16 → 8x8
        self.block3 = DiscriminatorBlock(fd * 2,      fd * 4)
        # 8x8 → 4x4
        self.block4 = DiscriminatorBlock(fd * 4,      fd * 8)

        # 4x4 → scalar score
        self.output = nn.Conv2d(fd * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self._initialize_weights()

    def forward(self, image, attrs):
        """
        Args:
            image : [batch, 3, 64, 64]
            attrs : [batch, NUM_ATTRIBUTES]

        Returns:
            score : [batch, 1]
        """
        batch = image.size(0)

        # Project attributes → [batch, 1, 64, 64]
        attr_map = self.attr_embed(attrs.float())                        # [batch, 64*64]
        attr_map = attr_map.view(batch, 1, IMAGE_SIZE, IMAGE_SIZE)       # [batch, 1, 64, 64]

        # Concatenate along channel dimension
        x = torch.cat([image, attr_map], dim=1)                         # [batch, 4, 64, 64]

        x = self.block1(x)   # [batch, fd,    32, 32]
        x = self.block2(x)   # [batch, fd*2,  16, 16]
        x = self.block3(x)   # [batch, fd*4,  8,  8]
        x = self.block4(x)   # [batch, fd*8,  4,  4]
        x = self.output(x)   # [batch, 1,     1,  1]

        return x.view(batch, 1)   # [batch, 1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import DEVICE, LATENT_DIM

    D = Discriminator().to(DEVICE)
    print(D)
    print(f"\nTotal parameters: {sum(p.numel() for p in D.parameters()):,}")

    images = torch.randn(4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    attrs  = torch.randint(0, 2, (4, NUM_ATTRIBUTES)).float().to(DEVICE)
    scores = D(images, attrs)

    print(f"\nInput  image  : {images.shape}")
    print(f"Input  attrs  : {attrs.shape}")
    print(f"Output scores : {scores.shape}")    # expect [4, 1]
    print(f"Score range   : [{scores.min():.2f}, {scores.max():.2f}]")  # unbounded — no sigmoid
