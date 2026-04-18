# utils/visualize.py

import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import LATENT_DIM, NUM_ATTRIBUTES, SAMPLES_DIR, DEVICE
from data.dataset import denormalize


# ---------------------------------------------------------------------------
# Fixed noise and attributes — used throughout training for consistent grids
# ---------------------------------------------------------------------------

FIXED_Z = torch.randn(64, LATENT_DIM, device=DEVICE)

# Fixed attributes: 8 rows x 8 columns, each row gets a different attribute
# set to 1 so you can see how conditioning affects the output
FIXED_ATTRS = torch.zeros(64, NUM_ATTRIBUTES, device=DEVICE)
for i in range(8):
    FIXED_ATTRS[i * 8 : (i + 1) * 8, i * 5] = 1   # activate one attribute per row


def save_image_grid(generator, step, nrow=8):
    """
    Generates a grid of 64 images using fixed noise and attributes,
    and saves it to SAMPLES_DIR as a .png file.

    Args:
        generator : Generator model (in eval mode during generation)
        step      : current training step (used in filename)
        nrow      : number of images per row in the grid
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        fake_images = generator(FIXED_Z, FIXED_ATTRS)   # [64, 3, 64, 64]

    generator.train()

    # Denormalize from [-1, 1] to [0, 1] before saving
    fake_images = denormalize(fake_images).cpu()

    grid = vutils.make_grid(fake_images, nrow=nrow, padding=2, normalize=False)

    save_path = os.path.join(SAMPLES_DIR, f"step_{step:07d}.png")
    vutils.save_image(grid, save_path)

    print(f"  Saved sample grid → {save_path}")
    return save_path


def plot_losses(g_losses, d_losses, save=True):
    """
    Plots Generator and Discriminator losses over training steps.

    Args:
        g_losses : list of Generator loss values
        d_losses : list of Discriminator loss values
        save     : if True, saves the plot to SAMPLES_DIR
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(g_losses, label="Generator Loss",     alpha=0.8, linewidth=1)
    plt.plot(d_losses, label="Discriminator Loss", alpha=0.8, linewidth=1)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("WGAN-GP Training Losses")
    plt.legend()
    plt.tight_layout()

    if save:
        path = os.path.join(SAMPLES_DIR, "loss_curve.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved loss curve → {path}")

    plt.close()


def show_real_vs_fake(real_images, fake_images, n=8):
    """
    Side-by-side comparison of n real and n fake images.
    Useful for a quick visual sanity check during development.

    Args:
        real_images : [batch, 3, 64, 64] tensor, normalized to [-1, 1]
        fake_images : [batch, 3, 64, 64] tensor, normalized to [-1, 1]
        n           : number of image pairs to show
    """
    real = denormalize(real_images[:n]).cpu()
    fake = denormalize(fake_images[:n]).cpu()

    combined = torch.cat([real, fake], dim=0)   # [2n, 3, 64, 64]
    grid = vutils.make_grid(combined, nrow=n, padding=2, normalize=False)

    plt.figure(figsize=(16, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())   # CHW → HWC for matplotlib
    plt.axis("off")
    plt.title("Top row: Real    |    Bottom row: Fake")
    plt.tight_layout()

    path = os.path.join(SAMPLES_DIR, "real_vs_fake.png")
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved real vs fake → {path}")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models.generator import Generator

    G = Generator().to(DEVICE)
    save_image_grid(G, step=0)
    print("visualize.py sanity check passed.")
