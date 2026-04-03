# data/dataset.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS


def get_transforms():
    """
    Standard preprocessing pipeline for CelebA images.
    CelebA images are 178x218 — we center-crop then resize to IMAGE_SIZE.
    """
    return transforms.Compose([
        transforms.CenterCrop(178),        # crop out the face region
        transforms.Resize(IMAGE_SIZE),     # resize to 64x64
        transforms.ToTensor(),             # convert to tensor, scales to [0, 1]
        transforms.Normalize(              # scale to [-1, 1] to match Generator's tanh output
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])


def get_celeba_dataloader(split="train"):
    """
    Returns a DataLoader for the CelebA dataset.

    CelebA is downloaded automatically by torchvision on first run.
    It requires a Google Drive token — if auto-download fails, manual
    instructions are printed below.

    Args:
        split (str): "train", "valid", or "test"

    Returns:
        DataLoader, attribute_names (list of 40 strings)
    """
    dataset = datasets.CelebA(
        root=DATA_ROOT,
        split=split,
        target_type="attr",          # we want the 40 binary attribute labels
        transform=get_transforms(),
        download=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True if split == "train" else False,
        num_workers=NUM_WORKERS,
        pin_memory=True,             # speeds up CPU → GPU transfer
        drop_last=True               # drop the last incomplete batch for stable training
    )

    return dataloader, dataset.attr_names


def denormalize(tensor):
    """
    Reverses the [-1, 1] normalization back to [0, 1] for visualization.

    Args:
        tensor: image tensor normalized to [-1, 1]

    Returns:
        tensor in [0, 1] range
    """
    return (tensor * 0.5) + 0.5


# ---------------------------------------------------------------------------
# Manual download fallback instructions (printed if torchvision fails)
# ---------------------------------------------------------------------------
MANUAL_DOWNLOAD_INSTRUCTIONS = """
CelebA auto-download failed (Google Drive rate limits this frequently).

Manual steps:
1. Go to: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Download: img_align_celeba.zip, list_attr_celeba.txt, list_eval_partition.txt
3. Extract and place at:
   ./data/celeba/img_align_celeba/     ← all .jpg images go here
   ./data/celeba/list_attr_celeba.txt
   ./data/celeba/list_eval_partition.txt
4. Re-run — torchvision will detect the files and skip the download.
"""

if __name__ == "__main__":
    try:
        loader, attr_names = get_celeba_dataloader(split="train")
        images, attrs = next(iter(loader))
        print(f"Batch shape   : {images.shape}")       # [128, 3, 64, 64]
        print(f"Attr shape    : {attrs.shape}")         # [128, 40]
        print(f"Pixel range   : [{images.min():.2f}, {images.max():.2f}]")
        print(f"Attributes    : {attr_names}")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error: {e}")
        print(MANUAL_DOWNLOAD_INSTRUCTIONS)
