# data/dataset.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS


def get_transforms():
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


class CelebAKaggle(Dataset):
    """
    CelebA dataset loader using Kaggle CSV files.

    Expected folder structure:
        DATA_ROOT/celeba/
            img_align_celeba/
                img_align_celeba/
                    000001.jpg
                    ...
            list_attr_celeba.csv
            list_eval_partition.csv
    """
    def __init__(self, root, split='train', transform=None):
        self.img_dir = os.path.join(root, 'img_align_celeba', 'img_align_celeba')

        attrs = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv'))
        parts = pd.read_csv(os.path.join(root, 'list_eval_partition.csv'))

        split_map = {'train': 0, 'valid': 1, 'test': 2}
        mask = parts['partition'] == split_map[split]

        all_filenames   = attrs['image_id'][mask].values
        all_attr_vals   = attrs.drop(columns=['image_id'])[mask].values
        self.attr_names = list(attrs.columns[1:])
        self.transform  = transform

        # Filter to only files that exist on disk
        valid = [
            i for i, f in enumerate(all_filenames)
            if os.path.exists(os.path.join(self.img_dir, f))
        ]
        self.filenames = all_filenames[valid]
        self.attr_vals = all_attr_vals[valid]

        print(f"CelebA [{split}]: {len(self.filenames):,} images found.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(
            os.path.join(self.img_dir, self.filenames[idx])
        ).convert('RGB')

        if self.transform:
            img = self.transform(img)

        attrs = torch.tensor(
            (self.attr_vals[idx] > 0).astype('float32')
        )
        return img, attrs


def get_celeba_dataloader(split='train'):
    dataset = CelebAKaggle(
        root=os.path.join(DATA_ROOT, 'celeba'),
        split=split,
        transform=get_transforms()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == 'train'),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    return dataloader, dataset.attr_names


def denormalize(tensor):
    return (tensor * 0.5) + 0.5


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loader, attr_names = get_celeba_dataloader(split='train')
    images, attrs = next(iter(loader))
    print(f"Batch shape  : {images.shape}")
    print(f"Attr shape   : {attrs.shape}")
    print(f"Pixel range  : [{images.min():.2f}, {images.max():.2f}]")
    print(f"Batches/epoch: {len(loader)}")
