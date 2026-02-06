"""
Load digit dataset from raw PNG images in folder-per-class layout.

Expected structure (about 100 PNGs total, e.g. 70 train, 15 val, 15 test):

  data/
    train/  0/  *.png
            1/  *.png
            ...
            9/  *.png
    val/    0/  *.png
            ...
            9/  *.png
    test/   0/  *.png
            ...
            9/  *.png

Images are resized to 28x28 and converted to grayscale.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

DATA_DIR = "data"
IMAGE_SIZE = 28
DIGITS = list(range(10))


def get_transform():
    """Resize to 28x28, grayscale, normalize to [0,1]."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST-style normalization
    ])


class PNGDigitDataset(Dataset):
    """Dataset of digit PNGs from folders data/{split}/{0..9}/*.png."""

    def __init__(self, root: str, split: str, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform or get_transform()
        self.samples = []  # list of (path, label)
        for digit in DIGITS:
            folder = self.root / split / str(digit)
            if not folder.exists():
                continue
            for path in folder.glob("*.png"):
                self.samples.append((str(path), digit))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(root: str = DATA_DIR, batch_size: int = 16):
    """Return train, val, test DataLoaders from PNG folders."""
    train_ds = PNGDigitDataset(root, "train", transform=get_transform())
    val_ds = PNGDigitDataset(root, "val", transform=get_transform())
    test_ds = PNGDigitDataset(root, "test", transform=get_transform())

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader
