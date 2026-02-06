"""
Dataset loading for digit detection: train, validation, and test.

Uses MNIST (handwritten digits 0-9) from torchvision.
- Training set: 80% of MNIST train (used for training)
- Validation set: 20% of MNIST train (used for validation during training)
- Test set: full MNIST test set (used for final evaluation)
"""

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


# Default data directory (MNIST will be downloaded here if missing)
DEFAULT_DATA_DIR = "data"


def get_mnist_transforms(augment_train: bool = True):
    """Return transforms for MNIST. Optionally add light augmentation for training."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean, std

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if augment_train:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform


def get_mnist_datasets(
    root: str = DEFAULT_DATA_DIR,
    download: bool = True,
    val_fraction: float = 0.2,
    augment_train: bool = True,
):
    """
    Load MNIST and split into train / validation / test.

    Args:
        root: Directory for dataset (will create root/mnist or similar).
        download: Whether to download MNIST if not present.
        val_fraction: Fraction of training data to use for validation (0.0 to 1.0).
        augment_train: Whether to use data augmentation on training set.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    train_tf, test_tf = get_mnist_transforms(augment_train=augment_train)

    full_train = datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=train_tf,
    )
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=test_tf,
    )

    n = len(full_train)
    n_val = int(n * val_fraction)
    n_train = n - n_val
    train_dataset, val_dataset = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    batch_size: int = 64,
    num_workers: int = 0,
    root: str = DEFAULT_DATA_DIR,
    val_fraction: float = 0.2,
    augment_train: bool = True,
):
    """
    Return DataLoaders for train, validation, and test.

    Args:
        batch_size: Batch size for all loaders.
        num_workers: Number of worker processes (0 for Windows compatibility).
        root: Dataset root directory.
        val_fraction: Fraction of train data used for validation.
        augment_train: Whether to augment training data.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds, val_ds, test_ds = get_mnist_datasets(
        root=root,
        download=True,
        val_fraction=val_fraction,
        augment_train=augment_train,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
