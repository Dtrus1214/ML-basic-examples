"""
Create a small PNG dataset from MNIST with 80% train, 10% val, 10% test.

Run once to create:
  data/train/0..9/*.png   (80% of dataset)
  data/val/0..9/*.png     (10%)
  data/test/0..9/*.png   (10%)

Then train with: python train.py
"""

import argparse
from pathlib import Path

from torchvision import datasets, transforms

DATA_DIR = "data"
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Root folder for data/")
    parser.add_argument(
        "--per-class",
        type=int,
        default=100,
        help="Total images per digit (0-9); split 80%% train, 10%% val, 10%% test",
    )
    args = parser.parse_args()

    root = Path(args.root)
    n_total = args.per_class  # per digit
    n_train = int(n_total * TRAIN_FRAC)
    n_val = int(n_total * VAL_FRAC)
    n_test = int(n_total * TEST_FRAC)
    # Use any remainder for train
    n_train += n_total - (n_train + n_val + n_test)

    mnist_train = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())

    # Collect indices per digit (all from train set, then split 80/10/10)
    by_digit = {d: [] for d in range(10)}
    for i, (_, label) in enumerate(mnist_train):
        if len(by_digit[label]) < n_total:
            by_digit[label].append(i)

    to_pil = transforms.ToPILImage()
    count_train, count_val, count_test = 0, 0, 0

    for digit in range(10):
        indices = by_digit[digit]
        n_t, n_v, n_te = n_train, n_val, n_test
        # If we have fewer than n_total, split proportionally
        if len(indices) < n_total:
            n_t = int(len(indices) * TRAIN_FRAC)
            n_te = int(len(indices) * TEST_FRAC)
            n_v = len(indices) - n_t - n_te
        i0, i1 = 0, n_t
        i2 = i1 + n_v
        i3 = i2 + n_te

        out_train = root / "train" / str(digit)
        out_train.mkdir(parents=True, exist_ok=True)
        for j, idx in enumerate(indices[i0:i1]):
            img, _ = mnist_train[idx]
            to_pil(img.squeeze(0)).save(out_train / f"{digit}_{j:02d}.png")
            count_train += 1

        out_val = root / "val" / str(digit)
        out_val.mkdir(parents=True, exist_ok=True)
        for j, idx in enumerate(indices[i1:i2]):
            img, _ = mnist_train[idx]
            to_pil(img.squeeze(0)).save(out_val / f"{digit}_{j:02d}.png")
            count_val += 1

        out_test = root / "test" / str(digit)
        out_test.mkdir(parents=True, exist_ok=True)
        for j, idx in enumerate(indices[i2:i3]):
            img, _ = mnist_train[idx]
            to_pil(img.squeeze(0)).save(out_test / f"{digit}_{j:02d}.png")
            count_test += 1

    total = count_train + count_val + count_test
    print(f"Saved {total} PNGs under {root}/: train={count_train} (80%), val={count_val} (10%), test={count_test} (10%).")
    print("Run: python train.py")


if __name__ == "__main__":
    main()
