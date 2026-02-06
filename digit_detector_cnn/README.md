# Digit detector (0–9)

A PyTorch model that classifies handwritten digits 0–9 using the **MNIST** dataset.

## Dataset

- **Training**: ~48,000 images (80% of MNIST train)
- **Validation**: ~12,000 images (20% of MNIST train)
- **Test**: 10,000 images (MNIST test set)

MNIST is downloaded automatically to `data/` the first time you run training.

## Setup

```bash
cd digit_detector
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Options:

- `--epochs 10` – number of epochs (default: 10)
- `--batch-size 64` – batch size
- `--lr 0.001` – learning rate
- `--data-dir data` – where to store MNIST
- `--val-fraction 0.2` – fraction of train used for validation
- `--save checkpoint_best.pt` – path to save the best model
- `--no-augment` – turn off training augmentation (rotation/translate)

The best model (by validation accuracy) is saved to `checkpoint_best.pt`. At the end, accuracy on the test set is printed.

## Predict

Using the saved checkpoint:

```bash
# Predict on 5 random test samples
python predict.py --checkpoint checkpoint_best.pt

# Predict on your own 28×28 grayscale image
python predict.py --checkpoint checkpoint_best.pt --image path/to/digit.png
```

For custom images, use 28×28 grayscale; the script will resize if needed.

## Project layout

- `model.py` – CNN for digit classification
- `dataset.py` – MNIST loaders with train/val/test split
- `train.py` – training and evaluation script
- `predict.py` – inference script
- `requirements.txt` – PyTorch and torchvision
