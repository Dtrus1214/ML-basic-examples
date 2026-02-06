"""
Train the simple feedforward digit model on PNG dataset.

Expects data in data/train/0..9, data/val/0..9, data/test/0..9.
Run create_sample_data.py first to generate ~100 PNGs from MNIST if needed.
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders, DATA_DIR
from model import get_model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader) if len(loader) else 0, correct / total if total else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--save", type=str, default="model_best.pt")
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(root=args.data_dir, batch_size=args.batch_size)
    if len(train_loader.dataset) == 0:
        print("No training data found. Create folders data/train/0..9 and add PNGs, or run: python create_sample_data.py")
        return

    model = get_model(num_classes=10, hidden=args.hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}  train loss={train_loss:.4f} acc={train_acc:.4f}  val loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, args.save)
            print(f"  -> saved {args.save}")

    print("\nTest set:")
    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  loss={test_loss:.4f}  accuracy={test_acc:.4f}")


if __name__ == "__main__":
    main()
