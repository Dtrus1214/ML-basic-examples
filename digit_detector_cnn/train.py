"""
Train the digit detector model (0-9) on MNIST.

Downloads MNIST automatically. Splits into train / validation; reports test accuracy at the end.
Saves the best model (by validation accuracy) to checkpoint_best.pt.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders, DEFAULT_DATA_DIR
from model import get_model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(train_loader), correct / total


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
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Train digit detector (0-9) on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Dataset root directory")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of train data for validation")
    parser.add_argument("--save", type=str, default="checkpoint_best.pt", help="Path to save best model")
    parser.add_argument("--no-augment", action="store_true", help="Disable training data augmentation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets (MNIST will be downloaded if needed)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,
        root=args.data_dir,
        val_fraction=args.val_fraction,
        augment_train=not args.no_augment,
    )

    model = get_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                args.save,
            )
            print(f"  -> Saved best model to {args.save}")

    print("\nEvaluating on test set...")
    checkpoint = torch.load(args.save, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
