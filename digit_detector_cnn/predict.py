"""
Run digit prediction using a trained model.

Usage:
  python predict.py --checkpoint checkpoint_best.pt
  python predict.py --checkpoint checkpoint_best.pt --image path/to/28x28_grayscale.png
"""

import argparse
import torch
from torchvision import transforms

from model import get_model

# Same normalization as in dataset.py (MNIST)
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = get_model(num_classes=10)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_image(model, image_tensor: torch.Tensor, device: torch.device) -> int:
    """Predict digit (0-9) for a single image tensor (1, 1, 28, 28)."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        pred = logits.argmax(dim=1).item()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pt", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to 28x28 grayscale image (optional)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert("L")
        if img.size != (28, 28):
            img = img.resize((28, 28))
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])
        x = tf(img).unsqueeze(0)  # (1, 1, 28, 28)
        digit = predict_image(model, x, device)
        print(f"Predicted digit: {digit}")
    else:
        # Demo: load a few test samples from MNIST
        from torchvision import datasets
        test_set = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
        for i in range(5):
            img, label = test_set[i]
            img_norm = (img - MNIST_MEAN) / MNIST_STD
            pred = predict_image(model, img_norm.unsqueeze(0), device)
            print(f"Sample {i}: true={label}, predicted={pred}")


if __name__ == "__main__":
    main()
