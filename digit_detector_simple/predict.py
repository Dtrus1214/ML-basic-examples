"""
Predict digit (0-9) from a PNG image using the trained feedforward model.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import get_model

IMAGE_SIZE = 28
NORMALIZE_MEAN, NORMALIZE_STD = 0.1307, 0.3081


def load_model(checkpoint_path: str, device: torch.device, hidden: int = 128):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = get_model(num_classes=10, hidden=hidden)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_image(model, image_path: str, device: torch.device) -> int:
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((NORMALIZE_MEAN,), (NORMALIZE_STD,)),
    ])
    img = Image.open(image_path).convert("L")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    return logits.argmax(1).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="model_best.pt")
    parser.add_argument("--image", type=str, required=True, help="Path to digit PNG")
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, hidden=args.hidden)
    digit = predict_image(model, args.image, device)
    print(f"Predicted digit: {digit}")


if __name__ == "__main__":
    main()
