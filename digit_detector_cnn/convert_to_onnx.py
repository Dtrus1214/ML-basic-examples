"""
Convert a trained .pt checkpoint to ONNX format.

Usage:
  python convert_to_onnx.py --checkpoint checkpoint_best.pt --output model.onnx
  python convert_to_onnx.py --checkpoint checkpoint_best.pt --output model.onnx --dynamic  # batch dimension dynamic

If both model.onnx and model.onnx.data are created: keep both when deploying; the .onnx
file references the weights in .onnx.data. Some runtimes write external data for larger models.
"""

import argparse
import torch

from model import get_model


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch digit model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pt", help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (use 17+ to avoid version converter errors)")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic batch size (batch dim as variable)")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = get_model(num_classes=10)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Example input: (batch, channels, height, width)
    dummy = torch.randn(1, 1, 28, 28)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
