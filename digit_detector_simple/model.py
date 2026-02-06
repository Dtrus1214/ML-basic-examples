"""
Simple feedforward (fully connected) network for digit classification (0-9).

No convolutions: image is flattened to a vector, then a few dense layers.
Input: 28x28 grayscale → 784 features.
"""

import torch
import torch.nn as nn

INPUT_SIZE = 28 * 28  # 784


class DigitMLP(nn.Module):
    """
    Feedforward neural network for digit recognition.
    Input: (batch, 1, 28, 28) or (batch, 784).
    Output: (batch, 10) logits for classes 0-9.
    """

    def __init__(self, num_classes: int = 10, hidden: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)


def get_model(num_classes: int = 10, hidden: int = 128) -> nn.Module:
    """Return the feedforward digit model."""
    return DigitMLP(num_classes=num_classes, hidden=hidden)
