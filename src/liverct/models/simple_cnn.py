from __future__ import annotations

import torch
from torch import nn


class SimpleCNN2D(nn.Module):
    """
    Small 2D CNN baseline for one-channel 256x256 images.

    The model returns one logit for binary classification with
    BCEWithLogitsLoss.
    """

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
