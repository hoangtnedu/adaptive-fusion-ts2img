from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Compact 1D-CNN baseline for raw time-series classification."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        hidden_channels: tuple[int, int] = (32, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        c1, c2 = hidden_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # x: (B, 1, T)
        features = self.net(x)
        logits = self.classifier(features)
        return logits, None
