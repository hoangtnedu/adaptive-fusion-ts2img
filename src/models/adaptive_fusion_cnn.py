from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightCNNBranch(nn.Module):
    """A lightweight 2D CNN branch for one image representation."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(64, feature_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class AdaptiveFusionCNN(nn.Module):
    """Multi-branch lightweight CNN for 2D time-series representations.

    Supported fusion modes:
        adaptive_gating:
            z = concat(f_1, ..., f_M)
            alpha = softmax(gate(z))
            F = sum_m alpha_m * f_m
        feature_concat:
            F = concat(f_1, ..., f_M)
        feature_mean:
            F = mean(f_1, ..., f_M)

    Input shape:
        x: (B, M, 1, H, W)

    Output:
        logits: (B, C)
        alpha:  (B, M). For non-gating fusion, alpha is uniform and saved only
                for reporting compatibility.
    """

    def __init__(
        self,
        num_classes: int,
        num_representations: int = 4,
        feature_dim: int = 128,
        gate_hidden: int = 128,
        dropout: float = 0.3,
        fusion: Literal["adaptive_gating", "feature_concat", "feature_mean"] = "adaptive_gating",
    ):
        super().__init__()
        if num_representations < 1:
            raise ValueError("num_representations must be >= 1")
        if fusion not in {"adaptive_gating", "feature_concat", "feature_mean"}:
            raise ValueError(
                "fusion must be one of: adaptive_gating, feature_concat, feature_mean"
            )

        self.num_representations = num_representations
        self.feature_dim = feature_dim
        self.fusion = fusion

        self.branches = nn.ModuleList(
            [LightCNNBranch(feature_dim=feature_dim) for _ in range(num_representations)]
        )

        if fusion == "adaptive_gating":
            self.gate = nn.Sequential(
                nn.Linear(num_representations * feature_dim, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, num_representations),
                nn.Softmax(dim=1),
            )
            classifier_input_dim = feature_dim
        elif fusion == "feature_concat":
            self.gate = None
            classifier_input_dim = num_representations * feature_dim
        else:  # feature_mean
            self.gate = None
            classifier_input_dim = feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, M, 1, H, W)
        branch_features = [self.branches[m](x[:, m]) for m in range(self.num_representations)]
        feature_stack = torch.stack(branch_features, dim=1)  # (B, M, D)

        if self.fusion == "adaptive_gating":
            z = feature_stack.flatten(start_dim=1)
            alpha = self.gate(z)
            fused = torch.sum(feature_stack * alpha.unsqueeze(-1), dim=1)
        elif self.fusion == "feature_concat":
            alpha = torch.full(
                (x.size(0), self.num_representations),
                1.0 / self.num_representations,
                dtype=x.dtype,
                device=x.device,
            )
            fused = feature_stack.flatten(start_dim=1)
        else:  # feature_mean
            alpha = torch.full(
                (x.size(0), self.num_representations),
                1.0 / self.num_representations,
                dtype=x.dtype,
                device=x.device,
            )
            fused = torch.mean(feature_stack, dim=1)

        logits = self.classifier(fused)
        return logits, alpha
