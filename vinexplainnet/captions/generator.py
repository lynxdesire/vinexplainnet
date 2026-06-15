from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ExplanationGenerator(nn.Module):
    def __init__(self, attended_channels: int, low_level_channels: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_proj = nn.Conv2d(attended_channels, num_classes, kernel_size=1)
        self.low_proj = nn.Conv2d(low_level_channels, num_classes, kernel_size=1)
        self.refine = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1)

    def class_weights(self, attended: torch.Tensor) -> torch.Tensor:
        logits = self.class_proj(attended)
        b, c, h, w = logits.shape
        flat = logits.view(b, c, h * w)
        weights = torch.softmax(flat, dim=-1)
        return weights.view(b, c, h, w)

    def forward(
        self, attended: torch.Tensor, fused: torch.Tensor, low_level: torch.Tensor
    ) -> torch.Tensor:
        weights = self.class_weights(attended)
        coarse = F.relu(weights * fused)
        size = low_level.shape[-2:]
        coarse_up = F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
        low = self.low_proj(low_level)
        merged = torch.cat([coarse_up, low], dim=1)
        return F.relu(self.refine(merged))
