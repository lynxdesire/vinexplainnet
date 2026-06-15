from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PredictionHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def pool(self, attended: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(attended, 1).flatten(1)

    def forward(self, attended: torch.Tensor) -> torch.Tensor:
        h = self.pool(attended)
        h = F.relu(self.fc1(h))
        return self.fc2(h)
