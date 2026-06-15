from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Frame:
    logits: torch.Tensor
    probs: torch.Tensor
    fused: torch.Tensor
    explanations: torch.Tensor
    low_level: torch.Tensor


@dataclass
class Batch:
    images: torch.Tensor
    labels: torch.Tensor


@dataclass
class TakeRecord:
    step: int
    losses: dict[str, float]
