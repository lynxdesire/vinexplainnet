from __future__ import annotations

import torch


def class_balance_weights(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=num_classes).float()
    freq = counts / counts.sum().clamp_min(1.0)
    inverse = 1.0 / freq.clamp_min(1e-6)
    return inverse / inverse.sum() * num_classes


def weighted_bce(
    logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    onehot = torch.zeros_like(logits)
    onehot.scatter_(1, targets.view(-1, 1), 1.0)
    probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    pos = pos_weight.view(1, -1) * onehot * torch.log(probs)
    neg = (1.0 - onehot) * torch.log(1.0 - probs)
    per_class = -(pos + neg)
    return per_class.mean()
