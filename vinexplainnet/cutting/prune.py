from __future__ import annotations

import torch


def channel_importance(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    grad_mag = gradients.abs().mean(dim=(0, 2, 3))
    act_norm = activations.pow(2).sum(dim=(2, 3)).sqrt().mean(dim=0)
    return grad_mag * act_norm


def keep_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    threshold = ratio * scores.max()
    return scores >= threshold


def structured_prune(weight: torch.Tensor, scores: torch.Tensor, ratio: float) -> torch.Tensor:
    mask = keep_mask(scores, ratio)
    return weight[mask]


def retained_fraction(scores: torch.Tensor, ratio: float) -> float:
    mask = keep_mask(scores, ratio)
    return float(mask.float().mean())
