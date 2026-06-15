from __future__ import annotations

import torch
import torch.nn.functional as F


def coherence_term(maps: torch.Tensor) -> torch.Tensor:
    dh = maps[:, :, 1:, :] - maps[:, :, :-1, :]
    dw = maps[:, :, :, 1:] - maps[:, :, :, :-1]
    return dh.pow(2).mean() + dw.pow(2).mean()


def sparsity_term(maps: torch.Tensor) -> torch.Tensor:
    return maps.abs().mean()


def _spatial_gradient(single: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gy = single[:, :, 1:, :] - single[:, :, :-1, :]
    gx = single[:, :, :, 1:] - single[:, :, :, :-1]
    gy = F.pad(gy, (0, 0, 0, 1))
    gx = F.pad(gx, (0, 1, 0, 0))
    return gy, gx


def boundary_term(maps: torch.Tensor, image: torch.Tensor, beta: float) -> torch.Tensor:
    explanation = maps.mean(dim=1, keepdim=True)
    target = F.interpolate(image, size=explanation.shape[-2:], mode="bilinear", align_corners=False)
    target = target.mean(dim=1, keepdim=True)
    ey, ex = _spatial_gradient(explanation)
    iy, ix = _spatial_gradient(target)
    return (ey - beta * iy).pow(2).mean() + (ex - beta * ix).pow(2).mean()


def explanation_quality(maps: torch.Tensor, image: torch.Tensor, beta: float) -> torch.Tensor:
    return coherence_term(maps) + sparsity_term(maps) + boundary_term(maps, image, beta)
