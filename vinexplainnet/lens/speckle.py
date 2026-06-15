from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveSpeckleSuppression(nn.Module):
    def __init__(self, sigma_s: float = 10.0, sigma_r: float = 0.1, kernel: int = 5) -> None:
        super().__init__()
        self.kernel = kernel
        self.sigma_r = sigma_r
        self.alpha_raw = nn.Parameter(torch.zeros(1))
        offset = torch.arange(kernel, dtype=torch.float32) - kernel // 2
        gy, gx = torch.meshgrid(offset, offset, indexing="ij")
        spatial = torch.exp(-(gx**2 + gy**2) / (2.0 * sigma_s**2))
        spatial = spatial.reshape(-1)
        self.spatial: torch.Tensor
        self.register_buffer("spatial", spatial)

    def bilateral(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad = self.kernel // 2
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        patches = F.unfold(padded, kernel_size=self.kernel)
        patches = patches.view(b, c, self.kernel * self.kernel, h * w)
        center = x.view(b, c, 1, h * w)
        rng = torch.exp(-((patches - center) ** 2) / (2.0 * self.sigma_r**2))
        weight = rng * self.spatial.view(1, 1, -1, 1)
        out = (weight * patches).sum(dim=2) / (weight.sum(dim=2) + 1e-8)
        return out.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filtered = self.bilateral(x)
        alpha = torch.sigmoid(self.alpha_raw)
        return (1.0 - alpha) * x + alpha * filtered
