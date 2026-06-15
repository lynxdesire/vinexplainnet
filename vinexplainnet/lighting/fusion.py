from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vinexplainnet.lighting.streams import GlobalStream, LocalStream, RegionalStream
from vinexplainnet.treatment.schema import ModelSpec


class HierarchicalAttentionFusion(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        _, c2, c3, c4 = spec.channels
        self.use_local = spec.use_local
        self.use_regional = spec.use_regional
        self.use_global = spec.use_global
        self.use_fusion = spec.use_fusion
        self.local = LocalStream(c4) if spec.use_local else None
        self.regional = RegionalStream(c3) if spec.use_regional else None
        self.glob = GlobalStream(c2, spec.attention_dim) if spec.use_global else None
        self.weights = nn.Parameter(torch.zeros(3))

    def streams(
        self, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = f4.shape[-2:]
        zero = torch.zeros(f4.shape[0], 1, size[0], size[1], device=f4.device, dtype=f4.dtype)
        a_local = self.local(f4) if self.local is not None else zero
        if self.regional is not None:
            a_regional = F.interpolate(
                self.regional(f3), size=size, mode="bilinear", align_corners=False
            )
        else:
            a_regional = zero
        if self.glob is not None:
            a_global = F.interpolate(self.glob(f2), size=size, mode="bilinear", align_corners=False)
        else:
            a_global = zero
        return a_local, a_regional, a_global

    def fuse(
        self, a_local: torch.Tensor, a_regional: torch.Tensor, a_global: torch.Tensor
    ) -> torch.Tensor:
        coeff = torch.softmax(self.weights, dim=0)
        return coeff[0] * a_local + coeff[1] * a_regional + coeff[2] * a_global

    def modulate(self, f4: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        return f4 * fused + f4

    def forward(
        self, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_local, a_regional, a_global = self.streams(f2, f3, f4)
        fused = self.fuse(a_local, a_regional, a_global)
        attended = self.modulate(f4, fused)
        return attended, fused
