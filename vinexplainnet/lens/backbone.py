from __future__ import annotations

import torch
from torch import nn

from vinexplainnet.lens.speckle import AdaptiveSpeckleSuppression
from vinexplainnet.treatment.schema import AssSpec, ModelSpec


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.depthwise(x)
        h = self.pointwise(h)
        h = self.norm(h)
        return self.act(h)


class EfficientBackbone(nn.Module):
    def __init__(self, spec: ModelSpec, ass: AssSpec, in_ch: int = 1) -> None:
        super().__init__()
        c1, c2, c3, c4 = spec.channels
        self.channels = spec.channels
        self.speckle: AdaptiveSpeckleSuppression | None
        if spec.use_speckle:
            self.speckle = AdaptiveSpeckleSuppression(ass.sigma_s, ass.sigma_r, ass.kernel)
        else:
            self.speckle = None
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, spec.stem, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(spec.stem),
            nn.SiLU(inplace=True),
        )
        self.stage1 = DepthwiseSeparable(spec.stem, c1, stride=1)
        self.stage2 = DepthwiseSeparable(c1, c2, stride=2)
        self.stage3 = DepthwiseSeparable(c2, c3, stride=2)
        self.stage4 = DepthwiseSeparable(c3, c4, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.speckle is not None:
            x = self.speckle(x)
        s = self.stem(x)
        f1 = self.stage1(s)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return f1, f2, f3, f4
