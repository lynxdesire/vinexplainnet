from __future__ import annotations

import torch
from torch import nn


def quantize_tensor(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, float, float]:
    w_min = float(weight.min())
    w_max = float(weight.max())
    span = w_max - w_min
    levels = float(2**bits - 1)
    scale = span / levels if span > 0 else 1.0
    codes = torch.round((weight - w_min) / scale)
    return codes, scale, w_min


def dequantize(codes: torch.Tensor, scale: float, w_min: float) -> torch.Tensor:
    return codes * scale + w_min


class MixedPrecisionQuantizer:
    def __init__(self, bits: int = 8) -> None:
        self.bits = bits

    def _sensitive(self, name: str, names: list[str]) -> bool:
        return name == names[0] or name == names[-1]

    def quantize(self, model: nn.Module) -> dict[str, float]:
        weight_names = [n for n, _ in model.named_parameters() if n.endswith("weight")]
        fp32_bytes = 0.0
        quant_bytes = 0.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not name.endswith("weight"):
                    continue
                elements = param.numel()
                fp32_bytes += elements * 4
                if self._sensitive(name, weight_names):
                    quant_bytes += elements * 2
                    continue
                codes, scale, w_min = quantize_tensor(param.data, self.bits)
                param.data = dequantize(codes, scale, w_min)
                quant_bytes += elements * (self.bits / 8.0)
        return {
            "fp32_bytes": fp32_bytes,
            "quant_bytes": quant_bytes,
            "ratio": quant_bytes / fp32_bytes if fp32_bytes > 0 else 1.0,
        }
