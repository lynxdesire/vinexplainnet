from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_size_mb(model: nn.Module, bytes_per: float = 4.0) -> float:
    return count_params(model) * bytes_per / 1.0e6


def count_flops(model: nn.Module, sample: torch.Tensor) -> float:
    tally = {"flops": 0.0}

    def conv_hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        conv = module
        assert isinstance(conv, nn.Conv2d)
        out_h, out_w = output.shape[-2], output.shape[-1]
        kh, kw = conv.kernel_size
        cin = conv.in_channels / conv.groups
        tally["flops"] += 2.0 * conv.out_channels * cin * kh * kw * out_h * out_w

    def linear_hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        linear = module
        assert isinstance(linear, nn.Linear)
        tally["flops"] += 2.0 * linear.in_features * linear.out_features

    handles = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            handles.append(layer.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(sample)
    for handle in handles:
        handle.remove()
    return tally["flops"]


def measure_latency(model: nn.Module, sample: torch.Tensor, runs: int = 20) -> float:
    model.eval()
    with torch.no_grad():
        model(sample)
        start = time.perf_counter()
        for _ in range(runs):
            model(sample)
        elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / runs


def profile_footprint(model: nn.Module, sample: torch.Tensor, runs: int = 20) -> dict[str, float]:
    return {
        "params_m": count_params(model) / 1.0e6,
        "size_mb": model_size_mb(model),
        "gflops": count_flops(model, sample) / 1.0e9,
        "latency_ms": measure_latency(model, sample, runs),
    }
