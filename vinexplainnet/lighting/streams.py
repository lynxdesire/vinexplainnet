from __future__ import annotations

import math

import torch
from torch import nn


class LocalStream(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.project = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, f4: torch.Tensor) -> torch.Tensor:
        h = self.depthwise(f4)
        return torch.sigmoid(self.project(h))


class RegionalStream(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dilated = nn.Conv2d(
            channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False
        )
        self.project = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, f3: torch.Tensor) -> torch.Tensor:
        h = self.dilated(f3)
        return torch.sigmoid(self.project(h))


class GlobalStream(nn.Module):
    def __init__(self, channels: int, attention_dim: int) -> None:
        super().__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(channels, attention_dim)
        self.key = nn.Linear(channels, attention_dim)
        self.value = nn.Linear(channels, attention_dim)
        self.out = nn.Linear(attention_dim, 1)

    def forward(self, f2: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f2.shape
        tokens = f2.flatten(2).transpose(1, 2)
        q = self.query(tokens)
        k = self.key(tokens)
        v = self.value(tokens)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.attention_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        mapped = self.out(context).transpose(1, 2).view(b, 1, h, w)
        return torch.sigmoid(mapped)
