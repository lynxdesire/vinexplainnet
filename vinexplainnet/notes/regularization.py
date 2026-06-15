from __future__ import annotations

from collections.abc import Iterable

import torch


def l2_penalty(params: Iterable[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros(1)
    for param in params:
        if param.requires_grad:
            total = total.to(param.device)
            total = total + param.pow(2).sum()
    return total.squeeze()
