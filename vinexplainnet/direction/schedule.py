from __future__ import annotations

import math


def cosine_lr(step: int, total: int, lr0: float, lr_min: float) -> float:
    if total <= 0:
        return lr0
    progress = min(step, total) / total
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress))
