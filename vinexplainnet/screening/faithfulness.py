from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F


def gini_sparsity(maps: torch.Tensor) -> float:
    values = maps.detach().abs().flatten().cpu().numpy().astype(np.float64)
    if values.sum() <= 0:
        return 0.0
    values = np.sort(values)
    n = values.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * values) / (n * values.sum())) - (n + 1.0) / n)


def deletion_faithfulness(
    predict_class_prob: Callable[[torch.Tensor], float],
    image: torch.Tensor,
    explanation: torch.Tensor,
    steps: int = 10,
) -> float:
    h, w = image.shape[-2], image.shape[-1]
    resized = explanation
    if resized.shape[-2:] != (h, w):
        resized = F.interpolate(
            resized.view(1, 1, *resized.shape[-2:]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).view(h, w)
    flat = resized.detach().flatten()
    order = torch.argsort(flat, descending=True)
    total = flat.numel()
    base = predict_class_prob(image)
    work = image.clone()
    remaining = []
    for k in range(1, steps + 1):
        upto = int(total * k / steps)
        idx = order[:upto]
        mask = torch.ones(total, device=image.device)
        mask[idx] = 0.0
        masked = work * mask.view(1, 1, h, w)
        remaining.append(predict_class_prob(masked))
    mean_after = float(np.mean(remaining))
    faith = (base - mean_after) / (abs(base) + 1e-8)
    return float(min(1.0, max(0.0, faith)))
