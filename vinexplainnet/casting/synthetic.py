from __future__ import annotations

import numpy as np
import torch


def _layered_retina(rng: np.random.Generator, size: int) -> np.ndarray:
    rows = np.linspace(0.0, 1.0, size).reshape(-1, 1)
    base = 0.35 + 0.25 * np.sin(rows * np.pi * 3.0 + rng.uniform(0.0, 0.5))
    canvas = np.repeat(base, size, axis=1)
    canvas = canvas + rng.normal(0.0, 0.02, size=(size, size))
    return canvas.astype(np.float32)


def _disc(size: int, cy: float, cx: float, radius: float) -> np.ndarray:
    ys = np.arange(size).reshape(-1, 1)
    xs = np.arange(size).reshape(1, -1)
    dist = np.sqrt((ys - cy * size) ** 2 + (xs - cx * size) ** 2)
    return np.clip(1.0 - dist / (radius * size), 0.0, 1.0).astype(np.float32)


def _lesion(rng: np.random.Generator, size: int, label: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    if label == 0:
        mask += 0.6 * _disc(size, 0.3, rng.uniform(0.3, 0.7), 0.18)
    elif label == 1:
        for _ in range(3):
            mask -= 0.5 * _disc(size, rng.uniform(0.4, 0.6), rng.uniform(0.3, 0.7), 0.08)
    elif label == 2:
        line = int(0.55 * size)
        for cx in np.linspace(0.2, 0.8, 5):
            mask += 0.5 * _disc(size, line / size, float(cx), 0.05)
    elif label == 3:
        mask += 0.0
    else:
        col = int(rng.uniform(0.3, 0.7) * size)
        mask[:, max(col - 2, 0) : col + 2] += 0.5
    return mask


def synth_image(rng: np.random.Generator, label: int, size: int) -> np.ndarray:
    canvas = _layered_retina(rng, size)
    canvas = canvas + _lesion(rng, size, label)
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


class SyntheticOCT:
    def __init__(self, num_classes: int, image_size: int, seed: int) -> None:
        self.num_classes = num_classes
        self.image_size = image_size
        self.seed = seed

    def build(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        images = np.empty((count, 1, self.image_size, self.image_size), dtype=np.float32)
        labels = np.empty((count,), dtype=np.int64)
        for i in range(count):
            rng = np.random.default_rng(self.seed * 100003 + i)
            label = i % self.num_classes
            images[i, 0] = synth_image(rng, label, self.image_size)
            labels[i] = label
        return torch.from_numpy(images), torch.from_numpy(labels)
