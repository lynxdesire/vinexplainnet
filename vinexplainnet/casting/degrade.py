from __future__ import annotations

import torch
import torch.nn.functional as F


def motion_blur(images: torch.Tensor, length: int = 7) -> torch.Tensor:
    kernel = torch.zeros(1, 1, 1, length, device=images.device, dtype=images.dtype)
    kernel[..., :] = 1.0 / length
    pad = length // 2
    blurred = F.conv2d(F.pad(images, (pad, pad, 0, 0), mode="reflect"), kernel)
    return blurred


def add_speckle(images: torch.Tensor, factor: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(7)
    noise = torch.randn(images.shape, generator=generator).to(images.device)
    return (images + factor * 0.05 * noise * images).clamp(0.0, 1.0)


def reduce_resolution(images: torch.Tensor, ratio: float) -> torch.Tensor:
    size = images.shape[-1]
    small = max(int(size * ratio), 1)
    down = F.interpolate(images, size=(small, small), mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(size, size), mode="bilinear", align_corners=False)


def shift_contrast(images: torch.Tensor, scale: float) -> torch.Tensor:
    mean = images.mean(dim=(-1, -2), keepdim=True)
    return ((images - mean) * scale + mean).clamp(0.0, 1.0)


class ConsumerDegradation:
    def apply(self, images: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "motion_mild":
            return motion_blur(images, 5)
        if kind == "motion_severe":
            return motion_blur(images, 11)
        if kind == "noise_2x":
            return add_speckle(images, 2.0)
        if kind == "noise_3x":
            return add_speckle(images, 3.0)
        if kind == "resolution_75":
            return reduce_resolution(images, 0.75)
        if kind == "resolution_50":
            return reduce_resolution(images, 0.5)
        if kind == "contrast":
            return shift_contrast(images, 0.7)
        if kind == "combined_severe":
            stepped = motion_blur(images, 11)
            stepped = add_speckle(stepped, 3.0)
            return reduce_resolution(stepped, 0.5)
        return images
