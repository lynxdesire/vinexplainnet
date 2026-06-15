from __future__ import annotations

import torch


class ConsumerAugment:
    def __init__(self, seed: int = 0) -> None:
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def _rand(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        cpu = images.detach().to("cpu")
        b = cpu.shape[0]
        flip = self._rand((b,)) < 0.5
        cpu[flip] = torch.flip(cpu[flip], dims=[-1])
        brightness = 1.0 + (self._rand((b, 1, 1, 1)) - 0.5) * 0.4
        contrast = 1.0 + (self._rand((b, 1, 1, 1)) - 0.5) * 0.4
        mean = cpu.mean(dim=(-1, -2), keepdim=True)
        cpu = (cpu - mean) * contrast + mean * brightness
        noise = torch.randn(cpu.shape, generator=self.generator) * 0.03
        cpu = (cpu + noise).clamp(0.0, 1.0)
        return cpu.to(images.device)
