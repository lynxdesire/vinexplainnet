from __future__ import annotations

from collections.abc import Iterator

import torch

from vinexplainnet.casting.sources import ManifestSource
from vinexplainnet.casting.synthetic import SyntheticOCT
from vinexplainnet.crew.reel_types import Batch
from vinexplainnet.treatment.schema import Treatment


class Reel:
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.images = images
        self.labels = labels
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self) -> int:
        return (self.images.shape[0] + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        count = self.images.shape[0]
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed)
            order = torch.randperm(count, generator=generator)
        else:
            order = torch.arange(count)
        for start in range(0, count, self.batch_size):
            idx = order[start : start + self.batch_size]
            yield Batch(images=self.images[idx], labels=self.labels[idx])


def _materialise(treatment: Treatment, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if treatment.data.manifest:
        return ManifestSource(treatment.data.manifest, treatment.data.image_size).build()
    generator = SyntheticOCT(treatment.num_classes, treatment.data.image_size, seed)
    return generator.build(count)


def make_loaders(treatment: Treatment) -> tuple[Reel, Reel]:
    train_images, train_labels = _materialise(
        treatment, treatment.data.n_train, treatment.train.seed
    )
    val_images, val_labels = _materialise(treatment, treatment.data.n_val, treatment.train.seed + 1)
    train = Reel(train_images, train_labels, treatment.data.batch_size, True, treatment.train.seed)
    val = Reel(val_images, val_labels, treatment.data.batch_size, False, treatment.train.seed)
    return train, val
