from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vinexplainnet.crew.vault import read_json


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    classes: tuple[str, ...]
    access: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "kermany": DatasetSpec(
        "kermany",
        ("CNV", "DME", "Drusen", "Normal"),
        "https://data.mendeley.com/datasets/rscbjbr9sj/3",
    ),
    "duke": DatasetSpec(
        "duke",
        ("AMD", "DME", "Normal"),
        "https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm",
    ),
}


class ManifestSource:
    def __init__(self, manifest_path: str | Path, image_size: int) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size

    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        records = read_json(self.manifest_path)
        entries = records["items"]
        count = len(entries)
        images = np.empty((count, 1, self.image_size, self.image_size), dtype=np.float32)
        labels = np.empty((count,), dtype=np.int64)
        for i, entry in enumerate(entries):
            array = np.load(entry["path"]).astype(np.float32)
            images[i, 0] = array
            labels[i] = int(entry["label"])
        return torch.from_numpy(images), torch.from_numpy(labels)
