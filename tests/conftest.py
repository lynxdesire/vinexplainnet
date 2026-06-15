from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vinexplainnet.casting.synthetic import SyntheticOCT
from vinexplainnet.feature import VINExplainNet, build_feature
from vinexplainnet.treatment.loader import read_treatment
from vinexplainnet.treatment.schema import Treatment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke_treatment() -> Treatment:
    return read_treatment(ROOT / "configs" / "experiment" / "_smoke.conf")


@pytest.fixture
def model(smoke_treatment: Treatment) -> VINExplainNet:
    return build_feature(smoke_treatment)


@pytest.fixture
def sample(smoke_treatment: Treatment) -> tuple[torch.Tensor, torch.Tensor]:
    generator = SyntheticOCT(smoke_treatment.num_classes, smoke_treatment.data.image_size, 0)
    return generator.build(8)
