from __future__ import annotations

from vinexplainnet.casting.reels import Reel
from vinexplainnet.casting.synthetic import SyntheticOCT
from vinexplainnet.crew.seed import set_seed
from vinexplainnet.direction.runner import Director
from vinexplainnet.feature import build_feature
from vinexplainnet.treatment.schema import Treatment


def _fixed_reel(treatment: Treatment) -> Reel:
    images, labels = SyntheticOCT(treatment.num_classes, treatment.data.image_size, 0).build(8)
    return Reel(images, labels, 8, False, 0)


def test_overfit_single_batch_decreases(smoke_treatment: Treatment) -> None:
    set_seed(0)
    reel = _fixed_reel(smoke_treatment)
    model = build_feature(smoke_treatment)
    history = Director(smoke_treatment).shoot(model, reel, 30, augment_on=False)
    assert history[-1]["total"] < history[0]["total"]
    assert history[-1]["cls"] < history[0]["cls"]


def test_training_is_deterministic(smoke_treatment: Treatment) -> None:
    def run() -> float:
        set_seed(123)
        reel = _fixed_reel(smoke_treatment)
        model = build_feature(smoke_treatment)
        history = Director(smoke_treatment).shoot(model, reel, 5, augment_on=False)
        return history[-1]["total"]

    assert abs(run() - run()) < 1e-6
