from __future__ import annotations

import torch
from sklearn.metrics import accuracy_score

from vinexplainnet.casting.degrade import ConsumerDegradation
from vinexplainnet.casting.reels import Reel
from vinexplainnet.feature import VINExplainNet

DEFAULT_KINDS: tuple[str, ...] = (
    "none",
    "motion_mild",
    "motion_severe",
    "noise_2x",
    "noise_3x",
    "resolution_75",
    "resolution_50",
    "contrast",
    "combined_severe",
)


def degradation_sweep(
    model: VINExplainNet, reel: Reel, device: torch.device, kinds: tuple[str, ...] = DEFAULT_KINDS
) -> dict[str, float]:
    model.eval()
    model.to(device)
    degrade = ConsumerDegradation()
    report: dict[str, float] = {}
    with torch.no_grad():
        for kind in kinds:
            preds: list[int] = []
            truth: list[int] = []
            for batch in reel:
                images = degrade.apply(batch.images, kind).to(device)
                frame = model(images)
                preds.extend(frame.logits.argmax(dim=1).cpu().tolist())
                truth.extend(batch.labels.tolist())
            report[kind] = float(accuracy_score(truth, preds))
    return report
