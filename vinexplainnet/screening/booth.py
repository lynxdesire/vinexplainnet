from __future__ import annotations

import torch

from vinexplainnet.casting.reels import Reel
from vinexplainnet.feature import VINExplainNet
from vinexplainnet.screening.metrics import classification_report


def collect(
    model: VINExplainNet, reel: Reel, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    model.to(device)
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in reel:
            frame = model(batch.images.to(device))
            logits.append(frame.logits.cpu())
            labels.append(batch.labels)
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def evaluate_reel(model: VINExplainNet, reel: Reel, device: torch.device) -> dict[str, float]:
    logits, labels = collect(model, reel, device)
    return classification_report(logits, labels)
