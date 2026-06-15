from __future__ import annotations

import torch

from vinexplainnet.casting.reels import Reel
from vinexplainnet.feature import VINExplainNet
from vinexplainnet.screening.booth import evaluate_reel


def cross_dataset(model: VINExplainNet, target: Reel, device: torch.device) -> float:
    return evaluate_reel(model, target, device)["accuracy"]
