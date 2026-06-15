from __future__ import annotations

from collections.abc import Iterable

import torch

from vinexplainnet.crew.reel_types import Batch, Frame
from vinexplainnet.notes.classification import weighted_bce
from vinexplainnet.notes.explanation import explanation_quality
from vinexplainnet.notes.regularization import l2_penalty
from vinexplainnet.treatment.schema import TrainSpec


class TotalObjective:
    def __init__(self, spec: TrainSpec) -> None:
        self.lambda_cls = spec.lambda_cls
        self.lambda_e = spec.lambda_e
        self.lambda_r = spec.lambda_r
        self.beta = spec.boundary_beta

    def evaluate(
        self,
        frame: Frame,
        batch: Batch,
        params: Iterable[torch.Tensor],
        pos_weight: torch.Tensor,
        with_explain: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cls = weighted_bce(frame.logits, batch.labels, pos_weight)
        reg = l2_penalty(params)
        if with_explain:
            explain = explanation_quality(frame.explanations, batch.images, self.beta)
        else:
            explain = torch.zeros((), device=cls.device)
        total = self.lambda_cls * cls + self.lambda_e * explain + self.lambda_r * reg
        report = {
            "total": float(total.detach()),
            "cls": float(cls.detach()),
            "explain": float(explain.detach()),
            "reg": float(reg.detach()),
        }
        return total, report
