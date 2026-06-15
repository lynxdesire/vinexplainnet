from __future__ import annotations

from dataclasses import dataclass

from vinexplainnet.treatment.schema import Treatment


@dataclass(frozen=True)
class Stage:
    name: str
    epochs: int
    train_backbone: bool
    explain_weight: float


def build_stages(treatment: Treatment) -> list[Stage]:
    return [
        Stage("backbone_pretrain", 0, True, 0.0),
        Stage("task_specific", treatment.train.stage2_epochs, True, 1.0),
        Stage("explanation_refinement", treatment.train.stage3_epochs, False, 3.0),
    ]
