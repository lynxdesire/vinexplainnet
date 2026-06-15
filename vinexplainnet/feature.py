from __future__ import annotations

import torch
from torch import nn

from vinexplainnet.captions.generator import ExplanationGenerator
from vinexplainnet.crew.reel_types import Frame
from vinexplainnet.framing.heads import PredictionHead
from vinexplainnet.lens.backbone import EfficientBackbone
from vinexplainnet.lighting.fusion import HierarchicalAttentionFusion
from vinexplainnet.treatment.schema import Treatment


class VINExplainNet(nn.Module):
    def __init__(self, treatment: Treatment) -> None:
        super().__init__()
        spec = treatment.model
        c1, _, _, c4 = spec.channels
        num_classes = treatment.num_classes
        self.backbone = EfficientBackbone(spec, treatment.ass)
        self.hafm = HierarchicalAttentionFusion(spec)
        self.head = PredictionHead(c4, spec.hidden, num_classes)
        self.explainer = ExplanationGenerator(c4, c1, num_classes)

    def forward(self, x: torch.Tensor) -> Frame:
        f1, f2, f3, f4 = self.backbone(x)
        attended, fused = self.hafm(f2, f3, f4)
        logits = self.head(attended)
        probs = torch.sigmoid(logits)
        explanations = self.explainer(attended, fused, f1)
        return Frame(
            logits=logits,
            probs=probs,
            fused=fused,
            explanations=explanations,
            low_level=f1,
        )


class LogitsFeature(nn.Module):
    def __init__(self, model: VINExplainNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).logits


def build_feature(treatment: Treatment) -> VINExplainNet:
    return VINExplainNet(treatment)
