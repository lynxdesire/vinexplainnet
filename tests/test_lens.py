from __future__ import annotations

import torch

from vinexplainnet.lens.backbone import EfficientBackbone
from vinexplainnet.lens.speckle import AdaptiveSpeckleSuppression
from vinexplainnet.treatment.schema import Treatment


def test_ass_alpha_in_unit_range() -> None:
    ass = AdaptiveSpeckleSuppression()
    alpha = torch.sigmoid(ass.alpha_raw)
    assert 0.0 <= float(alpha) <= 1.0


def test_ass_alpha_zero_is_identity() -> None:
    ass = AdaptiveSpeckleSuppression()
    with torch.no_grad():
        ass.alpha_raw.fill_(-50.0)
    x = torch.rand(2, 1, 16, 16)
    out = ass(x)
    assert torch.allclose(out, x, atol=1e-4)


def test_ass_full_blend_smooths() -> None:
    ass = AdaptiveSpeckleSuppression(sigma_r=10.0)
    with torch.no_grad():
        ass.alpha_raw.fill_(50.0)
    x = torch.rand(1, 1, 16, 16)
    out = ass(x)
    assert float(out.var()) <= float(x.var()) + 1e-6


def test_backbone_pyramid_shapes(smoke_treatment: Treatment) -> None:
    backbone = EfficientBackbone(smoke_treatment.model, smoke_treatment.ass)
    f1, f2, f3, f4 = backbone(torch.randn(2, 1, 32, 32))
    channels = smoke_treatment.model.channels
    assert f1.shape[1] == channels[0]
    assert f2.shape[1] == channels[1]
    assert f3.shape[1] == channels[2]
    assert f4.shape[1] == channels[3]
    assert f1.shape[-1] > f2.shape[-1] > f3.shape[-1] > f4.shape[-1]
