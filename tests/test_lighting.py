from __future__ import annotations

import torch

from vinexplainnet.lighting.fusion import HierarchicalAttentionFusion
from vinexplainnet.treatment.schema import Treatment


def test_fusion_weights_sum_to_one(smoke_treatment: Treatment) -> None:
    hafm = HierarchicalAttentionFusion(smoke_treatment.model)
    coeff = torch.softmax(hafm.weights, dim=0)
    assert abs(float(coeff.sum()) - 1.0) < 1e-6


def test_streams_in_unit_range_and_shapes(smoke_treatment: Treatment) -> None:
    hafm = HierarchicalAttentionFusion(smoke_treatment.model)
    _, c2, c3, c4 = smoke_treatment.model.channels
    f2 = torch.randn(2, c2, 8, 8)
    f3 = torch.randn(2, c3, 4, 4)
    f4 = torch.randn(2, c4, 2, 2)
    a_local, a_regional, a_global = hafm.streams(f2, f3, f4)
    for attn in (a_local, a_regional, a_global):
        assert attn.shape == (2, 1, 2, 2)
        assert float(attn.min()) >= 0.0
        assert float(attn.max()) <= 1.0
    attended, fused = hafm(f2, f3, f4)
    assert fused.shape == (2, 1, 2, 2)
    assert attended.shape == (2, c4, 2, 2)


def test_modulation_residual_identity_when_fused_zero(smoke_treatment: Treatment) -> None:
    hafm = HierarchicalAttentionFusion(smoke_treatment.model)
    c4 = smoke_treatment.model.channels[3]
    f4 = torch.randn(2, c4, 2, 2)
    fused = torch.zeros(2, 1, 2, 2)
    assert torch.allclose(hafm.modulate(f4, fused), f4)
