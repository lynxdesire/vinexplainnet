from __future__ import annotations

import torch

from vinexplainnet.feature import LogitsFeature, VINExplainNet
from vinexplainnet.treatment.schema import Treatment


def test_forward_shapes(smoke_treatment: Treatment, model: VINExplainNet) -> None:
    frame = model(torch.randn(3, 1, 32, 32))
    assert frame.logits.shape == (3, smoke_treatment.num_classes)
    assert frame.probs.shape == (3, smoke_treatment.num_classes)
    assert frame.explanations.shape[0] == 3
    assert frame.explanations.shape[1] == smoke_treatment.num_classes


def test_probs_in_unit_range(model: VINExplainNet) -> None:
    frame = model(torch.randn(2, 1, 32, 32))
    assert float(frame.probs.min()) >= 0.0
    assert float(frame.probs.max()) <= 1.0


def test_gradient_flow_to_backbone(model: VINExplainNet) -> None:
    frame = model(torch.randn(2, 1, 32, 32))
    loss = frame.logits.sum() + frame.explanations.sum()
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "backbone" in n and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_logits_wrapper_matches(model: VINExplainNet) -> None:
    wrapper = LogitsFeature(model)
    x = torch.randn(2, 1, 32, 32)
    assert torch.allclose(wrapper(x), model(x).logits, atol=1e-5)
