from __future__ import annotations

import torch

from vinexplainnet.cutting.prune import channel_importance, retained_fraction, structured_prune
from vinexplainnet.cutting.quantize import MixedPrecisionQuantizer, dequantize, quantize_tensor
from vinexplainnet.feature import VINExplainNet


def test_structured_prune_reduces_channels() -> None:
    scores = torch.tensor([1.0, 0.05, 0.8, 0.02])
    weight = torch.randn(4, 3, 3, 3)
    kept = structured_prune(weight, scores, ratio=0.3)
    assert kept.shape[0] < weight.shape[0]
    assert kept.shape[1:] == weight.shape[1:]
    assert retained_fraction(scores, 0.3) < 1.0


def test_channel_importance_shape() -> None:
    acts = torch.randn(2, 5, 4, 4)
    grads = torch.randn(2, 5, 4, 4)
    scores = channel_importance(acts, grads)
    assert scores.shape == (5,)
    assert float(scores.min()) >= 0.0


def test_quantize_roundtrip_within_one_lsb() -> None:
    weight = torch.randn(64) * 0.5
    codes, scale, w_min = quantize_tensor(weight, bits=8)
    restored = dequantize(codes, scale, w_min)
    assert float((restored - weight).abs().max()) <= scale


def test_mixed_precision_reduces_size(smoke_treatment, model: VINExplainNet) -> None:
    quantizer = MixedPrecisionQuantizer(bits=8)
    report = quantizer.quantize(model)
    assert report["quant_bytes"] < report["fp32_bytes"]
    assert 0.0 < report["ratio"] < 1.0
