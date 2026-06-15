from __future__ import annotations

import torch

from vinexplainnet.notes.classification import class_balance_weights, weighted_bce
from vinexplainnet.notes.explanation import boundary_term, coherence_term, sparsity_term
from vinexplainnet.notes.regularization import l2_penalty


def test_coherence_zero_on_constant() -> None:
    assert float(coherence_term(torch.ones(2, 3, 8, 8))) == 0.0


def test_coherence_positive_on_noise() -> None:
    torch.manual_seed(0)
    assert float(coherence_term(torch.randn(2, 3, 8, 8))) > 0.0


def test_sparsity_known_values() -> None:
    assert float(sparsity_term(torch.zeros(1, 1, 4, 4))) == 0.0
    assert abs(float(sparsity_term(torch.ones(1, 1, 4, 4))) - 1.0) < 1e-6


def test_boundary_zero_when_flat() -> None:
    maps = torch.ones(2, 1, 8, 8)
    image = torch.ones(2, 1, 16, 16)
    assert float(boundary_term(maps, image, 1.0)) == 0.0


def test_class_balance_weights_sum_to_classes() -> None:
    targets = torch.tensor([0, 0, 0, 1, 2])
    weights = class_balance_weights(targets, 3)
    assert abs(float(weights.sum()) - 3.0) < 1e-5
    assert float(weights[1]) > float(weights[0])


def test_weighted_bce_lower_when_correct() -> None:
    good = torch.tensor([[6.0, -6.0, -6.0, -6.0]])
    bad = torch.tensor([[-6.0, 6.0, 6.0, 6.0]])
    target = torch.tensor([0])
    weights = torch.ones(4)
    assert float(weighted_bce(good, target, weights)) < float(weighted_bce(bad, target, weights))


def test_l2_penalty_closed_form() -> None:
    param = torch.tensor([3.0, 4.0], requires_grad=True)
    assert abs(float(l2_penalty([param])) - 25.0) < 1e-5
