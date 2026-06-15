from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from vinexplainnet.screening.clinical import clinical_alignment, synthetic_ratings
from vinexplainnet.screening.faithfulness import deletion_faithfulness, gini_sparsity
from vinexplainnet.screening.metrics import classification_report
from vinexplainnet.screening.significance import bonferroni, bootstrap_ci, mcnemar_test


def test_metrics_match_sklearn() -> None:
    logits = torch.tensor(
        [[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [3.0, 0.0, 0.0, 0.0]]
    )
    targets = torch.tensor([0, 1, 2, 3])
    report = classification_report(logits, targets)
    preds = logits.argmax(dim=1).numpy()
    assert abs(report["accuracy"] - accuracy_score(targets.numpy(), preds)) < 1e-9
    expected_f1 = f1_score(targets.numpy(), preds, average="macro", zero_division=0)
    assert abs(report["f1"] - expected_f1) < 1e-9


def test_gini_bounds() -> None:
    torch.manual_seed(1)
    value = gini_sparsity(torch.rand(1, 1, 8, 8))
    assert 0.0 <= value < 1.0


def test_deletion_faithfulness_prefers_aligned() -> None:
    image = torch.zeros(1, 1, 8, 8)
    image[..., 2:5, 2:5] = 1.0
    aligned = image[0, 0].clone()
    torch.manual_seed(0)
    random_map = torch.rand(8, 8)

    def prob(masked: torch.Tensor) -> float:
        return float(masked.sum().item()) / (float(image.sum().item()) + 1e-6)

    aligned_score = deletion_faithfulness(prob, image, aligned)
    random_score = deletion_faithfulness(prob, image, random_map)
    assert aligned_score >= random_score


def test_mcnemar_symmetry_and_range() -> None:
    a = np.array([True, True, False, False, True])
    b = np.array([False, False, True, True, False])
    p = mcnemar_test(a, b)
    assert 0.0 <= p <= 1.0
    assert abs(mcnemar_test(a, b) - mcnemar_test(b, a)) < 1e-9


def test_bonferroni_threshold() -> None:
    flags = bonferroni([0.001, 0.02, 0.4], 0.05)
    assert flags == [True, False, False]


def test_bootstrap_ci_brackets_mean() -> None:
    values = np.full(50, 0.9)
    low, high = bootstrap_ci(values, 200, 0.05, seed=3)
    assert low <= 0.9 <= high


def test_clinical_alignment_ci() -> None:
    ratings = synthetic_ratings(seed=7)
    report = clinical_alignment(ratings, seed=7)
    assert report["ci_low"] <= report["mean"] <= report["ci_high"]
    assert 1.0 <= report["mean"] <= 5.0
