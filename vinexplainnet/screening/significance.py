from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n = a_only + b_only
    if n == 0:
        return 1.0
    statistic = (abs(a_only - b_only) - 1.0) ** 2 / n
    return float(stats.chi2.sf(statistic, df=1))


def bonferroni(pvalues: list[float], alpha: float) -> list[bool]:
    m = len(pvalues)
    return [p < alpha / m for p in pvalues]


def bootstrap_ci(
    values: np.ndarray, iterations: int, alpha: float, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    means = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = float(sample.mean())
    low = float(np.percentile(means, 100.0 * alpha / 2.0))
    high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high
