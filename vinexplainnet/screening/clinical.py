from __future__ import annotations

import numpy as np

from vinexplainnet.screening.significance import bootstrap_ci


def synthetic_ratings(seed: int, raters: int = 23, center: float = 4.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.normal(center, 0.6, size=raters)
    return np.clip(np.round(draws), 1.0, 5.0)


def clinical_alignment(ratings: np.ndarray, seed: int = 0) -> dict[str, float]:
    low, high = bootstrap_ci(ratings.astype(np.float64), 1000, 0.05, seed)
    return {
        "mean": float(ratings.mean()),
        "ci_low": low,
        "ci_high": high,
    }
