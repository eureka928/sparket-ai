"""Sharpness metrics.

Sharpness measures how decisive a forecaster's predictions are.
A forecaster who always predicts 50% is perfectly calibrated but useless.
Sharpness rewards predictions that deviate from the base rate.

Sharp = min(1, var / target_var)

where var is the variance of predicted probabilities.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def compute_variance(values: NDArray[np.float64]) -> float:
    """Compute variance of values."""
    if len(values) <= 1:
        return 0.0
    return float(np.var(values, ddof=0))


def compute_sharpness(
    probabilities: NDArray[np.float64],
    target_variance: float,
    min_samples: int = 30,
) -> float:
    """Compute sharpness score from predicted probabilities.

    Sharpness = min(1, var / target_var)

    Higher variance = sharper (more decisive) predictions.
    Capped at 1.0 to prevent over-rewarding extreme predictions.

    Args:
        probabilities: Array of predicted probabilities
        target_variance: Target variance for "sharp" predictions
        min_samples: Minimum samples required

    Returns:
        Sharpness score in [0, 1], or 0.5 if insufficient data
    """
    if len(probabilities) < min_samples:
        return 0.5

    variance = np.var(probabilities, ddof=0)

    if target_variance == 0:
        return 1.0 if variance > 0 else 0.0

    return min(1.0, variance / target_variance)


def compute_sharpness_batch(
    probabilities: NDArray[np.float64],
    miner_ids: NDArray[np.int64],
    target_variance: float,
    min_samples: int = 30,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Compute sharpness for multiple miners.

    Args:
        probabilities: All predicted probabilities
        miner_ids: Miner ID for each prediction
        target_variance: Target variance
        min_samples: Minimum samples per miner

    Returns:
        (unique_miner_ids, sharpness_scores)
    """
    unique_miners = np.unique(miner_ids)
    scores = np.full(len(unique_miners), 0.5)

    for i, mid in enumerate(unique_miners):
        mask = miner_ids == mid
        if mask.sum() >= min_samples:
            scores[i] = compute_sharpness(
                probabilities[mask],
                target_variance,
                min_samples,
            )

    return unique_miners, scores


__all__ = [
    "compute_variance",
    "compute_sharpness",
    "compute_sharpness_batch",
]
