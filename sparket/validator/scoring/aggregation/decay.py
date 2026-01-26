"""Time decay weighting for rolling aggregates.

Implements exponential decay to weight recent submissions more heavily.
This ensures that miner scores reflect recent performance rather than
distant history.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


# Pre-computed ln(0.5) for efficiency
LN_HALF = np.log(0.5)


def compute_decay_weight(age_days: float, half_life_days: float) -> float:
    """Compute exponential decay weight for a given age.

    weight = 0.5 ** (age_days / half_life_days)

    Args:
        age_days: Age of observation in days
        half_life_days: Half-life for decay

    Returns:
        Weight in (0, 1], or 0.0 for invalid half_life
    """
    # Handle invalid half_life gracefully
    if half_life_days <= 0:
        return 0.0 if age_days > 0 else 1.0

    age = max(0.0, age_days)
    return float(np.exp(age * LN_HALF / half_life_days))


def compute_decay_weights(
    timestamps: NDArray[np.float64],
    reference_timestamp: float,
    half_life_days: float,
) -> NDArray[np.float64]:
    """Compute decay weights for an array of timestamps.

    Args:
        timestamps: Array of Unix timestamps
        reference_timestamp: Reference Unix timestamp
        half_life_days: Half-life for decay in days

    Returns:
        Array of weights (1.0 for current, decaying for older)
    """
    if len(timestamps) == 0:
        return np.array([])

    # Handle invalid half_life gracefully
    if half_life_days <= 0:
        # No decay - give weight 1 to current timestamp, 0 to others
        age_seconds = reference_timestamp - timestamps
        return np.where(age_seconds <= 0, 1.0, 0.0)

    age_seconds = reference_timestamp - timestamps
    age_days = np.maximum(0, age_seconds / 86400.0)
    return np.exp(age_days * LN_HALF / half_life_days)


def effective_sample_size(weights: NDArray[np.float64]) -> float:
    """Compute effective sample size from decay weights.

    n_eff = sum(weights)

    Args:
        weights: Array of decay weights

    Returns:
        Effective sample size
    """
    return float(np.sum(weights))


def weighted_mean(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Compute weighted mean.

    Args:
        values: Array of values
        weights: Array of weights

    Returns:
        Weighted mean
    """
    weight_sum = weights.sum()
    if weight_sum == 0:
        return 0.0
    return float(np.average(values, weights=weights))


def weighted_std(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    mean: float | None = None,
) -> float:
    """Compute weighted standard deviation.

    Args:
        values: Array of values
        weights: Array of weights
        mean: Pre-computed weighted mean (optional)

    Returns:
        Weighted standard deviation
    """
    weight_sum = weights.sum()
    if weight_sum == 0:
        return 0.0

    if mean is None:
        mean = np.average(values, weights=weights)

    variance = np.average((values - mean) ** 2, weights=weights)
    return float(np.sqrt(max(0, variance)))


def weighted_aggregates(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> Tuple[float, float, float]:
    """Compute weighted mean, std, and effective n in one pass.

    Args:
        values: Array of values
        weights: Array of weights

    Returns:
        (weighted_mean, weighted_std, n_eff)
    """
    n_eff = float(weights.sum())
    if n_eff == 0:
        return 0.0, 0.0, 0.0

    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    std = np.sqrt(max(0, variance))

    return float(mean), float(std), n_eff


def weighted_aggregates_batch(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    group_ids: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Compute weighted aggregates for multiple groups.

    Args:
        values: Array of values
        weights: Array of weights
        group_ids: Array of group identifiers

    Returns:
        (unique_ids, means, stds, n_effs)
    """
    unique_ids = np.unique(group_ids)
    n_groups = len(unique_ids)

    means = np.zeros(n_groups, dtype=np.float64)
    stds = np.zeros(n_groups, dtype=np.float64)
    n_effs = np.zeros(n_groups, dtype=np.float64)

    for i, gid in enumerate(unique_ids):
        mask = group_ids == gid
        g_values = values[mask]
        g_weights = weights[mask]

        mean, std, n_eff = weighted_aggregates(g_values, g_weights)
        means[i] = mean
        stds[i] = std
        n_effs[i] = n_eff

    return unique_ids, means, stds, n_effs


__all__ = [
    "compute_decay_weight",
    "compute_decay_weights",
    "effective_sample_size",
    "weighted_mean",
    "weighted_std",
    "weighted_aggregates",
    "weighted_aggregates_batch",
]
