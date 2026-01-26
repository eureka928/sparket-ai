"""Sample-size shrinkage toward population mean.

Implements Bayesian shrinkage to adjust miner metrics toward the population
mean when sample size is small. This prevents small-sample miners from
appearing unrealistically good or bad due to luck.

Uses log-scaled shrinkage to provide smooth diminishing returns without
a sharp "cliff" at any particular sample size. This prevents gaming by
targeting a specific n_eff threshold.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def shrink_toward_mean(
    raw_values: NDArray[np.float64],
    n_effs: NDArray[np.float64],
    k: float,
    population_mean: float | None = None,
    weights: NDArray[np.float64] | None = None,
    use_log_scaling: bool = True,
) -> NDArray[np.float64]:
    """Apply shrinkage to an array of values.

    Uses log-scaled shrinkage for smooth diminishing returns:
    weight_raw = log(1 + n_eff) / log(1 + n_eff + k)

    This eliminates the "cliff" in the original linear formula where
    the derivative of weight_raw was highest around n_eff = k.

    With log scaling:
    - Small n_eff still gets heavily shrunk
    - Large n_eff approaches raw value asymptotically
    - No optimal n_eff to target - always better to have more samples

    Args:
        raw_values: Array of raw metric values
        n_effs: Array of effective sample sizes
        k: Shrinkage factor (higher = more shrinkage)
        population_mean: Pre-computed population mean (computed if None)
        weights: Weights for computing population mean (uses n_effs if None)
        use_log_scaling: If True, use log-scaled shrinkage (default)

    Returns:
        Array of shrunk values
    """
    if len(raw_values) == 0:
        return raw_values.copy()

    # Compute population mean if not provided
    if population_mean is None:
        if weights is None:
            weights = n_effs
        total_weight = weights.sum()
        if total_weight > 0:
            population_mean = float(np.average(raw_values, weights=weights))
        else:
            population_mean = float(raw_values.mean())

    # Apply shrinkage
    n_safe = np.maximum(n_effs, 0.0)
    k_safe = max(0.0, k)

    if use_log_scaling:
        # Log-scaled shrinkage: smooth diminishing returns, no cliff
        # weight_raw = log(1 + n_eff) / log(1 + n_eff + k)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_n = np.log1p(n_safe)
            log_total = np.log1p(n_safe + k_safe)
            weight_raw = np.where(log_total > 0, log_n / log_total, 0.0)
    else:
        # Original linear shrinkage (for backwards compatibility)
        total = n_safe + k_safe
        with np.errstate(divide='ignore', invalid='ignore'):
            weight_raw = np.where(total > 0, n_safe / total, 0.0)

    # Clean up any NaN from edge cases
    weight_raw = np.nan_to_num(weight_raw, nan=0.0)
    weight_raw = np.clip(weight_raw, 0.0, 1.0)
    weight_pop = 1.0 - weight_raw

    return weight_raw * raw_values + weight_pop * population_mean


def compute_population_mean(
    values: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Compute population mean.

    Args:
        values: Array of metric values
        weights: Optional weights

    Returns:
        Population mean
    """
    if len(values) == 0:
        return 0.0

    if weights is None:
        return float(values.mean())

    total_weight = weights.sum()
    if total_weight == 0:
        return float(values.mean())

    return float(np.average(values, weights=weights))


def shrink_grouped(
    values: NDArray[np.float64],
    n_effs: NDArray[np.float64],
    group_ids: NDArray[np.int64],
    k: float,
) -> NDArray[np.float64]:
    """Apply shrinkage within groups.

    Each group gets its own population mean computed from group members.

    Args:
        values: Array of raw metric values
        n_effs: Array of effective sample sizes
        group_ids: Array of group identifiers
        k: Shrinkage factor

    Returns:
        Array of shrunk values (same shape as values)
    """
    result = np.zeros_like(values)
    unique_groups = np.unique(group_ids)

    for gid in unique_groups:
        mask = group_ids == gid
        g_values = values[mask]
        g_n_effs = n_effs[mask]

        result[mask] = shrink_toward_mean(g_values, g_n_effs, k)

    return result


__all__ = [
    "shrink_toward_mean",
    "compute_population_mean",
    "shrink_grouped",
]
