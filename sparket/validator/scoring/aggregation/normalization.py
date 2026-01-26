"""Score normalization utilities.

Normalizes raw metrics to [0, 1] range for combining into composite scores.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def normalize_zscore_logistic(
    values: NDArray[np.float64],
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Normalize values using z-score + logistic transformation.

    Formula:
    z = (x - mean) / std
    normalized = 1 / (1 + exp(-α * z))

    Args:
        values: Array of values
        alpha: Steepness parameter (higher = sharper transition)

    Returns:
        Normalized array in (0, 1)
    """
    n = len(values)
    if n == 0:
        return values.copy()
    if n == 1:
        return np.array([0.5])

    mean = values.mean()
    std = values.std()

    if std == 0:
        return np.full(n, 0.5)

    z = (values - mean) / std
    return 1.0 / (1.0 + np.exp(-alpha * z))


def normalize_percentile(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize values by percentile rank.

    Uses scipy.stats.rankdata with average tie-breaking.

    Args:
        values: Array of values

    Returns:
        Normalized array in (0, 1)
    """
    n = len(values)
    if n == 0:
        return values.copy()
    if n == 1:
        return np.array([0.5])

    ranks = stats.rankdata(values, method='average')
    return ranks / (n + 1)


def normalize_minmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize values using min-max scaling.

    (x - min) / (max - min) → [0, 1]

    Args:
        values: Array of values

    Returns:
        Normalized array in [0, 1]
    """
    n = len(values)
    if n == 0:
        return values.copy()
    if n == 1:
        return np.array([0.5])

    # Handle non-finite values
    if not np.all(np.isfinite(values)):
        # Replace inf with extreme values, NaN with median
        clean = np.where(np.isfinite(values), values, np.nan)
        median = np.nanmedian(clean) if np.any(np.isfinite(values)) else 0.5
        values = np.nan_to_num(values, nan=median, posinf=1e10, neginf=-1e10)

    min_val = values.min()
    max_val = values.max()
    range_val = max_val - min_val

    if range_val == 0 or not np.isfinite(range_val):
        return np.full(n, 0.5)

    return (values - min_val) / range_val


def normalize(
    values: NDArray[np.float64],
    method: Literal["zscore_logistic", "percentile", "minmax"] = "zscore_logistic",
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Normalize values using specified method.

    Args:
        values: Array of values
        method: Normalization method
        alpha: Steepness for zscore_logistic

    Returns:
        Normalized array in [0, 1]
    """
    if method == "zscore_logistic":
        return normalize_zscore_logistic(values, alpha)
    elif method == "percentile":
        return normalize_percentile(values)
    elif method == "minmax":
        return normalize_minmax(values)
    else:
        return normalize_zscore_logistic(values, alpha)


def normalize_grouped(
    values: NDArray[np.float64],
    group_ids: NDArray[np.int64],
    method: Literal["zscore_logistic", "percentile", "minmax"] = "zscore_logistic",
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Normalize values within groups.

    Args:
        values: Array of values
        group_ids: Array of group identifiers
        method: Normalization method
        alpha: Steepness for zscore_logistic

    Returns:
        Normalized array (same shape as values)
    """
    result = np.zeros_like(values)
    unique_groups = np.unique(group_ids)

    for gid in unique_groups:
        mask = group_ids == gid
        result[mask] = normalize(values[mask], method, alpha)

    return result


__all__ = [
    "normalize_zscore_logistic",
    "normalize_percentile",
    "normalize_minmax",
    "normalize",
    "normalize_grouped",
]
