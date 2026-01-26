"""Time-to-close scoring adjustment.

Applied as SCORE MULTIPLIER (not weight multiplier) with asymmetric treatment:
- Early GOOD predictions: Full credit (rewarded for skill)
- Early BAD predictions: Clipped penalty (expected uncertainty)
- Late GOOD predictions: Reduced credit (might be copying)
- Late BAD predictions: Full penalty (no excuse)

The time factor is logarithmic to:
1. Heavily penalize very late submissions (copy-trading)
2. Provide diminishing returns for very early submissions
3. Create a smooth transition, not a cliff
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_time_factor(
    minutes_to_close: int | float,
    min_minutes: int = 60,
    max_minutes: int = 10080,  # 7 days
    floor_factor: float = 0.1,
) -> float:
    """Compute time-to-close factor for a single submission.

    Uses logarithmic scaling:
    - Submissions at or beyond max_minutes get factor 1.0
    - Submissions at min_minutes get floor_factor
    - Submissions between are log-scaled

    Args:
        minutes_to_close: Minutes between submission and event start
        min_minutes: Submissions below this get floor_factor (default 1 hour)
        max_minutes: Submissions beyond this get factor 1.0 (default 7 days)
        floor_factor: Minimum factor for very late submissions

    Returns:
        Factor in [floor_factor, 1.0]
    """
    if minutes_to_close <= 0:
        return floor_factor

    if minutes_to_close >= max_minutes:
        return 1.0

    if minutes_to_close <= min_minutes:
        return floor_factor

    # Log scaling between min and max
    log_min = np.log(min_minutes)
    log_max = np.log(max_minutes)
    log_val = np.log(minutes_to_close)

    # Normalize to [0, 1] then scale to [floor_factor, 1.0]
    normalized = (log_val - log_min) / (log_max - log_min)
    return floor_factor + normalized * (1.0 - floor_factor)


def compute_time_factors(
    minutes_to_close: NDArray[np.int64] | NDArray[np.float64],
    min_minutes: int = 60,
    max_minutes: int = 10080,
    floor_factor: float = 0.1,
) -> NDArray[np.float64]:
    """Compute time-to-close factors for a batch of submissions.

    Args:
        minutes_to_close: Array of minutes to close
        min_minutes: Submissions below this get floor_factor
        max_minutes: Submissions beyond this get factor 1.0
        floor_factor: Minimum factor for very late submissions

    Returns:
        Array of factors in [floor_factor, 1.0]
    """
    minutes = np.asarray(minutes_to_close, dtype=np.float64)

    # Initialize with floor factor
    factors = np.full_like(minutes, floor_factor, dtype=np.float64)

    # Max factor for early submissions
    factors = np.where(minutes >= max_minutes, 1.0, factors)

    # Log-scaled for middle range
    in_range = (minutes > min_minutes) & (minutes < max_minutes)
    if np.any(in_range):
        log_min = np.log(min_minutes)
        log_max = np.log(max_minutes)
        log_val = np.log(np.maximum(minutes[in_range], 1))

        normalized = (log_val - log_min) / (log_max - log_min)
        factors[in_range] = floor_factor + normalized * (1.0 - floor_factor)

    return factors


def apply_time_bonus(
    score: float,
    minutes_to_close: int,
    min_minutes: int = 60,
    max_minutes: int = 10080,
    floor_factor: float = 0.1,
    early_penalty_clip: float = 0.7,
) -> float:
    """Apply asymmetric time bonus to a skill score.

    For POSITIVE scores (beat baseline):
    - Early submission: Full credit (factor = 1.0)
    - Late submission: Reduced credit (factor = floor_factor)

    For NEGATIVE scores (worse than baseline):
    - Early submission: Clipped penalty (factor = early_penalty_clip)
    - Late submission: Full penalty (factor ≈ 1.0)

    Args:
        score: Raw PSS or skill score (positive = good, negative = bad)
        minutes_to_close: Minutes between submission and event start
        min_minutes: Threshold for "late" submissions
        max_minutes: Threshold for "early" submissions
        floor_factor: Minimum factor for late GOOD predictions
        early_penalty_clip: Penalty multiplier for early BAD predictions

    Returns:
        Time-adjusted score
    """
    time_factor = compute_time_factor(minutes_to_close, min_minutes, max_minutes, floor_factor)

    if score >= 0:
        # Good prediction: apply time factor directly
        # Early = full credit, Late = reduced credit
        return score * time_factor
    else:
        # Bad prediction: invert the logic
        # Early = clipped penalty, Late = full penalty
        # penalty_factor goes from early_penalty_clip (early) to ~1.0 (late)
        penalty_factor = early_penalty_clip + (1.0 - early_penalty_clip) * (1.0 - time_factor)
        return score * penalty_factor


def apply_time_bonus_batch(
    scores: NDArray[np.float64],
    minutes_to_close: NDArray[np.int64] | NDArray[np.float64],
    min_minutes: int = 60,
    max_minutes: int = 10080,
    floor_factor: float = 0.1,
    early_penalty_clip: float = 0.7,
) -> NDArray[np.float64]:
    """Apply asymmetric time bonus to a batch of skill scores.

    Vectorized version of apply_time_bonus.

    Args:
        scores: Array of raw PSS or skill scores
        minutes_to_close: Array of minutes to close
        min_minutes: Threshold for "late" submissions
        max_minutes: Threshold for "early" submissions
        floor_factor: Minimum factor for late GOOD predictions
        early_penalty_clip: Penalty multiplier for early BAD predictions

    Returns:
        Array of time-adjusted scores
    """
    time_factors = compute_time_factors(minutes_to_close, min_minutes, max_minutes, floor_factor)

    # For positive scores: multiply by time_factor
    # For negative scores: multiply by penalty_factor
    penalty_factors = early_penalty_clip + (1.0 - early_penalty_clip) * (1.0 - time_factors)

    result = np.where(
        scores >= 0,
        scores * time_factors,      # Good: early = full credit
        scores * penalty_factors,   # Bad: early = clipped penalty
    )

    return result


# Backwards compatibility aliases
compute_time_weight = compute_time_factor
compute_time_weights = compute_time_factors


def apply_time_weighting(
    scores: NDArray[np.float64],
    minutes_to_close: NDArray[np.int64] | NDArray[np.float64],
    min_minutes: int = 60,
    max_minutes: int = 10080,
    floor_weight: float = 0.1,
) -> NDArray[np.float64]:
    """Apply time-to-close weighting to an array of scores.

    DEPRECATED: Use apply_time_bonus_batch for asymmetric treatment.
    This function applies simple multiplication (symmetric).

    Args:
        scores: Raw scores to weight
        minutes_to_close: Minutes to close for each score
        min_minutes: Submissions below this get floor_weight
        max_minutes: Submissions beyond this get weight 1.0
        floor_weight: Minimum weight for very late submissions

    Returns:
        Time-weighted scores
    """
    time_weights = compute_time_factors(
        minutes_to_close, min_minutes, max_minutes, floor_weight
    )
    return scores * time_weights


__all__ = [
    "compute_time_factor",
    "compute_time_factors",
    "apply_time_bonus",
    "apply_time_bonus_batch",
    # Backwards compatibility
    "compute_time_weight",
    "compute_time_weights",
    "apply_time_weighting",
]
