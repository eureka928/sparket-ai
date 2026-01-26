"""Calibration metrics.

Calibration measures whether predicted probabilities match actual frequencies.
A well-calibrated model that predicts 70% should see the outcome occur
approximately 70% of the time.

We fit: logit(observed_freq) ≈ a + b * logit(predicted_prob)

- b ≈ 1, a ≈ 0: Well calibrated
- b > 1: Overconfident (probabilities too extreme)
- b < 1: Underconfident (probabilities too moderate)
- a ≠ 0: Systematic bias

Security note: Bin edges are deterministically jittered per scoring window
to prevent miners from exploiting knowledge of exact bin boundaries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats


# Small epsilon to prevent log(0)
EPS = 1e-6

# Maximum jitter as fraction of bin width (±2% of bin)
BIN_JITTER_FRACTION = 0.02


@dataclass
class CalibrationResult:
    """Result of calibration computation."""

    intercept: float  # a
    slope: float  # b
    score: float  # Cal score in (0, 1]
    bins_used: int


def get_jittered_bin_edges(
    num_bins: int,
    window_seed: str | None = None,
) -> NDArray[np.float64]:
    """Generate deterministically jittered bin edges.

    Jitter prevents miners from exploiting exact bin boundary knowledge.
    The jitter is deterministic per window, so all validators agree.

    Args:
        num_bins: Number of bins
        window_seed: Seed string (e.g., window start/end timestamp)
                    If None, uses standard evenly-spaced bins

    Returns:
        Array of bin edges (length num_bins + 1)
    """
    base_edges = np.linspace(0, 1, num_bins + 1)

    if window_seed is None:
        return base_edges

    # Generate deterministic jitter from seed
    seed_hash = hashlib.sha256(window_seed.encode()).digest()
    # Use first 8 bytes as seed for numpy RNG
    seed_int = int.from_bytes(seed_hash[:8], 'big') % (2**32)

    rng = np.random.Generator(np.random.PCG64(seed_int))

    # Jitter interior edges only (keep 0 and 1 fixed)
    bin_width = 1.0 / num_bins
    max_jitter = bin_width * BIN_JITTER_FRACTION

    jitter = rng.uniform(-max_jitter, max_jitter, num_bins + 1)
    jitter[0] = 0  # Keep 0 fixed
    jitter[-1] = 0  # Keep 1 fixed

    jittered = base_edges + jitter

    # Ensure edges are still monotonically increasing
    jittered = np.sort(jittered)
    jittered[0] = 0.0
    jittered[-1] = 1.0

    return jittered


def logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute logit (log-odds) of probabilities.

    logit(p) = log(p / (1 - p))

    Args:
        p: Array of probabilities

    Returns:
        Array of logit values
    """
    p_clamped = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p_clamped / (1.0 - p_clamped))


def fit_calibration_curve(
    probs: NDArray[np.float64],
    outcomes: NDArray[np.int8],
    num_bins: int = 10,
    min_samples_per_bin: int = 5,
    window_seed: str | None = None,
) -> Tuple[float, float]:
    """Fit calibration curve using linear regression on logit scale.

    Fits: logit(freq) = a + b * logit(predicted_prob)

    Args:
        probs: Predicted probabilities
        outcomes: Binary outcomes (0 or 1)
        num_bins: Number of calibration bins
        min_samples_per_bin: Minimum samples per bin
        window_seed: Seed for deterministic bin edge jitter (anti-gaming)

    Returns:
        (intercept, slope) coefficients
    """
    if len(probs) == 0:
        return 0.0, 1.0

    # Bin the probabilities with jittered edges
    bin_edges = get_jittered_bin_edges(num_bins, window_seed)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Compute mean prob and empirical freq per bin
    x_vals = []
    y_vals = []

    for i in range(num_bins):
        mask = bin_indices == i
        count = mask.sum()

        if count >= min_samples_per_bin:
            mean_prob = probs[mask].mean()
            empirical_freq = outcomes[mask].mean()

            # Clamp to avoid extreme logits
            mean_prob = np.clip(mean_prob, EPS, 1.0 - EPS)
            empirical_freq = np.clip(empirical_freq, EPS, 1.0 - EPS)

            x_vals.append(np.log(mean_prob / (1.0 - mean_prob)))
            y_vals.append(np.log(empirical_freq / (1.0 - empirical_freq)))

    if len(x_vals) < 2:
        return 0.0, 1.0

    x_arr = np.array(x_vals, dtype=np.float64)
    y_arr = np.array(y_vals, dtype=np.float64)

    result = stats.linregress(x_arr, y_arr)
    return float(result.intercept), float(result.slope)


def calibration_score(intercept: float, slope: float) -> float:
    """Compute calibration score from regression coefficients.

    Cal = 1 / (1 + |b - 1| + |a|)

    Perfect calibration (a=0, b=1) gives Cal = 1.
    Poor calibration gives Cal → 0.

    Args:
        intercept: a coefficient
        slope: b coefficient

    Returns:
        Calibration score in (0, 1]
    """
    error = abs(slope - 1.0) + abs(intercept)
    return 1.0 / (1.0 + error)


def compute_calibration(
    probs: NDArray[np.float64],
    outcomes: NDArray[np.int8],
    num_bins: int = 10,
    min_samples: int = 30,
    min_samples_per_bin: int = 5,
    window_seed: str | None = None,
) -> CalibrationResult:
    """Compute calibration metrics.

    Args:
        probs: Predicted probabilities
        outcomes: Binary outcomes
        num_bins: Number of bins
        min_samples: Minimum total samples
        min_samples_per_bin: Minimum samples per bin
        window_seed: Seed for deterministic bin edge jitter (anti-gaming)

    Returns:
        CalibrationResult with all metrics
    """
    if len(probs) < min_samples:
        return CalibrationResult(
            intercept=0.0,
            slope=1.0,
            score=0.5,
            bins_used=0,
        )

    a, b = fit_calibration_curve(probs, outcomes, num_bins, min_samples_per_bin, window_seed)
    score = calibration_score(a, b)

    # Count bins used (with same jittered edges for consistency)
    bin_edges = get_jittered_bin_edges(num_bins, window_seed)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    bin_counts = np.bincount(bin_indices, minlength=num_bins)
    bins_used = int((bin_counts >= min_samples_per_bin).sum())

    return CalibrationResult(
        intercept=a,
        slope=b,
        score=score,
        bins_used=bins_used,
    )


def compute_calibration_batch(
    probs: NDArray[np.float64],
    outcomes: NDArray[np.int8],
    miner_ids: NDArray[np.int64],
    num_bins: int = 10,
    min_samples: int = 30,
    min_samples_per_bin: int = 5,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Compute calibration scores for multiple miners.

    Args:
        probs: All predicted probabilities
        outcomes: All binary outcomes
        miner_ids: Miner ID for each prediction
        num_bins: Number of bins
        min_samples: Minimum samples per miner
        min_samples_per_bin: Minimum samples per bin

    Returns:
        (unique_miner_ids, calibration_scores)
    """
    unique_miners = np.unique(miner_ids)
    scores = np.full(len(unique_miners), 0.5)

    for i, mid in enumerate(unique_miners):
        mask = miner_ids == mid
        if mask.sum() >= min_samples:
            result = compute_calibration(
                probs[mask],
                outcomes[mask],
                num_bins,
                min_samples,
                min_samples_per_bin,
            )
            scores[i] = result.score

    return unique_miners, scores


__all__ = [
    "CalibrationResult",
    "get_jittered_bin_edges",
    "logit",
    "fit_calibration_curve",
    "calibration_score",
    "compute_calibration",
    "compute_calibration_batch",
]
