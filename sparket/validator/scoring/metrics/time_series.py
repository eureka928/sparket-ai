"""Time series metrics: correlation and lead-lag analysis.

These metrics measure the informational content of miner submissions:
- Correlation: How similar are miner movements to ground truth?
- Lead-lag: Does the miner lead or follow the market?

Low correlation = originality (independent information)
High lead ratio = miner anticipates market moves
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class LeadLagResult:
    """Result of lead-lag analysis."""

    moves_led: int
    moves_matched: int
    total_truth_moves: int
    lead_ratio: float
    sos_score: float  # Source of Signal (1 - |correlation|)


def bucket_time_series(
    timestamps: NDArray[np.float64],
    values: NDArray[np.float64],
    bucket_seconds: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bucket a time series by flooring timestamps.

    Takes the last value in each bucket (forward-fill semantics).

    Args:
        timestamps: Array of Unix timestamps
        values: Array of values
        bucket_seconds: Bucket size in seconds

    Returns:
        (bucketed_timestamps, bucketed_values) - unique buckets with last value
    """
    if len(timestamps) == 0:
        return np.array([]), np.array([])

    # Floor to bucket boundaries
    bucket_ts = timestamps - (timestamps % bucket_seconds)

    # Sort by bucket (then by original timestamp within bucket)
    sort_idx = np.lexsort((timestamps, bucket_ts))
    sorted_buckets = bucket_ts[sort_idx]
    sorted_values = values[sort_idx]

    # Find unique buckets - take last occurrence (which is last value in bucket due to sort)
    _, unique_idx = np.unique(sorted_buckets[::-1], return_index=True)
    n = len(sorted_buckets)
    unique_idx = n - 1 - unique_idx  # Convert from reversed indices

    return sorted_buckets[unique_idx], sorted_values[unique_idx]


def align_time_series(
    ts1: NDArray[np.float64],
    vals1: NDArray[np.float64],
    ts2: NDArray[np.float64],
    vals2: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Align two time series to common timestamps.

    Args:
        ts1: Timestamps for series 1 (need not be sorted)
        vals1: Values for series 1
        ts2: Timestamps for series 2 (need not be sorted)
        vals2: Values for series 2

    Returns:
        (aligned_vals1, aligned_vals2) at common timestamps
    """
    common_ts = np.intersect1d(ts1, ts2)

    if len(common_ts) == 0:
        return np.array([]), np.array([])

    # Sort both series by timestamp for correct lookup
    sort1 = np.argsort(ts1)
    sort2 = np.argsort(ts2)
    ts1_sorted = ts1[sort1]
    vals1_sorted = vals1[sort1]
    ts2_sorted = ts2[sort2]
    vals2_sorted = vals2[sort2]

    # Find values at common timestamps using sorted arrays
    idx1 = np.searchsorted(ts1_sorted, common_ts)
    idx2 = np.searchsorted(ts2_sorted, common_ts)

    return vals1_sorted[idx1], vals2_sorted[idx2]


def compute_correlation(
    vals1: NDArray[np.float64],
    vals2: NDArray[np.float64],
    min_points: int = 3,
) -> float:
    """Compute Pearson correlation between two aligned time series.

    Args:
        vals1: First value array (already aligned)
        vals2: Second value array (already aligned)
        min_points: Minimum points required

    Returns:
        Correlation coefficient in [-1, 1]
    """
    if len(vals1) < min_points:
        return 0.0

    if vals1.std() == 0 or vals2.std() == 0:
        return 0.0

    corr = np.corrcoef(vals1, vals2)[0, 1]

    if np.isnan(corr):
        return 0.0

    return float(np.clip(corr, -1.0, 1.0))


def compute_sos(correlation: float) -> float:
    """Compute Source of Signal (originality) score.

    SOS = 1 - |correlation|

    Higher SOS = more independent from ground truth = more original.

    Args:
        correlation: Pearson correlation coefficient

    Returns:
        SOS score in [0, 1]
    """
    return 1.0 - abs(correlation)


def detect_moves(
    timestamps: NDArray[np.float64],
    values: NDArray[np.float64],
    threshold: float,
) -> Tuple[NDArray[np.float64], NDArray[np.int8], NDArray[np.float64]]:
    """Detect significant moves in a time series.

    A move is detected when the value changes by more than threshold.

    Args:
        timestamps: Sorted array of timestamps
        values: Array of values aligned with timestamps
        threshold: Minimum change to count as move

    Returns:
        (move_times, directions, magnitudes)
        - move_times: timestamps where moves occurred
        - directions: +1 for increase, -1 for decrease
        - magnitudes: absolute change
    """
    if len(timestamps) < 2:
        return np.array([]), np.array([], dtype=np.int8), np.array([])

    # Compute deltas
    deltas = np.diff(values)
    magnitudes = np.abs(deltas)

    # Find moves
    move_mask = magnitudes >= threshold

    move_times = timestamps[1:][move_mask]
    move_directions = np.sign(deltas[move_mask]).astype(np.int8)
    move_magnitudes = magnitudes[move_mask]

    return move_times, move_directions, move_magnitudes


def analyze_lead_lag(
    truth_times: NDArray[np.float64],
    truth_values: NDArray[np.float64],
    miner_times: NDArray[np.float64],
    miner_values: NDArray[np.float64],
    lead_window_seconds: float,
    lag_window_seconds: float,
    move_threshold: float,
) -> LeadLagResult:
    """Analyze lead-lag relationship between miner and truth.

    For each significant move in truth, check if miner moved same direction
    before (lead) or after (lag).

    Args:
        truth_times: Truth timestamps
        truth_values: Truth values
        miner_times: Miner timestamps
        miner_values: Miner values
        lead_window_seconds: Look-back window for lead detection
        lag_window_seconds: Look-forward window for lag detection
        move_threshold: Minimum change for move

    Returns:
        LeadLagResult with move counts and scores
    """
    # Detect moves
    truth_move_times, truth_dirs, _ = detect_moves(
        truth_times, truth_values, move_threshold
    )
    miner_move_times, miner_dirs, _ = detect_moves(
        miner_times, miner_values, move_threshold
    )

    total_truth_moves = len(truth_move_times)

    if total_truth_moves == 0:
        # Compute correlation on aligned series for SOS
        aligned_miner, aligned_truth = align_time_series(
            miner_times, miner_values, truth_times, truth_values
        )
        corr = compute_correlation(aligned_miner, aligned_truth)
        return LeadLagResult(
            moves_led=0,
            moves_matched=0,
            total_truth_moves=0,
            lead_ratio=0.5,
            sos_score=compute_sos(corr),
        )

    if len(miner_move_times) == 0:
        aligned_miner, aligned_truth = align_time_series(
            miner_times, miner_values, truth_times, truth_values
        )
        corr = compute_correlation(aligned_miner, aligned_truth)
        return LeadLagResult(
            moves_led=0,
            moves_matched=0,
            total_truth_moves=total_truth_moves,
            lead_ratio=0.0,
            sos_score=compute_sos(corr),
        )

    moves_led = 0
    moves_matched = 0

    for i, truth_time in enumerate(truth_move_times):
        truth_dir = truth_dirs[i]

        # Find miner moves with same direction
        same_dir_mask = miner_dirs == truth_dir
        same_dir_times = miner_move_times[same_dir_mask]

        if len(same_dir_times) == 0:
            continue

        lead_start = truth_time - lead_window_seconds
        lag_end = truth_time + lag_window_seconds

        # Check for lead (miner moved before truth)
        lead_mask = (same_dir_times >= lead_start) & (same_dir_times < truth_time)
        found_lead = np.any(lead_mask)

        # Check for any match (lead or lag)
        any_mask = (same_dir_times >= lead_start) & (same_dir_times <= lag_end)
        found_any = np.any(any_mask)

        if found_any:
            moves_matched += 1
            if found_lead:
                moves_led += 1

    # Compute lead ratio
    lead_ratio = moves_led / moves_matched if moves_matched > 0 else 0.5

    # Compute SOS from correlation
    aligned_miner, aligned_truth = align_time_series(
        miner_times, miner_values, truth_times, truth_values
    )
    corr = compute_correlation(aligned_miner, aligned_truth)

    return LeadLagResult(
        moves_led=moves_led,
        moves_matched=moves_matched,
        total_truth_moves=total_truth_moves,
        lead_ratio=lead_ratio,
        sos_score=compute_sos(corr),
    )


__all__ = [
    "LeadLagResult",
    "bucket_time_series",
    "align_time_series",
    "compute_correlation",
    "compute_sos",
    "detect_moves",
    "analyze_lead_lag",
]
