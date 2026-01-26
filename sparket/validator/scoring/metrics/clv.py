"""Closing Line Value (CLV) and Closing Line Edge (CLE) calculations.

CLV measures how a miner's submitted odds compare to the closing line,
which is considered the most efficient market price.

- CLV_odds: (O_miner - O_close) / O_close
- CLV_prob: (p_close - p_miner) / p_close
- CLE: O_miner * p_close - 1 (expected value of betting at miner's odds)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class CLVResult:
    """Result of CLV/CLE computation."""

    clv_odds: float
    clv_prob: float
    cle: float
    minutes_to_close: int


def compute_clv(
    miner_odds: float,
    miner_prob: float,
    truth_odds: float,
    truth_prob: float,
    submitted_ts: float,
    event_start_ts: float,
    cle_min: float = -1.0,
    cle_max: float = 1.0,
) -> CLVResult:
    """Compute CLV/CLE metrics for a submission.

    Args:
        miner_odds: Miner's submitted decimal odds
        miner_prob: Miner's implied probability (1/odds)
        truth_odds: Ground truth closing decimal odds
        truth_prob: Ground truth closing probability
        submitted_ts: Unix timestamp of submission
        event_start_ts: Unix timestamp of event start
        cle_min: Minimum CLE value (clamped)
        cle_max: Maximum CLE value (clamped)

    Returns:
        CLVResult with computed metrics
    """
    # CLV (odds-based): positive means miner offered better odds than close
    if truth_odds <= 1:
        clv_odds = 0.0
    else:
        clv_odds = (miner_odds - truth_odds) / truth_odds

    # CLV (probability-based): positive means miner had lower prob than truth
    if truth_prob <= 0:
        clv_prob = 0.0
    else:
        clv_prob = (truth_prob - miner_prob) / truth_prob

    # CLE: Expected value of betting at miner's odds if truth_prob is correct
    cle = miner_odds * truth_prob - 1.0
    cle = np.clip(cle, cle_min, cle_max)

    # Minutes to close
    minutes_to_close = max(0, int((event_start_ts - submitted_ts) / 60))

    return CLVResult(
        clv_odds=clv_odds,
        clv_prob=clv_prob,
        cle=cle,
        minutes_to_close=minutes_to_close,
    )


def compute_clv_batch(
    miner_odds: NDArray[np.float64],
    miner_probs: NDArray[np.float64],
    truth_odds: NDArray[np.float64],
    truth_probs: NDArray[np.float64],
    submitted_ts: NDArray[np.float64],
    event_start_ts: NDArray[np.float64],
    cle_min: float = -1.0,
    cle_max: float = 1.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Compute CLV/CLE for a batch of submissions.

    Args:
        miner_odds: Array of miner decimal odds
        miner_probs: Array of miner implied probabilities
        truth_odds: Array of truth decimal odds
        truth_probs: Array of truth probabilities
        submitted_ts: Array of submission timestamps
        event_start_ts: Array of event start timestamps
        cle_min: Minimum CLE value
        cle_max: Maximum CLE value

    Returns:
        (clv_odds, clv_prob, cle, minutes_to_close)
    """
    # CLV odds - avoid division by zero warning
    with np.errstate(divide='ignore', invalid='ignore'):
        clv_odds_raw = (miner_odds - truth_odds) / truth_odds
    clv_odds = np.where(truth_odds > 1, clv_odds_raw, 0.0)
    clv_odds = np.nan_to_num(clv_odds, nan=0.0, posinf=0.0, neginf=0.0)

    # CLV prob - avoid division by zero warning
    with np.errstate(divide='ignore', invalid='ignore'):
        clv_prob_raw = (truth_probs - miner_probs) / truth_probs
    clv_prob = np.where(truth_probs > 0, clv_prob_raw, 0.0)
    clv_prob = np.nan_to_num(clv_prob, nan=0.0, posinf=0.0, neginf=0.0)

    # CLE
    cle = miner_odds * truth_probs - 1.0
    cle = np.clip(cle, cle_min, cle_max)

    # Minutes to close
    minutes_to_close = np.maximum(0, ((event_start_ts - submitted_ts) / 60)).astype(np.int64)

    return clv_odds, clv_prob, cle, minutes_to_close


def compute_mes(clv_prob: float) -> float:
    """Compute Market Efficiency Score from CLV.

    MES = 1 - min(1, |CLV_prob|)

    Higher MES means miner is closer to the efficient market.

    Args:
        clv_prob: CLV in probability space

    Returns:
        MES in [0, 1]
    """
    deviation = min(1.0, abs(clv_prob))
    return 1.0 - deviation


def compute_mes_batch(clv_prob: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute MES for a batch of CLV values.

    Args:
        clv_prob: Array of CLV probability values

    Returns:
        Array of MES values in [0, 1]
    """
    deviation = np.minimum(1.0, np.abs(clv_prob))
    return 1.0 - deviation


__all__ = [
    "CLVResult",
    "compute_clv",
    "compute_clv_batch",
    "compute_mes",
    "compute_mes_batch",
]
