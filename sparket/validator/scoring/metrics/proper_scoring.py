"""Proper scoring rules: Brier score and log-loss.

Proper scoring rules reward honest, calibrated probability forecasts.
They are "proper" in the sense that the best expected score is achieved
by reporting your true beliefs.

- Brier score: Mean squared error of probability forecasts
- Log-loss: Negative log-likelihood of realized outcome
- PSS: Probability Skill Score (relative improvement over baseline)

Two types of PSS:
1. PSS vs Closing: Compares miner to closing line (economic baseline)
2. PSS vs Matched: Compares miner to snapshot at submission time (fair skill comparison)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# Small epsilon to prevent log(0)
EPS = 1e-9


@dataclass
class ProperScoringResult:
    """Result of proper scoring computation."""

    brier_miner: float
    brier_truth: float
    logloss_miner: float
    logloss_truth: float
    pss_brier: float
    pss_log: float


@dataclass
class OutcomeScore:
    """Outcome-based score for a single submission.
    
    Contains both absolute accuracy (vs outcome) and relative skill (vs baselines).
    """
    # Absolute accuracy vs actual outcome
    brier: float              # Miner's Brier score (lower = better)
    logloss: float            # Miner's log-loss (lower = better)
    
    # PSS vs matched snapshot (fair comparison at submission time)
    pss_brier_matched: Optional[float]  # None if no matched snapshot
    pss_log_matched: Optional[float]
    
    # PSS vs closing line (economic baseline)
    pss_brier_close: Optional[float]    # None if closing not yet available
    pss_log_close: Optional[float]


def brier_score(p_forecast: NDArray[np.float64], outcome: NDArray[np.int8]) -> float:
    """Compute Brier score for a single forecast.

    Brier = Σ_k (p_k - y_k)²

    Lower is better (0 = perfect, 2 = worst for binary).

    Args:
        p_forecast: Probability vector shape (K,), should sum to ~1
        outcome: One-hot outcome vector shape (K,), exactly one 1

    Returns:
        Brier score (2.0 = worst/penalty for invalid input)
    """
    # Check for invalid input - return worst score (don't crash)
    if not np.all(np.isfinite(p_forecast)):
        return 2.0  # Worst possible Brier score

    # Normalize to sum to 1
    p_sum = p_forecast.sum()
    if p_sum <= 0:
        return 2.0  # Worst possible Brier score

    p_norm = p_forecast / p_sum

    return float(np.sum((p_norm - outcome) ** 2))


def brier_score_batch(
    forecasts: NDArray[np.float64],
    outcomes: NDArray[np.int8],
) -> NDArray[np.float64]:
    """Compute Brier scores for a batch of forecasts.

    Args:
        forecasts: Shape (N, K) array of probability vectors
        outcomes: Shape (N, K) array of one-hot outcome vectors

    Returns:
        Shape (N,) array of Brier scores
    """
    # Normalize each row to sum to 1
    row_sums = forecasts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    p_norm = forecasts / row_sums

    return np.sum((p_norm - outcomes) ** 2, axis=1)


def log_loss(p_forecast: NDArray[np.float64], outcome: NDArray[np.int8]) -> float:
    """Compute log loss (cross-entropy) for a single forecast.

    LogLoss = -log(p_k*) where k* is the realized outcome

    Lower is better (0 = perfect when p=1).

    Args:
        p_forecast: Probability vector shape (K,)
        outcome: One-hot outcome vector shape (K,)

    Returns:
        Log loss (always non-negative, large penalty for invalid input)
    """
    # Check for invalid input - return penalty score (don't crash)
    if not np.all(np.isfinite(p_forecast)):
        return -np.log(EPS)  # Worst possible log-loss

    # Normalize to sum to 1 (consistent with brier_score)
    p_sum = p_forecast.sum()
    if p_sum <= 0:
        return -np.log(EPS)  # Worst possible log-loss

    p_norm = p_forecast / p_sum

    # Get probability of realized outcome
    p_realized = np.sum(p_norm * outcome)

    # Clamp to prevent log(0)
    p_clamped = np.clip(p_realized, EPS, 1.0 - EPS)

    return float(-np.log(p_clamped))


def log_loss_batch(
    forecasts: NDArray[np.float64],
    outcomes: NDArray[np.int8],
) -> NDArray[np.float64]:
    """Compute log loss for a batch of forecasts.

    Args:
        forecasts: Shape (N, K) array of probability vectors
        outcomes: Shape (N, K) array of one-hot outcome vectors

    Returns:
        Shape (N,) array of log loss values
    """
    # Normalize each row to sum to 1 (consistent with brier_score_batch)
    row_sums = forecasts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    p_norm = forecasts / row_sums

    # Get probability of realized outcome for each row
    p_realized = np.sum(p_norm * outcomes, axis=1)

    # Clamp and compute log loss
    p_clamped = np.clip(p_realized, EPS, 1.0 - EPS)
    return -np.log(p_clamped)


def pss(miner_score: float, truth_score: float) -> float:
    """Compute Probability Skill Score.

    PSS = 1 - (miner_score / truth_score)

    Interpretation:
    - PSS > 0: Miner beats ground truth
    - PSS = 0: Miner equals ground truth
    - PSS < 0: Miner worse than ground truth

    Args:
        miner_score: Miner's score (Brier or log-loss)
        truth_score: Ground truth score

    Returns:
        PSS value
    """
    if truth_score == 0:
        return 0.0

    return 1.0 - (miner_score / truth_score)


def pss_batch(
    miner_scores: NDArray[np.float64],
    truth_scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute PSS for a batch of scores.

    Args:
        miner_scores: Shape (N,) array of miner scores
        truth_scores: Shape (N,) array of truth scores

    Returns:
        Shape (N,) array of PSS values
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = miner_scores / truth_scores
        result = 1.0 - ratio
        result = np.where(truth_scores == 0, 0.0, result)
        result = np.nan_to_num(result, nan=0.0)
    return result


def compute_proper_scoring(
    miner_probs: NDArray[np.float64],
    truth_probs: NDArray[np.float64],
    outcome: NDArray[np.int8],
) -> ProperScoringResult:
    """Compute all proper scoring metrics for a single submission.

    Args:
        miner_probs: Miner's probability vector shape (K,)
        truth_probs: Ground truth probability vector shape (K,)
        outcome: One-hot outcome vector shape (K,)

    Returns:
        ProperScoringResult with all metrics
    """
    brier_m = brier_score(miner_probs, outcome)
    brier_t = brier_score(truth_probs, outcome)
    ll_m = log_loss(miner_probs, outcome)
    ll_t = log_loss(truth_probs, outcome)

    return ProperScoringResult(
        brier_miner=brier_m,
        brier_truth=brier_t,
        logloss_miner=ll_m,
        logloss_truth=ll_t,
        pss_brier=pss(brier_m, brier_t),
        pss_log=pss(ll_m, ll_t),
    )


def compute_proper_scoring_batch(
    miner_probs: NDArray[np.float64],
    truth_probs: NDArray[np.float64],
    outcomes: NDArray[np.int8],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute proper scoring metrics for a batch.

    Args:
        miner_probs: Shape (N, K) miner probability vectors
        truth_probs: Shape (N, K) truth probability vectors
        outcomes: Shape (N, K) one-hot outcome vectors

    Returns:
        (brier_scores, logloss_scores, pss_brier, pss_log) each shape (N,)
    """
    brier_m = brier_score_batch(miner_probs, outcomes)
    brier_t = brier_score_batch(truth_probs, outcomes)
    ll_m = log_loss_batch(miner_probs, outcomes)
    ll_t = log_loss_batch(truth_probs, outcomes)

    return brier_m, ll_m, pss_batch(brier_m, brier_t), pss_batch(ll_m, ll_t)


def outcome_to_vector(result_idx: int, num_outcomes: int) -> NDArray[np.int8]:
    """Convert outcome index to one-hot vector.

    Args:
        result_idx: Index of the realized outcome
        num_outcomes: Total number of possible outcomes

    Returns:
        One-hot vector shape (num_outcomes,)
    """
    vec = np.zeros(num_outcomes, dtype=np.int8)
    vec[result_idx] = 1
    return vec


def compute_outcome_score(
    miner_probs: NDArray[np.float64],
    outcome: NDArray[np.int8],
    matched_probs: Optional[NDArray[np.float64]] = None,
    closing_probs: Optional[NDArray[np.float64]] = None,
) -> OutcomeScore:
    """Compute comprehensive outcome-based score for a submission.
    
    Provides both:
    1. Absolute accuracy (vs actual outcome)
    2. Relative skill (vs matched snapshot and closing line)
    
    Args:
        miner_probs: Miner's probability vector shape (K,)
        outcome: One-hot outcome vector shape (K,)
        matched_probs: Snapshot probs at submission time (for fair PSS)
        closing_probs: Closing line probs (for economic PSS)
        
    Returns:
        OutcomeScore with all metrics
    """
    # Absolute accuracy
    brier = brier_score(miner_probs, outcome)
    ll = log_loss(miner_probs, outcome)
    
    # PSS vs matched snapshot
    pss_brier_matched = None
    pss_log_matched = None
    if matched_probs is not None:
        matched_brier = brier_score(matched_probs, outcome)
        matched_ll = log_loss(matched_probs, outcome)
        pss_brier_matched = pss(brier, matched_brier)
        pss_log_matched = pss(ll, matched_ll)
    
    # PSS vs closing line
    pss_brier_close = None
    pss_log_close = None
    if closing_probs is not None:
        close_brier = brier_score(closing_probs, outcome)
        close_ll = log_loss(closing_probs, outcome)
        pss_brier_close = pss(brier, close_brier)
        pss_log_close = pss(ll, close_ll)
    
    return OutcomeScore(
        brier=brier,
        logloss=ll,
        pss_brier_matched=pss_brier_matched,
        pss_log_matched=pss_log_matched,
        pss_brier_close=pss_brier_close,
        pss_log_close=pss_log_close,
    )


def compute_outcome_scores_batch(
    miner_probs: NDArray[np.float64],
    outcomes: NDArray[np.int8],
    matched_probs: Optional[NDArray[np.float64]] = None,
    closing_probs: Optional[NDArray[np.float64]] = None,
) -> Tuple[
    NDArray[np.float64],  # brier
    NDArray[np.float64],  # logloss
    Optional[NDArray[np.float64]],  # pss_brier_matched
    Optional[NDArray[np.float64]],  # pss_log_matched
    Optional[NDArray[np.float64]],  # pss_brier_close
    Optional[NDArray[np.float64]],  # pss_log_close
]:
    """Compute outcome scores for a batch of submissions.
    
    Vectorized version for high performance.
    
    Args:
        miner_probs: Shape (N, K) miner probability vectors
        outcomes: Shape (N, K) one-hot outcome vectors
        matched_probs: Shape (N, K) snapshot probs at submission time (optional)
        closing_probs: Shape (N, K) closing line probs (optional)
        
    Returns:
        Tuple of score arrays, None for unavailable metrics
    """
    # Absolute accuracy
    brier = brier_score_batch(miner_probs, outcomes)
    ll = log_loss_batch(miner_probs, outcomes)
    
    # PSS vs matched
    pss_brier_matched = None
    pss_log_matched = None
    if matched_probs is not None:
        matched_brier = brier_score_batch(matched_probs, outcomes)
        matched_ll = log_loss_batch(matched_probs, outcomes)
        pss_brier_matched = pss_batch(brier, matched_brier)
        pss_log_matched = pss_batch(ll, matched_ll)
    
    # PSS vs closing
    pss_brier_close = None
    pss_log_close = None
    if closing_probs is not None:
        close_brier = brier_score_batch(closing_probs, outcomes)
        close_ll = log_loss_batch(closing_probs, outcomes)
        pss_brier_close = pss_batch(brier, close_brier)
        pss_log_close = pss_batch(ll, close_ll)
    
    return (
        brier,
        ll,
        pss_brier_matched,
        pss_log_matched,
        pss_brier_close,
        pss_log_close,
    )


__all__ = [
    "ProperScoringResult",
    "OutcomeScore",
    "brier_score",
    "brier_score_batch",
    "log_loss",
    "log_loss_batch",
    "pss",
    "pss_batch",
    "compute_proper_scoring",
    "compute_proper_scoring_batch",
    "compute_outcome_score",
    "compute_outcome_scores_batch",
    "outcome_to_vector",
]
