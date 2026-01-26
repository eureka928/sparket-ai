"""Type definitions and constants for the scoring system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, TypedDict


# Decimal precision for scoring calculations
DECIMAL_PRECISION = 28  # Python Decimal default
DECIMAL_PLACES = 8  # Rounding precision for stored values


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


class ScoringError(Exception):
    """Raised when scoring computation fails."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# TypedDicts for data transfer
# ─────────────────────────────────────────────────────────────────────────────


class BookProbability(TypedDict):
    """Probability data from a single sportsbook."""

    sportsbook_id: int
    prob: Decimal
    odds: Decimal
    timestamp: datetime


class BiasEstimate(TypedDict):
    """Bias estimate for a sportsbook."""

    sportsbook_id: int
    sport_id: int
    market_kind: str
    bias_factor: Decimal
    variance: Decimal
    sample_count: int


class ConsensusResult(TypedDict):
    """Result of consensus probability computation."""

    prob_consensus: Decimal
    odds_consensus: Decimal
    contributing_books: int
    min_prob: Decimal
    max_prob: Decimal
    std_dev: Decimal


class SubmissionMetrics(TypedDict, total=False):
    """Metrics computed for a single submission."""

    clv_odds: Decimal
    clv_prob: Decimal
    cle: Decimal
    minutes_to_close: int


class OutcomeMetrics(TypedDict, total=False):
    """Outcome-based metrics for a submission."""

    brier: Decimal
    logloss: Decimal
    pss_brier: Decimal
    pss_log: Decimal


class RollingMetrics(TypedDict, total=False):
    """Rolling aggregate metrics for a miner."""

    n_submissions: int
    n_eff: Decimal
    es_mean: Decimal
    es_std: Decimal
    es_adj: Decimal
    mes_mean: Decimal
    sos_mean: Decimal
    pss_mean: Decimal
    fq_raw: Decimal
    lead_ratio: Decimal


class NormalizedScores(TypedDict, total=False):
    """Normalized [0,1] scores for a miner."""

    fq_score: Decimal
    cal_score: Decimal
    sharp_score: Decimal
    edge_score: Decimal
    mes_score: Decimal
    sos_score: Decimal
    lead_score: Decimal
    forecast_dim: Decimal
    econ_dim: Decimal
    info_dim: Decimal
    skill_score: Decimal


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses for internal use
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CLVResult:
    """Result of CLV/CLE computation."""

    clv_odds: Decimal
    clv_prob: Decimal
    cle: Decimal
    minutes_to_close: int


@dataclass(frozen=True)
class ProperScoringResult:
    """Result of proper scoring rule computation."""

    brier_miner: Decimal
    brier_truth: Decimal
    logloss_miner: Decimal
    logloss_truth: Decimal
    pss_brier: Decimal
    pss_log: Decimal


@dataclass(frozen=True)
class CalibrationResult:
    """Result of calibration analysis."""

    a: Decimal  # Intercept
    b: Decimal  # Slope
    cal_score: Decimal  # 1 / (1 + |b-1| + |a|)
    bins_used: int


@dataclass(frozen=True)
class TimeSeriesCorrelation:
    """Result of time series correlation analysis."""

    correlation: Decimal
    sos_score: Decimal  # 1 - |correlation|
    n_observations: int


@dataclass(frozen=True)
class LeadLagResult:
    """Result of lead-lag analysis."""

    moves_led: int
    moves_matched: int
    lead_ratio: Decimal
    n_truth_moves: int


# ─────────────────────────────────────────────────────────────────────────────
# Job status enum
# ─────────────────────────────────────────────────────────────────────────────

JobStatus = Literal["pending", "running", "completed", "failed"]


__all__ = [
    "DECIMAL_PRECISION",
    "DECIMAL_PLACES",
    "ValidationError",
    "ScoringError",
    "BookProbability",
    "BiasEstimate",
    "ConsensusResult",
    "SubmissionMetrics",
    "OutcomeMetrics",
    "RollingMetrics",
    "NormalizedScores",
    "CLVResult",
    "ProperScoringResult",
    "CalibrationResult",
    "TimeSeriesCorrelation",
    "LeadLagResult",
    "JobStatus",
]

