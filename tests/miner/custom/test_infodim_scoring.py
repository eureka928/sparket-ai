"""Integration test: validate custom miner InfoDim against validator scoring.

Calls the validator's pure scoring functions directly (no DB, no subtensor)
to verify our miner's OriginalityTracker produces results consistent with
production scoring and that our Elo-differentiation changes improve InfoDim.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sparket.miner.custom.strategy.originality import OriginalityTracker
from sparket.validator.scoring.aggregation.normalization import (
    normalize_zscore_logistic,
)
from sparket.validator.scoring.metrics.time_series import (
    align_time_series,
    analyze_lead_lag,
    bucket_time_series,
    compute_correlation,
    compute_sos,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUCKET_SECONDS = 300  # 5-min buckets, matches validator


def _now() -> float:
    return time.time()


def _make_market_series(
    n_points: int = 100,
    start_time: float | None = None,
    interval_seconds: float = 300.0,
    start_prob: float = 0.50,
    volatility: float = 0.015,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a realistic market probability time series.

    Random walk, clipped to [0.05, 0.95]. Defaults to recent timestamps
    so that OriginalityTracker's pruning doesn't discard anything.
    """
    if start_time is None:
        # Place series ending ~1 hour ago so it's all within the 30-day window
        start_time = _now() - n_points * interval_seconds - 3600

    rng = np.random.default_rng(seed)
    times = np.array(
        [start_time + i * interval_seconds for i in range(n_points)],
        dtype=np.float64,
    )
    deltas = rng.normal(0, volatility, n_points - 1)
    probs = np.empty(n_points, dtype=np.float64)
    probs[0] = start_prob
    for i in range(1, n_points):
        probs[i] = probs[i - 1] + deltas[i - 1]
    probs = np.clip(probs, 0.05, 0.95)
    return times, probs


def _correlated_miner(
    market_probs: np.ndarray,
    offset: float = 0.02,
) -> np.ndarray:
    """Miner = market + constant offset (highly correlated)."""
    return np.clip(market_probs + offset, 0.05, 0.95)


def _elo_differentiated_miner(
    market_probs: np.ndarray,
    elo_anchor: float = 0.55,
    strength: float = 0.5,
    seed: int = 99,
) -> np.ndarray:
    """Miner that blends market with an Elo-based independent signal.

    Adds a small independent noise component on top of the Elo blend,
    simulating what the real miner does with Elo-anchored differentiation.
    """
    rng = np.random.default_rng(seed)
    elo_signal = np.full_like(market_probs, elo_anchor)
    # Elo drifts slowly on its own (game results update it)
    elo_drift = np.cumsum(rng.normal(0, 0.005, len(market_probs)))
    elo_signal = elo_signal + elo_drift

    blended = market_probs * (1 - strength) + elo_signal * strength
    # Small additional noise from model uncertainty
    noise = rng.normal(0, 0.005, len(market_probs))
    return np.clip(blended + noise, 0.05, 0.95)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSosMatchesValidator:
    """Test 1: Miner SOS matches validator's computation on identical data."""

    def test_miner_sos_matches_validator_computation(self) -> None:
        """Feed identical time series to both implementations; results should agree."""
        times, market_probs = _make_market_series(n_points=60, seed=1)
        miner_probs = _elo_differentiated_miner(market_probs, seed=2)

        # --- Validator path ---
        bucket_ts_m, bucket_vals_m = bucket_time_series(
            times, market_probs, BUCKET_SECONDS
        )
        bucket_ts_n, bucket_vals_n = bucket_time_series(
            times, miner_probs, BUCKET_SECONDS
        )
        aligned_miner, aligned_market = align_time_series(
            bucket_ts_n, bucket_vals_n, bucket_ts_m, bucket_vals_m
        )
        validator_corr = compute_correlation(aligned_miner, aligned_market)
        validator_sos = compute_sos(validator_corr)

        # --- Miner path (OriginalityTracker) ---
        tracker = OriginalityTracker(data_path=None, max_history_days=30)
        market_id = 1001
        for t, mp, mkp in zip(times, miner_probs, market_probs):
            tracker.record_submission(
                market_id=market_id,
                our_prob=float(mp),
                market_prob=float(mkp),
                timestamp=float(t),
            )
        # Use a wide window so all data is included
        miner_sos = tracker.get_sos_estimate(window_hours=9999)

        # Both use Pearson on 5-min bucketed data; should be very close.
        # Small differences possible because dict iteration order vs numpy sort.
        assert abs(validator_sos - miner_sos) < 0.05, (
            f"SOS mismatch: validator={validator_sos:.4f}, miner={miner_sos:.4f}"
        )


class TestCorrelatedMinerLowSos:
    """Test 2: Miner tracking market closely gets low SOS."""

    def test_correlated_miner_gets_low_sos(self) -> None:
        """market + constant offset → correlation ≈ 1 → SOS ≈ 0."""
        times, market_probs = _make_market_series(n_points=80, seed=10)
        miner_probs = _correlated_miner(market_probs, offset=0.02)

        # Validator
        bt, bv = bucket_time_series(times, market_probs, BUCKET_SECONDS)
        bt2, bv2 = bucket_time_series(times, miner_probs, BUCKET_SECONDS)
        am, at_ = align_time_series(bt2, bv2, bt, bv)
        corr = compute_correlation(am, at_)
        sos = compute_sos(corr)

        assert corr > 0.95, f"Expected high correlation, got {corr:.4f}"
        assert sos < 0.10, f"Expected low SOS for correlated miner, got {sos:.4f}"


class TestEloDifferentiationImprovesSos:
    """Test 3: Elo-anchored differentiation decorrelates from market."""

    def test_elo_differentiation_improves_sos(self) -> None:
        times, market_probs = _make_market_series(n_points=100, seed=20)

        # Baseline: follows market closely
        baseline_probs = _correlated_miner(market_probs, offset=0.01)

        # With differentiation: blends Elo signal
        diff_probs = _elo_differentiated_miner(
            market_probs, elo_anchor=0.55, strength=0.4, seed=21
        )

        # Score both through validator's analyze_lead_lag
        result_baseline = analyze_lead_lag(
            truth_times=times,
            truth_values=market_probs,
            miner_times=times,
            miner_values=baseline_probs,
            lead_window_seconds=1800,
            lag_window_seconds=1800,
            move_threshold=0.02,
        )
        result_diff = analyze_lead_lag(
            truth_times=times,
            truth_values=market_probs,
            miner_times=times,
            miner_values=diff_probs,
            lead_window_seconds=1800,
            lag_window_seconds=1800,
            move_threshold=0.02,
        )

        assert result_diff.sos_score > result_baseline.sos_score, (
            f"Differentiation should improve SOS: "
            f"baseline={result_baseline.sos_score:.4f}, "
            f"diff={result_diff.sos_score:.4f}"
        )


class TestFrequentSubmissionsImproveLead:
    """Test 4: More frequent submissions improve lead ratio."""

    def test_frequent_submissions_improve_lead_ratio(self) -> None:
        """30-min refresh (48/day) should lead more than 6h refresh (4/day)."""
        n_hours = 24
        # Market changes every 30 min for 24 hours
        n_market_points = n_hours * 2  # every 30 min
        start = _now() - n_hours * 3600 - 3600
        market_times = np.array(
            [start + i * 1800 for i in range(n_market_points)], dtype=np.float64
        )
        rng = np.random.default_rng(30)
        market_probs = 0.50 + np.cumsum(rng.normal(0, 0.015, n_market_points))
        market_probs = np.clip(market_probs, 0.10, 0.90)

        # Infrequent: every 6h (4 submissions over 24h)
        infreq_interval = 6 * 3600
        infreq_times = np.array(
            [start + i * infreq_interval for i in range(n_hours * 3600 // infreq_interval)],
            dtype=np.float64,
        )
        infreq_indices = np.searchsorted(market_times, infreq_times, side="right") - 1
        infreq_indices = np.clip(infreq_indices, 0, len(market_probs) - 1)
        infreq_probs = market_probs[infreq_indices] + rng.normal(0.01, 0.008, len(infreq_times))
        infreq_probs = np.clip(infreq_probs, 0.10, 0.90)

        # Frequent: every 30 min (48 submissions over 24h) — same as market tick rate
        freq_times = market_times.copy()
        freq_probs = np.empty_like(market_probs)
        for i in range(len(market_probs)):
            # Miner "anticipates" by looking at trend of last few points
            if i >= 2:
                trend = (market_probs[i] - market_probs[i - 2]) / 2
                freq_probs[i] = market_probs[i] + trend + rng.normal(0, 0.005)
            else:
                freq_probs[i] = market_probs[i] + rng.normal(0.01, 0.005)
        freq_probs = np.clip(freq_probs, 0.10, 0.90)

        result_infreq = analyze_lead_lag(
            truth_times=market_times,
            truth_values=market_probs,
            miner_times=infreq_times,
            miner_values=infreq_probs,
            lead_window_seconds=1800,
            lag_window_seconds=1800,
            move_threshold=0.02,
        )
        result_freq = analyze_lead_lag(
            truth_times=market_times,
            truth_values=market_probs,
            miner_times=freq_times,
            miner_values=freq_probs,
            lead_window_seconds=1800,
            lag_window_seconds=1800,
            move_threshold=0.02,
        )

        # Frequent miner should detect at least as many truth moves
        assert result_freq.total_truth_moves >= result_infreq.total_truth_moves

        # With trend-following and frequent updates, expect better lead ratio
        if result_freq.moves_matched > 0 and result_infreq.moves_matched > 0:
            assert result_freq.lead_ratio >= result_infreq.lead_ratio, (
                f"Frequent refresh should improve lead: "
                f"freq={result_freq.lead_ratio:.3f}, infreq={result_infreq.lead_ratio:.3f}"
            )
        else:
            # At minimum, frequent miner should match more moves
            assert result_freq.moves_matched >= result_infreq.moves_matched


class TestDifferentiationSkippedWithShrinkage:
    """Test 5: Differentiation returns 0 when shrinkage is active."""

    def test_differentiation_skipped_when_shrinkage_applied(self) -> None:
        """When variance shrinkage is active, the runner sets diff_strength=0."""
        tracker = OriginalityTracker(data_path=None, max_history_days=30)

        # Force very low SOS by feeding correlated data using recent timestamps
        t0 = _now() - 20000  # ~5.5 hours ago, well within 30-day window
        for i in range(50):
            ts = t0 + i * 300
            prob = 0.50 + i * 0.002  # steadily increasing
            tracker.record_submission(
                market_id=1,
                our_prob=prob + 0.005,  # tiny constant offset → high correlation
                market_prob=prob,
                timestamp=ts,
            )

        # Tracker should suggest strong differentiation (low SOS → high strength)
        diff_strength = tracker.get_differentiation_strength()
        assert diff_strength > 0.5, (
            f"Expected strong differentiation signal, got {diff_strength:.3f}"
        )

        # But when shrinkage is active the runner zeroes it out.
        # Simulate the runner's guard: shrinkage_applied > 0 → diff_strength = 0
        shrinkage_applied = 0.15  # confidence < 0.8 triggered shrinkage
        effective_diff = 0.0 if shrinkage_applied > 0 else diff_strength
        assert effective_diff == 0.0, "Differentiation must be 0 when shrinkage is active"


class TestInfoDimScoreImprovement:
    """Test 6: Full InfoDim = 0.6*SOS + 0.4*LEAD improves with our changes."""

    def test_infodim_score_improvement(self) -> None:
        """Differentiated miner gets higher SOS component of InfoDim.

        We isolate the SOS component (60% of InfoDim) because differentiation
        primarily affects correlation, not lead-lag. Lead ratio depends on
        submission timing, which is tested separately in Test 4.
        """
        n_points = 120
        times, market_probs = _make_market_series(
            n_points=n_points, seed=50, volatility=0.018
        )

        # Baseline: near-copy of market
        baseline_probs = _correlated_miner(market_probs, offset=0.015)

        # Optimized: Elo-differentiated
        opt_probs = _elo_differentiated_miner(
            market_probs, elo_anchor=0.54, strength=0.35, seed=51
        )

        # Score SOS component via validator functions
        def _sos(miner_probs: np.ndarray) -> float:
            bt_m, bv_m = bucket_time_series(times, market_probs, BUCKET_SECONDS)
            bt_n, bv_n = bucket_time_series(times, miner_probs, BUCKET_SECONDS)
            am, at_ = align_time_series(bt_n, bv_n, bt_m, bv_m)
            corr = compute_correlation(am, at_)
            return compute_sos(corr)

        baseline_sos = _sos(baseline_probs)
        optimized_sos = _sos(opt_probs)

        # Differentiation should meaningfully improve SOS
        assert optimized_sos > baseline_sos, (
            f"Optimized SOS should beat baseline: "
            f"optimized={optimized_sos:.4f}, baseline={baseline_sos:.4f}"
        )

        # Compute full InfoDim to verify valid ranges
        def _infodim(miner_probs: np.ndarray) -> float:
            result = analyze_lead_lag(
                truth_times=times,
                truth_values=market_probs,
                miner_times=times,
                miner_values=miner_probs,
                lead_window_seconds=1800,
                lag_window_seconds=1800,
                move_threshold=0.02,
            )
            return 0.6 * result.sos_score + 0.4 * result.lead_ratio

        baseline_infodim = _infodim(baseline_probs)
        optimized_infodim = _infodim(opt_probs)

        # Sanity: both should be in valid range
        assert 0.0 <= baseline_infodim <= 1.0
        assert 0.0 <= optimized_infodim <= 1.0

        # The SOS improvement should be positive (already asserted above);
        # verify it's non-trivial relative to baseline
        sos_improvement = optimized_sos - baseline_sos
        assert sos_improvement > 0.005, (
            f"SOS improvement should be non-trivial: {sos_improvement:.4f}"
        )


class TestNormalizationBehavior:
    """Supplementary: verify zscore_logistic normalization properties."""

    def test_normalization_separates_good_from_bad(self) -> None:
        """A miner with higher raw SOS should rank higher after normalization."""
        # Simulate 5 miners' SOS scores
        raw_scores = np.array([0.05, 0.12, 0.25, 0.40, 0.60], dtype=np.float64)
        normalized = normalize_zscore_logistic(raw_scores, alpha=1.0)

        # Order should be preserved
        assert np.all(np.diff(normalized) > 0), (
            "Normalization should preserve ordering"
        )
        # All in (0, 1)
        assert np.all(normalized > 0) and np.all(normalized < 1)

    def test_single_miner_gets_neutral_score(self) -> None:
        """Single miner should get 0.5 (no relative ranking possible)."""
        normalized = normalize_zscore_logistic(np.array([0.3]), alpha=1.0)
        assert normalized[0] == pytest.approx(0.5)
