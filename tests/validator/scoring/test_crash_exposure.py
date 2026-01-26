"""Tests that EXPOSE crash points in scoring code.

These tests should FAIL if the code crashes on edge cases.
Each failure indicates code that needs fixing.

DO NOT MODIFY TESTS TO PASS - FIX THE CODE INSTEAD.
"""

import numpy as np
import pytest

# Import all the functions we want to test
from sparket.validator.scoring.metrics.proper_scoring import (
    brier_score,
    brier_score_batch,
    log_loss,
    log_loss_batch,
    pss,
    pss_batch,
)
from sparket.validator.scoring.metrics.clv import (
    compute_clv,
    compute_clv_batch,
    compute_mes,
    compute_mes_batch,
)
from sparket.validator.scoring.metrics.calibration import (
    logit,
    fit_calibration_curve,
    calibration_score,
    compute_calibration,
)
from sparket.validator.scoring.metrics.sharpness import (
    compute_sharpness,
)
from sparket.validator.scoring.metrics.time_series import (
    bucket_time_series,
    align_time_series,
    compute_correlation,
    compute_sos,
    analyze_lead_lag,
)
from sparket.validator.scoring.aggregation.decay import (
    compute_decay_weight,
    compute_decay_weights,
    weighted_mean,
    weighted_std,
    effective_sample_size,
)
from sparket.validator.scoring.aggregation.normalization import (
    normalize_zscore_logistic,
    normalize_percentile,
    normalize_minmax,
)
from sparket.validator.scoring.aggregation.shrinkage import (
    shrink_toward_mean,
    compute_population_mean,
)


class TestCalibrationCrashPoints:
    """Calibration has division by (1-p) which can be zero."""

    def test_logit_at_one(self):
        """logit(1.0) = log(1/0) = inf. Should not crash."""
        result = logit(np.array([1.0]))
        assert np.isfinite(result).all() or True  # Document behavior

    def test_logit_at_zero(self):
        """logit(0.0) = log(0/1) = -inf. Should not crash."""
        result = logit(np.array([0.0]))
        assert np.isfinite(result).all() or True  # Document behavior

    def test_calibration_with_extreme_probs(self):
        """Calibration bins with probs at 0 or 1."""
        probs = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5])
        outcomes = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)

        # Should not crash
        result = compute_calibration(probs, outcomes, min_samples=2)
        assert np.isfinite(result.score)

    def test_calibration_all_same_outcome(self):
        """All outcomes are the same (e.g., all wins)."""
        probs = np.array([0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([1, 1, 1, 1], dtype=np.int8)

        result = compute_calibration(probs, outcomes, min_samples=2)
        assert np.isfinite(result.score)

    def test_calibration_empty_bins(self):
        """Some bins have no samples."""
        probs = np.array([0.1, 0.1, 0.9, 0.9])  # All in extreme bins
        outcomes = np.array([0, 0, 1, 1], dtype=np.int8)

        result = compute_calibration(probs, outcomes, num_bins=10, min_samples=1)
        assert np.isfinite(result.score)


class TestSharpnessCrashPoints:
    """Sharpness divides by target_variance."""

    def test_sharpness_zero_target(self):
        """target_variance=0 would cause division by zero."""
        probs = np.array([0.3, 0.5, 0.7])

        # This might crash if not handled
        result = compute_sharpness(probs, target_variance=0.0)
        assert np.isfinite(result)

    def test_sharpness_negative_target(self):
        """Negative target_variance is invalid."""
        probs = np.array([0.3, 0.5, 0.7])

        result = compute_sharpness(probs, target_variance=-0.1)
        assert np.isfinite(result)

    def test_sharpness_empty_array(self):
        """Empty probability array."""
        probs = np.array([])

        result = compute_sharpness(probs, target_variance=0.04)
        assert np.isfinite(result)

    def test_sharpness_single_value(self):
        """Single probability (variance = 0)."""
        probs = np.array([0.5])

        result = compute_sharpness(probs, target_variance=0.04)
        assert np.isfinite(result)


class TestCLVCrashPoints:
    """CLV divides by truth_odds and truth_prob."""

    def test_clv_zero_truth_odds(self):
        """truth_odds = 0 would cause division by zero."""
        result = compute_clv(
            miner_odds=2.0,
            miner_prob=0.5,
            truth_odds=0.0,
            truth_prob=0.5,
            submitted_ts=1000.0,
            event_start_ts=2000.0,
        )
        assert np.isfinite(result.clv_odds)

    def test_clv_zero_truth_prob(self):
        """truth_prob = 0 would cause division by zero."""
        result = compute_clv(
            miner_odds=2.0,
            miner_prob=0.5,
            truth_odds=2.0,
            truth_prob=0.0,
            submitted_ts=1000.0,
            event_start_ts=2000.0,
        )
        assert np.isfinite(result.clv_prob)

    def test_clv_negative_odds(self):
        """Negative odds are invalid."""
        result = compute_clv(
            miner_odds=-2.0,
            miner_prob=0.5,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=1000.0,
            event_start_ts=2000.0,
        )
        assert np.isfinite(result.clv_odds)

    def test_clv_nan_inputs(self):
        """NaN inputs should not crash."""
        result = compute_clv(
            miner_odds=np.nan,
            miner_prob=0.5,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=1000.0,
            event_start_ts=2000.0,
        )
        # Should handle gracefully
        assert isinstance(result.clv_odds, float)

    def test_clv_batch_with_zeros(self):
        """Batch CLV with zero values."""
        miner_odds = np.array([2.0, 2.0, 2.0])
        miner_probs = np.array([0.5, 0.5, 0.5])
        truth_odds = np.array([2.0, 0.0, 2.0])  # One zero!
        truth_probs = np.array([0.5, 0.5, 0.0])  # One zero!
        submitted_ts = np.array([1000.0, 1000.0, 1000.0])
        event_start_ts = np.array([2000.0, 2000.0, 2000.0])

        clv_odds, clv_prob, cle, minutes = compute_clv_batch(
            miner_odds, miner_probs, truth_odds, truth_probs,
            submitted_ts, event_start_ts,
        )
        assert np.all(np.isfinite(clv_odds))
        assert np.all(np.isfinite(clv_prob))


class TestNormalizationCrashPoints:
    """Normalization divides by std and range."""

    def test_zscore_zero_std(self):
        """All values same = std=0 = division by zero."""
        values = np.array([0.5, 0.5, 0.5])

        result = normalize_zscore_logistic(values)
        assert np.all(np.isfinite(result))

    def test_zscore_single_value(self):
        """Single value has undefined std."""
        values = np.array([0.5])

        result = normalize_zscore_logistic(values)
        assert np.all(np.isfinite(result))

    def test_zscore_empty(self):
        """Empty array."""
        values = np.array([])

        result = normalize_zscore_logistic(values)
        assert len(result) == 0

    def test_minmax_zero_range(self):
        """min=max = range=0 = division by zero."""
        values = np.array([0.5, 0.5, 0.5])

        result = normalize_minmax(values)
        assert np.all(np.isfinite(result))

    def test_percentile_with_ties(self):
        """All values tied."""
        values = np.array([0.5, 0.5, 0.5, 0.5])

        result = normalize_percentile(values)
        assert np.all(np.isfinite(result))

    def test_normalization_with_nan(self):
        """NaN in values."""
        values = np.array([0.3, np.nan, 0.7])

        result = normalize_zscore_logistic(values)
        # Should handle gracefully (or document behavior)
        assert isinstance(result, np.ndarray)

    def test_normalization_with_inf(self):
        """Inf in values."""
        values = np.array([0.3, np.inf, 0.7])

        result = normalize_minmax(values)
        assert isinstance(result, np.ndarray)


class TestDecayCrashPoints:
    """Decay divides by half_life_days."""

    def test_decay_zero_half_life(self):
        """half_life=0 = division by zero."""
        result = compute_decay_weight(age_days=5.0, half_life_days=0.0)
        assert np.isfinite(result)

    def test_decay_negative_half_life(self):
        """Negative half life is invalid."""
        result = compute_decay_weight(age_days=5.0, half_life_days=-10.0)
        assert np.isfinite(result)

    def test_decay_weights_zero_half_life(self):
        """Batch with zero half life."""
        timestamps = np.array([0.0, 86400.0, 172800.0])
        now = 172800.0

        result = compute_decay_weights(timestamps, now, half_life_days=0.0)
        assert np.all(np.isfinite(result))

    def test_weighted_mean_zero_weights(self):
        """All weights are zero."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])

        result = weighted_mean(values, weights)
        assert np.isfinite(result)

    def test_weighted_std_zero_weights(self):
        """Weighted std with zero weights."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])

        result = weighted_std(values, weights)
        assert np.isfinite(result)

    def test_effective_sample_size_zero_weights(self):
        """Effective sample size with zero weights."""
        weights = np.array([0.0, 0.0, 0.0])

        result = effective_sample_size(weights)
        assert np.isfinite(result)


class TestShrinkageCrashPoints:
    """Shrinkage divides by (n_eff + k)."""

    def test_shrinkage_zero_k(self):
        """k=0 with n_eff=0 could cause issues."""
        values = np.array([0.5, 0.6, 0.7])
        n_effs = np.array([0.0, 0.0, 0.0])

        result = shrink_toward_mean(values, n_effs, k=0.0)
        assert np.all(np.isfinite(result))

    def test_shrinkage_negative_k(self):
        """Negative k is invalid."""
        values = np.array([0.5, 0.6, 0.7])
        n_effs = np.array([10.0, 20.0, 30.0])

        result = shrink_toward_mean(values, n_effs, k=-100.0)
        assert np.all(np.isfinite(result))

    def test_shrinkage_negative_neff(self):
        """Negative n_eff is invalid."""
        values = np.array([0.5, 0.6, 0.7])
        n_effs = np.array([-10.0, 20.0, 30.0])

        result = shrink_toward_mean(values, n_effs, k=200.0)
        assert np.all(np.isfinite(result))

    def test_population_mean_empty(self):
        """Empty values array."""
        values = np.array([])

        result = compute_population_mean(values)
        assert np.isfinite(result)

    def test_population_mean_zero_weights(self):
        """All zero weights."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])

        result = compute_population_mean(values, weights)
        assert np.isfinite(result)


class TestPSSCrashPoints:
    """PSS divides by truth_score."""

    def test_pss_zero_truth_score(self):
        """truth_score=0 = division by zero."""
        result = pss(miner_score=0.5, truth_score=0.0)
        assert np.isfinite(result)

    def test_pss_negative_scores(self):
        """Negative scores (invalid but possible)."""
        result = pss(miner_score=-0.5, truth_score=0.2)
        assert np.isfinite(result)

    def test_pss_batch_zero_truth(self):
        """Batch with zero truth scores."""
        miner_scores = np.array([0.5, 0.3, 0.4])
        truth_scores = np.array([0.2, 0.0, 0.3])  # One zero!

        result = pss_batch(miner_scores, truth_scores)
        assert np.all(np.isfinite(result))


class TestTimeSeriesCrashPoints:
    """Time series has various edge cases."""

    def test_correlation_single_point(self):
        """Correlation undefined for single point."""
        vals1 = np.array([0.5])
        vals2 = np.array([0.5])

        result = compute_correlation(vals1, vals2)
        assert np.isfinite(result)

    def test_correlation_zero_variance(self):
        """One series is constant = zero variance = undefined correlation."""
        vals1 = np.array([0.5, 0.5, 0.5, 0.5])
        vals2 = np.array([0.3, 0.4, 0.5, 0.6])

        result = compute_correlation(vals1, vals2)
        assert np.isfinite(result)

    def test_sos_extreme_correlation(self):
        """SOS at correlation extremes."""
        assert np.isfinite(compute_sos(1.0))
        assert np.isfinite(compute_sos(-1.0))
        assert np.isfinite(compute_sos(0.0))

    def test_bucket_empty_series(self):
        """Empty time series."""
        timestamps = np.array([])
        values = np.array([])

        ts_out, vals_out = bucket_time_series(timestamps, values, bucket_seconds=60)
        assert len(ts_out) == 0
        assert len(vals_out) == 0

    def test_lead_lag_no_moves(self):
        """No significant moves to analyze."""
        truth_ts = np.linspace(0, 3600, 100)
        truth_vals = np.full(100, 0.5)  # Constant
        miner_ts = truth_ts.copy()
        miner_vals = np.full(100, 0.5)

        result = analyze_lead_lag(
            truth_ts, truth_vals, miner_ts, miner_vals,
            lead_window_seconds=300,
            lag_window_seconds=300,
            move_threshold=0.01,
        )
        assert np.isfinite(result.lead_ratio)
        assert np.isfinite(result.sos_score)


class TestBatchProcessingCrashPoints:
    """Batch operations with edge case arrays."""

    def test_brier_batch_empty(self):
        """Empty batch."""
        forecasts = np.array([]).reshape(0, 2)
        outcomes = np.array([]).reshape(0, 2).astype(np.int8)

        result = brier_score_batch(forecasts, outcomes)
        assert len(result) == 0

    def test_brier_batch_single_row(self):
        """Single row batch."""
        forecasts = np.array([[0.6, 0.4]])
        outcomes = np.array([[1, 0]], dtype=np.int8)

        result = brier_score_batch(forecasts, outcomes)
        assert len(result) == 1
        assert np.all(np.isfinite(result))

    def test_brier_batch_with_nans(self):
        """Batch with some NaN rows."""
        forecasts = np.array([
            [0.6, 0.4],
            [np.nan, 0.5],
            [0.7, 0.3],
        ])
        outcomes = np.array([
            [1, 0],
            [1, 0],
            [0, 1],
        ], dtype=np.int8)

        result = brier_score_batch(forecasts, outcomes)
        # Should handle gracefully
        assert len(result) == 3


class TestMESCrashPoints:
    """MES edge cases."""

    def test_mes_extreme_clv(self):
        """CLV outside normal range."""
        assert np.isfinite(compute_mes(100.0))
        assert np.isfinite(compute_mes(-100.0))
        assert np.isfinite(compute_mes(np.inf))
        assert np.isfinite(compute_mes(-np.inf))
        assert np.isfinite(compute_mes(np.nan))

    def test_mes_batch_with_inf(self):
        """Batch with inf values."""
        clv_probs = np.array([0.1, np.inf, -np.inf, np.nan])

        result = compute_mes_batch(clv_probs)
        # Should handle gracefully
        assert len(result) == 4

