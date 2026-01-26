"""Adversarial and edge case tests for scoring system.

Tests for:
1. Manipulation attempts - miners gaming the scoring
2. Data corruption - NaN, Inf, extreme values
3. Numerical stability - precision issues
4. Statistical anomalies - degenerate distributions
5. Determinism verification
"""

import math
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone, timedelta
import hashlib

import numpy as np
import pytest

from sparket.validator.scoring.metrics.proper_scoring import (
    brier_score,
    brier_score_batch,
    log_loss,
    log_loss_batch,
    pss,
)
from sparket.validator.scoring.metrics.clv import (
    compute_clv,
    compute_clv_batch,
    compute_mes,
)
from sparket.validator.scoring.metrics.calibration import (
    compute_calibration,
    calibration_score,
)
from sparket.validator.scoring.metrics.sharpness import (
    compute_sharpness,
    compute_variance,
)
from sparket.validator.scoring.metrics.time_series import (
    compute_correlation,
    compute_sos,
    analyze_lead_lag,
    bucket_time_series,
)
from sparket.validator.scoring.aggregation.decay import (
    compute_decay_weight,
    compute_decay_weights,
    weighted_mean,
    weighted_std,
)
from sparket.validator.scoring.aggregation.normalization import (
    normalize_zscore_logistic,
    normalize_percentile,
    normalize_minmax,
)
from sparket.validator.scoring.aggregation.shrinkage import (
    shrink_toward_mean,
)
from sparket.validator.scoring.validation import SubmissionValidator
from sparket.validator.scoring.determinism import (
    to_decimal,
    round_decimal,
    safe_divide,
    safe_sqrt,
    safe_ln,
    clamp,
    compute_hash,
)
from sparket.validator.scoring.types import ValidationError


def make_binary_forecast(p: float) -> np.ndarray:
    """Create binary forecast array [p, 1-p]."""
    return np.array([p, 1.0 - p])


def make_binary_outcome(hit: int) -> np.ndarray:
    """Create binary outcome array [hit, 1-hit]."""
    return np.array([hit, 1 - hit], dtype=np.int8)


def make_batch_forecasts(probs: list) -> np.ndarray:
    """Create batch of binary forecasts."""
    return np.array([[p, 1.0 - p] for p in probs])


def make_batch_outcomes(hits: list) -> np.ndarray:
    """Create batch of binary outcomes."""
    return np.array([[h, 1 - h] for h in hits], dtype=np.int8)


class TestNaNAndInfHandling:
    """Tests for handling NaN and Inf values - common manipulation vectors."""

    def test_brier_score_nan_forecast(self):
        """Brier score should penalize NaN forecasts (not crash)."""
        forecast = make_binary_forecast(np.nan)
        outcome = make_binary_outcome(1)

        # NaN gets worst possible Brier score (2.0)
        result = brier_score(forecast, outcome)
        assert result == 2.0

    def test_brier_score_inf_forecast(self):
        """Brier score should penalize Inf forecasts (not crash)."""
        forecast = np.array([np.inf, 0.4])
        outcome = make_binary_outcome(1)

        # Inf gets worst possible Brier score (2.0)
        result = brier_score(forecast, outcome)
        assert result == 2.0

    def test_log_loss_nan_forecast(self):
        """Log loss should penalize NaN forecasts (not crash)."""
        forecast = make_binary_forecast(np.nan)
        outcome = make_binary_outcome(1)

        # NaN gets worst possible log-loss
        result = log_loss(forecast, outcome)
        assert result > 20  # Very high penalty

    def test_clv_with_normal_values(self):
        """CLV should work with normal values."""
        miner_prob = 0.6
        miner_odds = 1.0 / miner_prob
        truth_prob = 0.55
        truth_odds = 1.0 / truth_prob
        submitted_ts = 1000.0  # Mock timestamp
        event_start_ts = 2000.0  # Mock timestamp

        result = compute_clv(miner_odds, miner_prob, truth_odds, truth_prob, submitted_ts, event_start_ts)
        assert np.isfinite(result.clv_odds)
        assert np.isfinite(result.clv_prob)

    def test_normalization_with_nan(self):
        """Normalization should handle NaN values."""
        values = np.array([0.5, np.nan, 0.7, 0.3])
        result = normalize_zscore_logistic(values)
        # Should not crash; NaN may propagate but shouldn't break everything
        assert len(result) == 4

    def test_weighted_mean_nan_values(self):
        """Weighted mean should handle NaN values."""
        values = np.array([1.0, np.nan, 3.0])
        weights = np.array([1.0, 1.0, 1.0])

        result = weighted_mean(values, weights)
        # Result may be NaN but shouldn't crash
        assert isinstance(result, float)

    def test_correlation_with_nan(self):
        """Correlation should handle NaN values."""
        vals1 = np.array([1.0, 2.0, np.nan, 4.0])
        vals2 = np.array([1.1, 2.1, 3.1, 4.1])

        result = compute_correlation(vals1, vals2)
        # Should return 0 or NaN, not crash
        assert isinstance(result, float)


class TestExtremeValueHandling:
    """Tests for extreme values that could cause overflow/underflow."""

    def test_brier_score_extreme_probabilities(self):
        """Brier score with probabilities at machine epsilon."""
        tiny = np.finfo(np.float64).tiny
        forecast = np.array([tiny, 1.0 - tiny])
        outcome = make_binary_outcome(0)

        result = brier_score(forecast, outcome)
        assert np.isfinite(result)

    def test_log_loss_near_zero_probability(self):
        """Log loss with probability near zero (log singularity)."""
        # Very small probability for realized outcome
        forecast = np.array([1e-15, 1.0 - 1e-15])
        outcome = make_binary_outcome(1)  # First outcome realized

        result = log_loss(forecast, outcome)
        # Should be clamped/handled, not infinite
        assert np.isfinite(result)

    def test_log_loss_near_one_probability(self):
        """Log loss with probability near 1 for wrong outcome."""
        forecast = np.array([1.0 - 1e-15, 1e-15])
        outcome = make_binary_outcome(0)  # Second outcome realized

        result = log_loss(forecast, outcome)
        assert np.isfinite(result)

    def test_clv_extreme_odds(self):
        """CLV with extreme odds values."""
        # 1000:1 odds
        miner_prob = 0.001
        miner_odds = 1000.0
        truth_prob = 0.0015
        truth_odds = 666.67
        submitted_ts = 1000.0
        event_start_ts = 2000.0

        result = compute_clv(miner_odds, miner_prob, truth_odds, truth_prob, submitted_ts, event_start_ts)
        assert np.isfinite(result.clv_odds)
        assert np.isfinite(result.clv_prob)

    def test_decay_weight_extreme_age(self):
        """Decay weight with extreme ages."""
        # Very old submission
        weight_old = compute_decay_weight(365 * 10, 10.0)  # 10 years
        assert weight_old >= 0
        assert np.isfinite(weight_old)

        # Negative age (future submission - shouldn't happen but test it)
        weight_future = compute_decay_weight(-1.0, 10.0)
        assert weight_future >= 0  # Should clamp to 0 age

    def test_shrinkage_extreme_sample_size(self):
        """Shrinkage with extreme sample sizes."""
        values = np.array([0.5, 0.6, 0.7])

        # Huge sample size - should have almost no shrinkage
        n_eff_huge = np.array([1e10, 1e10, 1e10])
        shrunk_huge = shrink_toward_mean(values, n_eff_huge, k=200.0)
        np.testing.assert_array_almost_equal(shrunk_huge, values, decimal=5)

        # Zero sample size - should shrink entirely to mean
        n_eff_zero = np.array([0.0, 0.0, 0.0])
        shrunk_zero = shrink_toward_mean(values, n_eff_zero, k=200.0)
        pop_mean = values.mean()
        np.testing.assert_array_almost_equal(shrunk_zero, [pop_mean] * 3, decimal=5)


class TestManipulationAttempts:
    """Tests for potential miner manipulation strategies."""

    def test_always_predict_50_percent(self):
        """Miner always predicts 50% to avoid being wrong."""
        n = 100
        forecasts = make_batch_forecasts([0.5] * n)
        # 50% actually win
        outcomes = make_batch_outcomes([1] * 50 + [0] * 50)

        brier = brier_score_batch(forecasts, outcomes)
        # Brier should be 0.5 for 50% prediction on binary outcomes
        # (0.5 - 1)^2 + (0.5 - 0)^2 = 0.25 + 0.25 = 0.5
        assert np.allclose(brier.mean(), 0.5, atol=0.01)

        # Sharpness should be very low (no variance)
        probs = np.full(n, 0.5)
        sharp = compute_sharpness(probs, target_variance=0.04)
        assert sharp < 0.1  # Very low sharpness

    def test_extreme_predictions_manipulation(self):
        """Miner always predicts 99% or 1%."""
        # Half right, half wrong
        n = 100
        probs = [0.99] * 50 + [0.01] * 50
        forecasts = make_batch_forecasts(probs)
        outcomes = make_batch_outcomes([1] * 50 + [0] * 50)

        brier = brier_score_batch(forecasts, outcomes)
        # Should be excellent when right (~0.02 for correct extreme)
        assert brier[:50].mean() < 0.05

    def test_copying_ground_truth(self):
        """Miner copies ground truth exactly (should be caught by SOS)."""
        # Perfect correlation with truth
        truth_times = np.linspace(0, 3600, 100)
        truth_vals = 0.5 + 0.1 * np.sin(truth_times / 100)

        miner_times = truth_times.copy()
        miner_vals = truth_vals.copy()  # Exact copy

        # SOS should detect this
        result = analyze_lead_lag(
            truth_times, truth_vals,
            miner_times, miner_vals,
            lead_window_seconds=300,
            lag_window_seconds=300,
            move_threshold=0.01,
        )

        # SOS = 1 - |correlation| should be ~0 for perfect copy
        assert result.sos_score < 0.1

    def test_lagging_ground_truth(self):
        """Miner lags ground truth by fixed delay."""
        truth_times = np.linspace(0, 3600, 100)
        truth_vals = 0.5 + 0.1 * np.sin(truth_times / 100)

        # Miner lags by 5 minutes
        lag_seconds = 300
        miner_times = truth_times + lag_seconds
        miner_vals = truth_vals.copy()

        result = analyze_lead_lag(
            truth_times, truth_vals,
            miner_times, miner_vals,
            lead_window_seconds=300,
            lag_window_seconds=600,
            move_threshold=0.01,
        )

        # Lead ratio might be ~0.5 (neutral) since miner follows truth timing-wise
        assert isinstance(result.lead_ratio, float)

    def test_probability_sum_manipulation_overround(self):
        """Miner submits probabilities that sum > 1 (overround)."""
        validator = SubmissionValidator()

        # Overround (sum > 1) - typical of books
        # Default tolerance is 0.01, so 1.1 sum should fail
        probs = [0.55, 0.55]  # Sum = 1.1
        with pytest.raises(ValidationError):
            validator.validate_probability_vector(probs)

    def test_probability_sum_manipulation_underround(self):
        """Miner submits probabilities that sum < 1 (underround)."""
        validator = SubmissionValidator()

        # Underround (sum < 1)
        probs = [0.45, 0.45]  # Sum = 0.9
        with pytest.raises(ValidationError):
            validator.validate_probability_vector(probs)


class TestDegenerateDistributions:
    """Tests for degenerate statistical cases."""

    def test_single_miner(self):
        """Scoring with only one miner."""
        values = np.array([0.5])
        n_effs = np.array([100.0])

        # Normalization of single value
        normalized = normalize_zscore_logistic(values)
        assert np.isfinite(normalized[0])
        assert 0 <= normalized[0] <= 1

        # Shrinkage of single value
        shrunk = shrink_toward_mean(values, n_effs, k=200.0)
        assert np.isfinite(shrunk[0])

    def test_all_identical_values(self):
        """All miners have identical scores."""
        values = np.full(100, 0.5)

        # Z-score normalization with zero variance
        normalized = normalize_zscore_logistic(values)
        # All should be 0.5 (mapped from z=0)
        assert np.allclose(normalized, 0.5, atol=0.01)

        # Percentile normalization
        percentile_norm = normalize_percentile(values)
        # All should be around 0.5
        assert np.all(np.isfinite(percentile_norm))

    def test_bimodal_distribution(self):
        """Miners cluster into two distinct groups."""
        values = np.array([0.1] * 50 + [0.9] * 50)

        normalized = normalize_zscore_logistic(values)
        # Should separate clearly
        assert normalized[:50].mean() < 0.3
        assert normalized[50:].mean() > 0.7

    def test_single_outlier(self):
        """One miner is a massive outlier."""
        values = np.array([0.5] * 99 + [100.0])

        # Normalization should handle this
        normalized = normalize_zscore_logistic(values)
        assert np.all(np.isfinite(normalized))
        assert np.all(normalized >= 0) and np.all(normalized <= 1)

        # Min-max would be dominated by outlier
        minmax = normalize_minmax(values)
        assert minmax[-1] == 1.0  # Outlier gets max
        assert minmax[0] < 0.01  # Others get crushed

    def test_calibration_insufficient_bins(self):
        """Calibration with not enough data per bin."""
        probs = np.array([0.1, 0.9])  # Only 2 points
        outcomes = np.array([0, 1], dtype=np.int8)

        result = compute_calibration(probs, outcomes, num_bins=20, min_samples=30)
        # Should return default score, not crash
        assert result.score == 0.5  # Default for insufficient data


class TestDeterminismVerification:
    """Verify that scoring is deterministic."""

    def test_brier_score_deterministic(self):
        """Same inputs should always produce same Brier score."""
        forecast = make_binary_forecast(0.6)
        outcome = make_binary_outcome(1)

        results = [brier_score(forecast, outcome) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_normalization_deterministic(self):
        """Normalization should be deterministic."""
        values = np.array([0.3, 0.5, 0.7, 0.2, 0.8])

        results = [normalize_zscore_logistic(values.copy()) for _ in range(100)]
        assert all(np.array_equal(results[0], r) for r in results)

    def test_shrinkage_deterministic(self):
        """Shrinkage should be deterministic."""
        values = np.array([0.3, 0.5, 0.7])
        n_effs = np.array([10.0, 50.0, 100.0])

        results = [shrink_toward_mean(values.copy(), n_effs.copy(), k=200.0) for _ in range(100)]
        assert all(np.array_equal(results[0], r) for r in results)

    def test_hash_deterministic(self):
        """Hashing should be deterministic."""
        data = {"miner_id": 1, "score": 0.75, "as_of": "2024-01-01T00:00:00Z"}

        hashes = [compute_hash(data) for _ in range(100)]
        assert all(h == hashes[0] for h in hashes)

    def test_decimal_operations_deterministic(self):
        """Decimal operations should be deterministic."""
        a = to_decimal(0.1, "a")
        b = to_decimal(0.2, "b")

        results = [round_decimal(a + b, 6) for _ in range(100)]
        assert all(r == results[0] for r in results)


class TestNumericalStability:
    """Tests for numerical stability issues."""

    def test_safe_divide_by_zero(self):
        """Safe divide should handle division by zero."""
        result = safe_divide(Decimal("1"), Decimal("0"), Decimal("-1"))
        assert result == Decimal("-1")  # Default value

    def test_safe_sqrt_negative(self):
        """Safe sqrt should handle negative numbers."""
        result = safe_sqrt(Decimal("-1"))
        assert result == Decimal("0")  # Clamped to 0

    def test_safe_ln_zero(self):
        """Safe ln should handle zero and negative."""
        result_zero = safe_ln(Decimal("0"))
        assert result_zero is not None  # Should return something sensible

        result_neg = safe_ln(Decimal("-1"))
        assert result_neg is not None

    def test_clamp_bounds(self):
        """Clamp should enforce bounds correctly."""
        assert clamp(Decimal("1.5"), Decimal("0"), Decimal("1")) == Decimal("1")
        assert clamp(Decimal("-0.5"), Decimal("0"), Decimal("1")) == Decimal("0")
        assert clamp(Decimal("0.5"), Decimal("0"), Decimal("1")) == Decimal("0.5")

    def test_weighted_std_zero_weights(self):
        """Weighted std with all zero weights."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])

        result = weighted_std(values, weights)
        assert result == 0.0  # Should return 0, not error

    def test_correlation_constant_series(self):
        """Correlation of constant series (zero std)."""
        vals1 = np.array([0.5, 0.5, 0.5, 0.5])
        vals2 = np.array([0.6, 0.6, 0.6, 0.6])

        result = compute_correlation(vals1, vals2)
        # Should be 0 (undefined correlation), not NaN crash
        assert result == 0.0

    def test_decay_weight_very_small_half_life(self):
        """Decay weight with very small half-life."""
        weight = compute_decay_weight(1.0, 0.001)  # Very small, not zero
        assert np.isfinite(weight)


class TestValidationBoundaries:
    """Tests for validation at boundaries."""

    def test_probability_exactly_zero_rejected(self):
        """Probability of exactly 0 should be rejected."""
        validator = SubmissionValidator()
        with pytest.raises(ValidationError):
            validator.validate_probability(0.0)

    def test_probability_exactly_one_rejected(self):
        """Probability of exactly 1 should be rejected."""
        validator = SubmissionValidator()
        with pytest.raises(ValidationError):
            validator.validate_probability(1.0)

    def test_probability_valid_range(self):
        """Probabilities in valid range should pass."""
        validator = SubmissionValidator()
        # These should not raise
        validator.validate_probability(0.001)
        validator.validate_probability(0.5)
        validator.validate_probability(0.999)

    def test_odds_too_low_rejected(self):
        """Odds below minimum should be rejected."""
        validator = SubmissionValidator()
        with pytest.raises(ValidationError):
            validator.validate_odds(1.001)  # Below 1.01 min

    def test_odds_valid_range(self):
        """Odds in valid range should pass."""
        validator = SubmissionValidator()
        validator.validate_odds(1.5)
        validator.validate_odds(10.0)
        validator.validate_odds(100.0)


class TestLongTermPatterns:
    """Tests for patterns that might emerge over time."""

    def test_score_drift_over_samples(self):
        """Scores shouldn't drift as sample size grows."""
        np.random.seed(42)

        scores = []
        for n in [10, 100, 1000]:
            # Generate samples
            probs = np.random.uniform(0.3, 0.7, n)
            hits = (np.random.random(n) < probs).astype(int)

            forecasts = make_batch_forecasts(probs.tolist())
            outcomes = make_batch_outcomes(hits.tolist())

            brier = brier_score_batch(forecasts, outcomes).mean()
            scores.append(brier)

        # Scores should be stable (around 0.4-0.5 for random guessing)
        assert all(0.2 < s < 0.8 for s in scores)

    def test_normalization_stability_over_miners(self):
        """Adding miners shouldn't drastically change existing scores."""
        np.random.seed(42)

        # Initial miners
        initial = np.random.uniform(0.3, 0.7, 100)
        initial_normalized = normalize_zscore_logistic(initial)

        # Add more miners
        extended = np.concatenate([initial, np.random.uniform(0.3, 0.7, 50)])
        extended_normalized = normalize_zscore_logistic(extended)

        # Original miners' scores shouldn't change drastically
        diff = np.abs(initial_normalized - extended_normalized[:100])
        assert np.mean(diff) < 0.15  # Average change < 15%

    def test_calibration_convergence(self):
        """Calibration should converge with more samples."""
        np.random.seed(42)

        cal_scores = []
        for n in [50, 100, 500, 1000]:
            # Generate well-calibrated predictions
            probs = np.random.uniform(0.1, 0.9, n)
            outcomes = (np.random.random(n) < probs).astype(np.int8)

            result = compute_calibration(probs, outcomes, num_bins=10, min_samples=30)
            cal_scores.append(result.score)

        # Scores should be improving or stable
        if len(cal_scores) >= 2 and cal_scores[-1] != 0.5:
            assert cal_scores[-1] >= 0.4  # Should be reasonably calibrated


class TestEdgeCasesForScoringJobs:
    """Edge cases that could break the scoring jobs."""

    def test_empty_arrays(self):
        """Empty arrays should be handled gracefully."""
        empty = np.array([])

        # Normalization of empty
        norm = normalize_zscore_logistic(empty)
        assert len(norm) == 0

        # Shrinkage of empty
        shrunk = shrink_toward_mean(empty, empty, k=200.0)
        assert len(shrunk) == 0

    def test_single_element_arrays(self):
        """Single element arrays should work."""
        single = np.array([0.5])

        norm = normalize_zscore_logistic(single)
        assert len(norm) == 1
        assert 0 <= norm[0] <= 1

    def test_mixed_positive_negative(self):
        """Values can be positive and negative."""
        values = np.array([-0.5, 0.0, 0.5, 1.0, -1.0])

        norm = normalize_zscore_logistic(values)
        assert np.all(np.isfinite(norm))
        assert np.all((norm >= 0) & (norm <= 1))

    def test_very_large_batch(self):
        """Large batches should work without memory issues."""
        n = 10000
        values = np.random.uniform(0, 1, n)

        norm = normalize_zscore_logistic(values)
        assert len(norm) == n
        assert np.all(np.isfinite(norm))


class TestCLVEdgeCases:
    """Edge cases for CLV calculations."""

    def test_clv_miner_equals_truth(self):
        """CLV when miner exactly equals truth."""
        submitted_ts = 1000.0
        event_start_ts = 2000.0
        result = compute_clv(2.0, 0.5, 2.0, 0.5, submitted_ts, event_start_ts)
        assert result.clv_odds == 0.0
        assert result.clv_prob == 0.0

    def test_clv_miner_better_than_truth(self):
        """CLV when miner is more accurate than truth."""
        # Miner says 60%, truth says 55%
        miner_prob = 0.6
        truth_prob = 0.55
        submitted_ts = 1000.0
        event_start_ts = 2000.0
        result = compute_clv(
            1.0 / miner_prob, miner_prob,
            1.0 / truth_prob, truth_prob,
            submitted_ts, event_start_ts,
        )
        # CLV prob measures how much miner differs from truth
        assert isinstance(result.clv_prob, float)

    def test_mes_boundary(self):
        """MES at boundaries."""
        # Perfect prediction (clv_prob = 0)
        mes_perfect = compute_mes(0.0)
        assert mes_perfect == 1.0

        # Worst prediction (clv_prob = 1)
        mes_worst = compute_mes(1.0)
        assert mes_worst == 0.0

        # Typical prediction
        mes_typical = compute_mes(0.05)
        assert 0.9 < mes_typical < 1.0
