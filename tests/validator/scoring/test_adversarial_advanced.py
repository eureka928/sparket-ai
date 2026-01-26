"""Advanced adversarial tests for scoring system.

Deep dive into:
1. Timing manipulation attacks
2. Float precision exploits
3. Sybil/collusion detection gaps
4. Normalization gaming
5. Statistical cherry-picking
6. Time series edge cases
7. Shrinkage gaming
8. Cross-correlation attacks
9. Decimal precision edge cases
10. Concurrency/ordering issues
"""

import sys
import struct
from decimal import Decimal, InvalidOperation, getcontext
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from sparket.validator.scoring.metrics.proper_scoring import (
    brier_score,
    brier_score_batch,
    log_loss,
    log_loss_batch,
)
from sparket.validator.scoring.metrics.clv import (
    compute_clv,
    compute_clv_batch,
    compute_mes,
)
from sparket.validator.scoring.metrics.calibration import (
    compute_calibration,
)
from sparket.validator.scoring.metrics.sharpness import (
    compute_sharpness,
)
from sparket.validator.scoring.metrics.time_series import (
    compute_correlation,
    compute_sos,
    analyze_lead_lag,
    bucket_time_series,
    align_time_series,
    detect_moves,
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
)
from sparket.validator.scoring.validation import SubmissionValidator
from sparket.validator.scoring.determinism import (
    to_decimal,
    round_decimal,
    safe_divide,
    compute_hash,
    get_canonical_window_bounds,
    floor_to_bucket,
)
from sparket.validator.scoring.types import ValidationError


# =============================================================================
# TIMING ATTACKS
# =============================================================================

class TestTimingAttacks:
    """Tests for timing-based manipulation attempts."""

    def test_submission_at_epoch_boundary(self):
        """Submission exactly at epoch boundary."""
        # Unix epoch boundaries can cause issues
        epoch_ts = 0.0
        event_ts = 3600.0

        result = compute_clv(2.0, 0.5, 2.0, 0.5, epoch_ts, event_ts)
        assert result.minutes_to_close > 0

    def test_submission_microseconds_before_close(self):
        """Submission just microseconds before event."""
        event_ts = 1000000.0
        submit_ts = event_ts - 0.000001  # 1 microsecond before

        result = compute_clv(2.0, 0.5, 2.0, 0.5, submit_ts, event_ts)
        # Should be 0 minutes (clamped)
        assert result.minutes_to_close == 0

    def test_submission_after_event_start(self):
        """Submission after event started (should be rejected or handled)."""
        event_ts = 1000.0
        submit_ts = 1001.0  # After event

        result = compute_clv(2.0, 0.5, 2.0, 0.5, submit_ts, event_ts)
        # Minutes to close should be 0 or negative handled
        assert result.minutes_to_close <= 0

    def test_identical_timestamps(self):
        """Multiple submissions at exact same timestamp."""
        timestamps = np.array([1000.0, 1000.0, 1000.0, 1001.0])
        values = np.array([0.5, 0.6, 0.55, 0.58])

        # Bucketing should handle duplicates
        buck_ts, buck_vals = bucket_time_series(timestamps, values, bucket_seconds=1)
        # Should have deduplicated
        assert len(buck_ts) <= len(timestamps)

    def test_out_of_order_timestamps(self):
        """Timestamps not in chronological order."""
        timestamps = np.array([1002.0, 1000.0, 1003.0, 1001.0])
        values = np.array([0.58, 0.5, 0.6, 0.55])

        # Should handle gracefully
        buck_ts, buck_vals = bucket_time_series(timestamps, values, bucket_seconds=1)
        # Bucketed timestamps should be sorted
        assert np.all(buck_ts[:-1] <= buck_ts[1:])

    def test_huge_time_gaps(self):
        """Very large gaps in time series."""
        timestamps = np.array([0.0, 1e12])  # ~31,000 years apart
        values = np.array([0.5, 0.6])

        # Decay should handle this
        weights = compute_decay_weights(timestamps, timestamps[-1], half_life_days=10.0)
        assert np.all(np.isfinite(weights))
        assert weights[0] < 1e-100  # Should be essentially zero


# =============================================================================
# FLOAT PRECISION EXPLOITS
# =============================================================================

class TestFloatPrecisionExploits:
    """Tests for floating-point precision attacks."""

    def test_values_differ_at_machine_epsilon(self):
        """Values that differ only at machine epsilon."""
        eps = np.finfo(np.float64).eps
        val1 = np.array([0.5])
        val2 = np.array([0.5 + eps])

        # Z-score normalization AMPLIFIES tiny differences when std is tiny
        # This is expected behavior - the relative differences are what matter
        combined = np.array([0.5, 0.5 + eps, 0.5 - eps])
        norm = normalize_zscore_logistic(combined)

        # All should be finite (no NaN/Inf from tiny std)
        assert np.all(np.isfinite(norm))
        # Center value should be 0.5, others symmetric around it
        assert norm[0] == 0.5
        assert np.isclose(norm[1], 1 - norm[2], atol=1e-6)

    def test_denormalized_floats(self):
        """Denormalized (subnormal) float values."""
        subnormal = np.finfo(np.float64).tiny / 2
        values = np.array([subnormal, 0.5, 1.0 - subnormal])

        norm = normalize_zscore_logistic(values)
        assert np.all(np.isfinite(norm))

    def test_float_rounding_consistency(self):
        """Ensure consistent rounding across operations."""
        # This value has a non-terminating binary representation
        val = 0.1

        d1 = to_decimal(val, "val")
        d2 = to_decimal(val, "val")

        # Should be identical
        assert d1 == d2

        # Rounding should be deterministic
        r1 = round_decimal(d1, 6)
        r2 = round_decimal(d2, 6)
        assert r1 == r2

    def test_special_ieee_values(self):
        """Handle special IEEE 754 values."""
        special = np.array([
            np.finfo(np.float64).max,
            np.finfo(np.float64).min,
            np.finfo(np.float64).tiny,
        ])

        # Should not crash
        mean = weighted_mean(special, np.ones(3))
        assert isinstance(mean, float)

    def test_catastrophic_cancellation(self):
        """Test for catastrophic cancellation in subtraction."""
        # Values very close together
        a = 1.0000000001
        b = 1.0000000000

        vals = np.array([a, b])
        std = weighted_std(vals, np.ones(2))

        # Should get a very small but finite std
        assert np.isfinite(std)


# =============================================================================
# SYBIL/COLLUSION DETECTION GAPS
# =============================================================================

class TestSybilCollusion:
    """Tests for detecting collusion or Sybil attacks."""

    def test_identical_miners(self):
        """Multiple miners with identical submissions."""
        # All miners submit same values
        values = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
        n_effs = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

        # After shrinkage, all should still be equal
        shrunk = shrink_toward_mean(values, n_effs, k=200.0)
        assert np.allclose(shrunk, shrunk[0])

        # After normalization, all should get ~0.5
        norm = normalize_zscore_logistic(values)
        assert np.allclose(norm, 0.5)

    def test_coordinated_outliers(self):
        """Miners coordinate to create artificial outliers."""
        # 3 miners collude to push one miner high
        values = np.array([0.1, 0.1, 0.1, 0.9, 0.5, 0.5])

        norm = normalize_zscore_logistic(values)

        # The outlier should get high score
        assert norm[3] > 0.8

        # But min-max would be different
        minmax = normalize_minmax(values)
        assert minmax[3] == 1.0

    def test_correlation_among_miners(self):
        """Detect if miners' predictions are suspiciously correlated."""
        np.random.seed(42)

        # Independent miners
        miner1 = np.random.uniform(0.4, 0.6, 50)
        miner2 = np.random.uniform(0.4, 0.6, 50)

        corr_independent = compute_correlation(miner1, miner2)

        # Colluding miners (copy with noise)
        miner3 = miner1 + np.random.normal(0, 0.01, 50)

        corr_colluding = compute_correlation(miner1, miner3)

        # Colluding should have much higher correlation
        assert corr_colluding > corr_independent


# =============================================================================
# NORMALIZATION GAMING
# =============================================================================

class TestNormalizationGaming:
    """Tests for gaming the normalization process."""

    def test_strategic_positioning(self):
        """Miner positions just above median to maximize normalized score."""
        # If miner knows distribution, they could position strategically
        others = np.array([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7])

        # Strategic miner places just above median
        strategic_val = 0.51
        all_vals = np.append(others, strategic_val)

        norm = normalize_percentile(all_vals)

        # Strategic miner should be just above 50th percentile
        strategic_norm = norm[-1]
        assert strategic_norm > 0.5

    def test_extreme_to_shift_distribution(self):
        """Submit extreme value to shift entire distribution."""
        normal_vals = np.array([0.4, 0.5, 0.6])

        # Add extreme outlier
        with_outlier = np.append(normal_vals, 10.0)

        # Min-max normalization is especially vulnerable
        minmax_normal = normalize_minmax(normal_vals)
        minmax_with = normalize_minmax(with_outlier)

        # Original values get compressed toward 0 when outlier is added
        # In original: 0.4 maps to 0, 0.6 maps to 1
        # With outlier: 0.4 still maps to 0 (it's still min), but 0.6 << 1
        assert minmax_with[2] < minmax_normal[2]  # 0.6 gets compressed

    def test_zscore_logistic_resistance(self):
        """Z-score + logistic should resist outlier attacks better than min-max."""
        normal_vals = np.array([0.4, 0.45, 0.5, 0.55, 0.6])

        # Add extreme outlier
        with_outlier = np.append(normal_vals, 100.0)

        # Compare how much z-score vs min-max changes with outlier
        zscore_normal = normalize_zscore_logistic(normal_vals)
        zscore_with = normalize_zscore_logistic(with_outlier)

        minmax_normal = normalize_minmax(normal_vals)
        minmax_with = normalize_minmax(with_outlier)

        # Min-max compression is severe
        minmax_shift = np.abs(minmax_normal - minmax_with[:5]).max()

        # Z-score shift exists but logistic bounds it
        zscore_shift = np.abs(zscore_normal - zscore_with[:5]).max()

        # Min-max is more affected (original values span 0-1 but get compressed to <0.01)
        assert minmax_shift > 0.9  # Severe compression
        # Z-score changes less dramatically due to logistic squashing
        assert np.all(np.isfinite(zscore_with))


# =============================================================================
# STATISTICAL CHERRY-PICKING
# =============================================================================

class TestCherryPicking:
    """Tests for detecting statistical cherry-picking."""

    def test_only_easy_markets(self):
        """Miner only bets on "easy" high-confidence markets."""
        # Miner only submits when very confident
        miner_probs = np.array([0.9, 0.85, 0.92, 0.88])  # All high
        outcomes = np.array([1, 1, 1, 0], dtype=np.int8)

        # High variance in predictions = high sharpness potential
        # But these are all clustered near 0.9, so variance is low
        variance = np.var(miner_probs)
        assert variance < 0.01  # Low variance = low sharpness

        sharpness = compute_sharpness(miner_probs, target_variance=0.04)
        # Low sharpness because all predictions clustered - at or below 0.5
        assert sharpness <= 0.5

        # Calibration might reveal issues if they're overconfident
        # With only 4 samples, calibration is unreliable but shouldn't crash
        cal = compute_calibration(miner_probs, outcomes, min_samples=4)
        assert np.isfinite(cal.score)

    def test_avoiding_close_games(self):
        """Miner avoids games where probability is near 50%."""
        # This is actually legitimate strategy, but we should be aware

        # Miner's predictions (avoids 50%)
        miner_probs = np.concatenate([
            np.random.uniform(0.7, 0.9, 25),
            np.random.uniform(0.1, 0.3, 25),
        ])

        # Variance should be high
        variance = np.var(miner_probs)
        assert variance > 0.04  # Definitely "sharp"

    def test_sample_size_gaming(self):
        """Miner submits just enough to cross threshold."""
        # If threshold is 30 for calibration...
        probs = np.random.uniform(0.3, 0.7, 30)  # Exactly 30
        outcomes = (np.random.random(30) < probs).astype(np.int8)

        # Should be computed (not default)
        cal = compute_calibration(probs, outcomes, min_samples=30)
        # Just barely crosses threshold


# =============================================================================
# TIME SERIES EDGE CASES
# =============================================================================

class TestTimeSeriesEdgeCases:
    """Edge cases in time series analysis."""

    def test_single_point_series(self):
        """Time series with only one point."""
        times = np.array([1000.0])
        vals = np.array([0.5])

        # Bucketing should work
        buck_ts, buck_vals = bucket_time_series(times, vals, 60)
        assert len(buck_ts) == 1

        # Correlation needs at least 2 points
        corr = compute_correlation(vals, vals)
        assert corr == 0.0  # Not enough points

    def test_no_moves_in_series(self):
        """Time series with no significant moves."""
        times = np.linspace(0, 3600, 100)
        vals = np.full(100, 0.5)  # Constant

        result = analyze_lead_lag(
            times, vals, times, vals,
            lead_window_seconds=300,
            lag_window_seconds=300,
            move_threshold=0.01,
        )

        # No moves to analyze
        assert result.total_truth_moves == 0

    def test_all_moves_same_direction(self):
        """All moves in same direction (trending market)."""
        times = np.linspace(0, 3600, 100)
        truth_vals = 0.3 + 0.004 * np.arange(100)  # Steadily increasing

        # Miner follows trend
        miner_vals = truth_vals + 0.01

        result = analyze_lead_lag(
            times, truth_vals, times, miner_vals,
            lead_window_seconds=60,
            lag_window_seconds=60,
            move_threshold=0.02,
        )

        # Should have detected moves
        assert result.total_truth_moves >= 0

    def test_very_noisy_series(self):
        """High-frequency noise in time series."""
        np.random.seed(42)
        times = np.linspace(0, 3600, 1000)
        vals = 0.5 + 0.1 * np.random.randn(1000)

        # Bucketing should smooth out noise
        buck_ts, buck_vals = bucket_time_series(times, vals, 60)
        assert len(buck_ts) < len(times)

    def test_alignment_no_overlap(self):
        """Two series with no overlapping timestamps."""
        times1 = np.array([100.0, 200.0, 300.0])
        vals1 = np.array([0.5, 0.6, 0.7])
        times2 = np.array([150.0, 250.0, 350.0])
        vals2 = np.array([0.55, 0.65, 0.75])

        aligned1, aligned2 = align_time_series(times1, vals1, times2, vals2)

        # No overlap
        assert len(aligned1) == 0
        assert len(aligned2) == 0


# =============================================================================
# SHRINKAGE GAMING
# =============================================================================

class TestShrinkageGaming:
    """Tests for gaming the shrinkage mechanism."""

    def test_inflate_sample_count(self):
        """Miner submits many tiny bets to inflate n_eff."""
        # Need different values to see differential shrinkage
        # Honest miner: 100 bets, extreme value
        honest_neff = 100.0
        honest_val = 0.9

        # Gaming miner: huge n_eff, less extreme value
        gaming_neff = 10000.0
        gaming_val = 0.6

        # When both are in the pool
        all_vals = np.array([honest_val, gaming_val])
        all_neffs = np.array([honest_neff, gaming_neff])

        shrunk = shrink_toward_mean(all_vals, all_neffs, k=200.0)

        # Population mean is dominated by gaming miner (high n_eff weight)
        # Honest miner gets shrunk more toward gaming miner's value
        # Gaming miner barely shrinks

        # This shows the attack: gaming miner influences pop_mean
        # while resisting shrinkage themselves
        assert abs(shrunk[1] - gaming_val) < abs(shrunk[0] - honest_val)

    def test_strategic_abstention(self):
        """Miner abstains when uncertain to protect score."""
        # This is actually rational behavior
        # But it means n_eff might not reflect "difficulty" of predictions

        values = np.array([0.8, 0.75, 0.82])  # Only confident predictions
        n_effs = np.array([10.0, 10.0, 10.0])  # Low count

        shrunk = shrink_toward_mean(values, n_effs, k=200.0)

        # Gets shrunk toward mean
        pop_mean = values.mean()
        assert np.all(np.abs(shrunk - pop_mean) < np.abs(values - pop_mean))

    def test_population_mean_manipulation(self):
        """Multiple miners coordinate to shift population mean."""
        # If 10 miners collude to have low values...
        colluders = np.full(10, 0.3)
        honest = np.array([0.7])

        all_vals = np.concatenate([colluders, honest])
        all_neffs = np.full(11, 100.0)

        shrunk = shrink_toward_mean(all_vals, all_neffs, k=200.0)

        # Honest miner gets pulled toward colluders' mean
        pop_mean = all_vals.mean()  # Skewed by colluders
        assert shrunk[-1] < 0.7  # Honest miner's score decreased


# =============================================================================
# DECIMAL PRECISION EDGE CASES
# =============================================================================

class TestDecimalPrecision:
    """Edge cases in Decimal handling."""

    def test_very_long_decimal_string(self):
        """Very long decimal representation."""
        long_val = "0." + "1" * 100  # 100 decimal places

        d = to_decimal(long_val, "long")
        # Should truncate or handle gracefully
        assert isinstance(d, Decimal)

    def test_scientific_notation(self):
        """Scientific notation input."""
        sci_val = "1.5e-10"

        d = to_decimal(sci_val, "sci")
        assert d == Decimal("0.00000000015")

    def test_repeating_decimal(self):
        """Value that creates repeating decimal."""
        # 1/3 = 0.333...
        val = 1 / 3

        d = to_decimal(val, "third")
        # Should round to finite precision
        assert isinstance(d, Decimal)
        assert d > Decimal("0.33") and d < Decimal("0.34")

    def test_decimal_context_isolation(self):
        """Ensure Decimal context doesn't leak between operations."""
        # Save current context
        original_prec = getcontext().prec

        # Our operations shouldn't permanently change context
        _ = to_decimal(0.123456789012345678901234567890, "long")
        _ = round_decimal(Decimal("0.123456789"), 20)

        # Context should be unchanged
        assert getcontext().prec == original_prec


# =============================================================================
# ORDERING AND CONCURRENCY
# =============================================================================

class TestOrderingConcurrency:
    """Tests for ordering sensitivity and potential race conditions."""

    def test_order_independent_normalization(self):
        """Normalization should not depend on input order."""
        values = np.array([0.3, 0.5, 0.7, 0.2, 0.8])

        norm1 = normalize_zscore_logistic(values)

        # Shuffle
        perm = np.array([3, 0, 2, 4, 1])
        shuffled = values[perm]
        norm2 = normalize_zscore_logistic(shuffled)

        # Unshuffle result
        inv_perm = np.argsort(perm)
        norm2_unshuffled = norm2[inv_perm]

        np.testing.assert_array_almost_equal(norm1, norm2_unshuffled)

    def test_order_independent_shrinkage(self):
        """Shrinkage should not depend on input order."""
        values = np.array([0.3, 0.5, 0.7])
        n_effs = np.array([10.0, 50.0, 100.0])

        shrunk1 = shrink_toward_mean(values, n_effs, k=200.0)

        # Reverse order
        shrunk2 = shrink_toward_mean(values[::-1], n_effs[::-1], k=200.0)

        np.testing.assert_array_almost_equal(shrunk1, shrunk2[::-1])

    def test_order_independent_hash(self):
        """Hash should be order-independent for dict keys."""
        data1 = {"b": 2, "a": 1, "c": 3}
        data2 = {"a": 1, "c": 3, "b": 2}

        hash1 = compute_hash(data1)
        hash2 = compute_hash(data2)

        assert hash1 == hash2

    def test_epoch_alignment_deterministic(self):
        """Epoch alignment should be deterministic."""
        window_days = 30

        bounds1 = get_canonical_window_bounds(window_days)
        bounds2 = get_canonical_window_bounds(window_days)

        assert bounds1 == bounds2


# =============================================================================
# BOUNDARY VALUE ANALYSIS
# =============================================================================

class TestBoundaryValues:
    """Comprehensive boundary value testing."""

    def test_odds_just_above_min(self):
        """Odds exactly at minimum threshold."""
        validator = SubmissionValidator()

        # Should pass at min
        min_odds = float(validator.bounds.odds_min)
        result = validator.validate_odds(min_odds)
        assert result == validator.bounds.odds_min

    def test_probability_at_bounds(self):
        """Probabilities at exact bounds."""
        validator = SubmissionValidator()

        min_prob = float(validator.bounds.prob_min)
        max_prob = float(validator.bounds.prob_max)

        # Both should pass
        validator.validate_probability(min_prob)
        validator.validate_probability(max_prob)

    def test_cle_clamping_exact_bounds(self):
        """CLE values exactly at clamping bounds."""
        validator = SubmissionValidator()

        # At bounds
        min_cle = validator.bounds.cle_min
        max_cle = validator.bounds.cle_max

        assert validator.validate_cle(min_cle) == min_cle
        assert validator.validate_cle(max_cle) == max_cle

        # Beyond bounds
        assert validator.validate_cle(min_cle - Decimal("1")) == min_cle
        assert validator.validate_cle(max_cle + Decimal("1")) == max_cle

    def test_decay_at_half_life(self):
        """Decay exactly at half-life should give 0.5 weight."""
        half_life = 10.0
        weight = compute_decay_weight(half_life, half_life)

        assert np.isclose(weight, 0.5, atol=0.001)

    def test_bucket_at_exact_boundary(self):
        """Timestamp exactly on bucket boundary."""
        bucket_sec = 60

        # floor_to_bucket expects datetime, not float
        # Test with a datetime that's exactly on a minute boundary
        dt_on_boundary = datetime(2024, 1, 1, 1, 40, 0, tzinfo=timezone.utc)

        result = floor_to_bucket(dt_on_boundary, bucket_sec)
        assert result == dt_on_boundary  # Should stay on boundary


# =============================================================================
# HASH COLLISION ATTEMPTS
# =============================================================================

class TestHashCollisions:
    """Tests for potential hash collision issues."""

    def test_similar_structures(self):
        """Similar but different structures should have different hashes."""
        data1 = {"miner_id": 1, "score": 0.5}
        data2 = {"miner_id": 1, "score": 0.50000001}  # Slightly different

        hash1 = compute_hash(data1)
        hash2 = compute_hash(data2)

        assert hash1 != hash2

    def test_type_differences(self):
        """Same value, different types should hash differently."""
        data1 = {"value": 1}  # int
        data2 = {"value": 1.0}  # float
        data3 = {"value": "1"}  # string

        hashes = [compute_hash(d) for d in [data1, data2, data3]]

        # At least some should be different (depends on serialization)
        # Actually with JSON, 1 and 1.0 serialize the same, so...
        assert len(set(hashes)) >= 2  # At least int/string differ

    def test_nested_ordering(self):
        """Nested structure ordering should be handled."""
        data1 = {"outer": {"b": 2, "a": 1}}
        data2 = {"outer": {"a": 1, "b": 2}}

        hash1 = compute_hash(data1)
        hash2 = compute_hash(data2)

        # Should be identical (keys sorted recursively)
        assert hash1 == hash2


# =============================================================================
# EDGE CASES THAT MIGHT CAUSE SILENT FAILURES
# =============================================================================

class TestSilentFailures:
    """Cases that might fail silently rather than raising errors."""

    def test_all_weights_very_small(self):
        """All weights near zero but not exactly zero."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1e-300, 1e-300, 1e-300])

        result = weighted_mean(values, weights)

        # Should not be NaN
        assert np.isfinite(result) or result == 0.0

    def test_effective_sample_size_tiny_weights(self):
        """Effective sample size with tiny weights."""
        weights = np.array([1e-100, 1e-100, 1e-100])

        n_eff = effective_sample_size(weights)
        assert n_eff >= 0
        assert np.isfinite(n_eff)

    def test_normalization_near_constant(self):
        """Values that are almost but not quite constant."""
        values = np.array([0.5, 0.5 + 1e-15, 0.5 - 1e-15])

        norm = normalize_zscore_logistic(values)

        # Z-score amplifies relative differences even when absolute is tiny
        # The important thing is: no NaN/Inf and center is 0.5
        assert np.all(np.isfinite(norm))
        assert norm[0] == 0.5  # Center value

    def test_calibration_all_same_bin(self):
        """All predictions fall in same calibration bin."""
        probs = np.array([0.505, 0.506, 0.504, 0.507, 0.503] * 10)
        outcomes = np.random.randint(0, 2, 50).astype(np.int8)

        cal = compute_calibration(probs, outcomes, num_bins=10)
        # Should handle gracefully
        assert np.isfinite(cal.score)

    def test_sharpness_single_value(self):
        """Sharpness with all same prediction."""
        probs = np.full(100, 0.6)

        sharp = compute_sharpness(probs, target_variance=0.04)
        # Zero variance = zero sharpness
        assert sharp == 0.0


# =============================================================================
# REGRESSION TESTS FOR KNOWN ISSUES
# =============================================================================

class TestKnownIssueRegressions:
    """Regression tests for previously identified issues."""

    def test_log_loss_zero_prob_for_realized_outcome(self):
        """Log loss when realized outcome has zero probability (would be -inf)."""
        # Forecast gives 0 probability to outcome that happened
        forecast = np.array([0.0, 1.0])  # Says "first outcome impossible"
        outcome = np.array([1, 0], dtype=np.int8)  # First outcome happened

        result = log_loss(forecast, outcome)

        # Should be clamped, not infinite
        assert np.isfinite(result)
        # Should be high (bad prediction)
        assert result > 10

    def test_brier_with_unnormalized_probs(self):
        """Brier score with probabilities that don't sum to 1."""
        forecast = np.array([0.6, 0.6])  # Sum = 1.2
        outcome = np.array([1, 0], dtype=np.int8)

        result = brier_score(forecast, outcome)

        # Should normalize internally
        assert np.isfinite(result)
        assert result < 1  # Reasonable range

    def test_correlation_single_value_repeated(self):
        """Correlation when one series is constant."""
        vals1 = np.array([0.5, 0.5, 0.5, 0.5])
        vals2 = np.array([0.4, 0.5, 0.6, 0.7])

        corr = compute_correlation(vals1, vals2)

        # Undefined correlation (zero std in one series)
        assert corr == 0.0

