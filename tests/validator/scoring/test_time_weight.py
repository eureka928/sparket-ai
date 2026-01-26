"""Tests for time-to-close weighting.

Verifies that earlier predictions (further from event close) receive
higher weight than late submissions.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from sparket.validator.scoring.aggregation.time_weight import (
    compute_time_factor,
    compute_time_factors,
    apply_time_bonus,
    apply_time_bonus_batch,
    # Backwards compatibility aliases
    compute_time_weight,
    compute_time_weights,
    apply_time_weighting,
)


class TestComputeTimeFactor:
    """Tests for single time factor computation."""

    def test_very_late_submission_gets_floor(self):
        """Submissions at or below min_minutes get floor_factor."""
        assert compute_time_factor(0, floor_factor=0.1) == 0.1
        assert compute_time_factor(30, floor_factor=0.1) == 0.1  # Below default min
        assert compute_time_factor(60, floor_factor=0.1) == 0.1  # At min

    def test_very_early_submission_gets_max(self):
        """Submissions at or beyond max_minutes get factor 1.0."""
        assert compute_time_factor(10080) == 1.0  # Exactly 7 days
        assert compute_time_factor(20000) == 1.0  # Beyond 7 days

    def test_middle_values_logarithmic(self):
        """Middle values should follow logarithmic scaling."""
        # 1 day = 1440 minutes
        one_day = compute_time_factor(1440)
        # 3 days = 4320 minutes
        three_days = compute_time_factor(4320)
        # 6 days = 8640 minutes
        six_days = compute_time_factor(8640)

        # All should be between floor and 1.0
        assert 0.1 < one_day < 1.0
        assert 0.1 < three_days < 1.0
        assert 0.1 < six_days < 1.0

        # Earlier should be higher
        assert one_day < three_days < six_days

    def test_negative_minutes_returns_floor(self):
        """Negative minutes (shouldn't happen) returns floor."""
        assert compute_time_factor(-100) == 0.1

    def test_custom_floor_factor(self):
        """Custom floor factor is respected."""
        assert compute_time_factor(0, floor_factor=0.05) == 0.05
        assert compute_time_factor(0, floor_factor=0.5) == 0.5

    def test_custom_min_max_minutes(self):
        """Custom min/max values work correctly."""
        # With min=30, max=120:
        # 30 minutes should be at floor
        assert compute_time_factor(30, min_minutes=30, max_minutes=120) == 0.1
        # 120 minutes should be at 1.0
        assert compute_time_factor(120, min_minutes=30, max_minutes=120) == 1.0
        # 60 minutes should be in between
        f = compute_time_factor(60, min_minutes=30, max_minutes=120)
        assert 0.1 < f < 1.0


class TestComputeTimeFactors:
    """Tests for batch time factor computation."""

    def test_array_matches_scalar(self):
        """Batch results should match individual scalar calls."""
        minutes = np.array([0, 60, 1440, 10080, 20000])
        factors = compute_time_factors(minutes)

        for i, m in enumerate(minutes):
            expected = compute_time_factor(int(m))
            assert_allclose(factors[i], expected, rtol=1e-10)

    def test_monotonic_increasing(self):
        """Factors should increase with minutes_to_close."""
        minutes = np.array([60, 120, 360, 720, 1440, 4320, 10080])
        factors = compute_time_factors(minutes)

        # Each factor should be >= previous (strictly > for non-floor/max values)
        for i in range(1, len(factors)):
            assert factors[i] >= factors[i - 1]

    def test_empty_array(self):
        """Empty array returns empty result."""
        result = compute_time_factors(np.array([], dtype=np.int64))
        assert len(result) == 0

    def test_single_element(self):
        """Single element works correctly."""
        result = compute_time_factors(np.array([1440]))
        expected = compute_time_factor(1440)
        assert_allclose(result[0], expected)

    def test_all_same_minutes(self):
        """All same values produce same factors."""
        result = compute_time_factors(np.array([1000, 1000, 1000]))
        assert np.all(result == result[0])


class TestApplyTimeBonus:
    """Tests for asymmetric time bonus application."""

    def test_positive_score_early(self):
        """Good early prediction gets full credit."""
        result = apply_time_bonus(0.5, 10080)  # 7 days out, positive score
        assert_allclose(result, 0.5 * 1.0)  # Full factor

    def test_positive_score_late(self):
        """Good late prediction gets reduced credit."""
        result = apply_time_bonus(0.5, 60)  # 1 hour out, positive score
        assert_allclose(result, 0.5 * 0.1)  # Floor factor

    def test_negative_score_early(self):
        """Bad early prediction gets clipped penalty (forgiven)."""
        # With early_penalty_clip=0.7, early bad prediction gets 0.7 penalty
        result = apply_time_bonus(-0.5, 10080, early_penalty_clip=0.7)
        expected = -0.5 * 0.7  # Clipped penalty
        assert_allclose(result, expected)

    def test_negative_score_late(self):
        """Bad late prediction gets full penalty."""
        result = apply_time_bonus(-0.5, 60, early_penalty_clip=0.7)
        # penalty_factor = 0.7 + 0.3 * (1 - 0.1) = 0.7 + 0.27 = 0.97
        expected = -0.5 * 0.97
        assert_allclose(result, expected, rtol=0.01)

    def test_zero_score_unchanged(self):
        """Zero score remains zero regardless of timing."""
        assert apply_time_bonus(0.0, 60) == 0.0
        assert apply_time_bonus(0.0, 10080) == 0.0


class TestApplyTimeBonusBatch:
    """Tests for batch asymmetric time bonus."""

    def test_matches_scalar(self):
        """Batch results match individual scalar calls."""
        scores = np.array([0.5, -0.3, 0.0, 0.8])
        minutes = np.array([60, 10080, 1440, 30])

        batch_result = apply_time_bonus_batch(scores, minutes)

        for i in range(len(scores)):
            expected = apply_time_bonus(scores[i], minutes[i])
            assert_allclose(batch_result[i], expected)

    def test_asymmetric_behavior(self):
        """Positive and negative scores treated differently."""
        # Same magnitude, opposite sign, same timing
        scores = np.array([0.5, -0.5])
        minutes = np.array([10080, 10080])  # Both early

        result = apply_time_bonus_batch(scores, minutes)

        # Positive gets full credit (0.5 * 1.0 = 0.5)
        assert_allclose(result[0], 0.5)
        # Negative gets clipped penalty (default 0.7)
        assert_allclose(result[1], -0.5 * 0.7)

    def test_incentive_structure(self):
        """Early good beats late good, early bad less penalized than late bad."""
        scores = np.array([0.5, 0.5, -0.5, -0.5])
        minutes = np.array([10080, 60, 10080, 60])  # early good, late good, early bad, late bad

        result = apply_time_bonus_batch(scores, minutes)

        # Early good > late good
        assert result[0] > result[1]
        # Early bad (less penalty) > late bad (more penalty)
        assert result[2] > result[3]


class TestApplyTimeWeighting:
    """Tests for applying time weights to scores (legacy symmetric)."""

    def test_weights_applied_correctly(self):
        """Scores are multiplied by time factors."""
        scores = np.array([1.0, 2.0, 3.0])
        minutes = np.array([60, 1440, 10080])  # late, middle, early

        weighted = apply_time_weighting(scores, minutes)
        expected_factors = compute_time_factors(minutes)

        assert_allclose(weighted, scores * expected_factors)

    def test_early_prediction_valued_more(self):
        """Same score at different times produces different weighted values."""
        scores = np.array([1.0, 1.0])  # Same raw scores
        minutes = np.array([60, 10080])  # Late vs early

        weighted = apply_time_weighting(scores, minutes)

        # Early prediction should be weighted more
        assert weighted[1] > weighted[0]
        # Late should be at floor (0.1 * 1.0 = 0.1)
        assert_allclose(weighted[0], 0.1)
        # Early should be at max (1.0 * 1.0 = 1.0)
        assert_allclose(weighted[1], 1.0)

    def test_preserves_zeros(self):
        """Zero scores remain zero after weighting."""
        scores = np.array([0.0, 0.0])
        minutes = np.array([1440, 10080])

        weighted = apply_time_weighting(scores, minutes)
        assert_allclose(weighted, 0.0)


class TestIncentiveStructure:
    """Tests verifying the incentive structure is correct."""

    def test_copy_trading_penalized(self):
        """Submitting 1 hour before close gets heavily penalized."""
        # Miner A: Predicts 3 days before close
        # Miner B: Copies at 30 minutes before close

        same_score = 0.8  # Both have same raw prediction quality

        miner_a_weighted = same_score * compute_time_factor(3 * 24 * 60)  # 3 days
        miner_b_weighted = same_score * compute_time_factor(30)  # 30 mins

        # A should get much more credit
        assert miner_a_weighted > miner_b_weighted * 5  # At least 5x more

    def test_early_accurate_beats_late_accurate(self):
        """Early accurate prediction beats late accurate prediction."""
        # Same accuracy, different timing
        early_7_days = compute_time_factor(10080)
        late_1_hour = compute_time_factor(60)

        # Early is 10x better weighted
        ratio = early_7_days / late_1_hour
        assert ratio == 10.0  # 1.0 / 0.1 = 10

    def test_progressive_decay_not_cliff(self):
        """Weighting is progressive, not a sudden cliff."""
        # Check that moving from 2 days to 1 day isn't a massive drop
        two_days = compute_time_factor(2880)
        one_day = compute_time_factor(1440)
        half_day = compute_time_factor(720)

        # Changes should be gradual
        diff_2_to_1 = two_days - one_day
        diff_1_to_half = one_day - half_day

        # Both differences should be positive and reasonable
        assert 0 < diff_2_to_1 < 0.5
        assert 0 < diff_1_to_half < 0.5


class TestEdgeCases:
    """Edge case testing."""

    def test_float_minutes(self):
        """Float inputs work (e.g., from division)."""
        result = compute_time_factor(1440.5)
        assert 0.0 <= result <= 1.0

    def test_large_minutes(self):
        """Very large values capped at 1.0."""
        assert compute_time_factor(1_000_000) == 1.0

    def test_nan_in_batch(self):
        """NaN in batch produces NaN in that position (log behavior)."""
        minutes = np.array([1440, np.nan, 10080])
        with np.errstate(invalid='ignore'):
            factors = compute_time_factors(minutes)
        # First and last should be valid
        assert np.isfinite(factors[0])
        assert np.isfinite(factors[2])

    def test_zero_minutes(self):
        """Zero minutes returns floor factor."""
        assert compute_time_factor(0) == 0.1

    def test_exactly_min_minutes(self):
        """Exactly at min_minutes threshold."""
        assert compute_time_factor(60, min_minutes=60) == 0.1

    def test_exactly_max_minutes(self):
        """Exactly at max_minutes threshold."""
        assert compute_time_factor(10080, max_minutes=10080) == 1.0


class TestBackwardsCompatibility:
    """Test that old function names still work."""

    def test_compute_time_weight_alias(self):
        """compute_time_weight works as alias for compute_time_factor."""
        assert compute_time_weight(1440) == compute_time_factor(1440)

    def test_compute_time_weights_alias(self):
        """compute_time_weights works as alias for compute_time_factors."""
        minutes = np.array([60, 1440, 10080])
        assert_allclose(compute_time_weights(minutes), compute_time_factors(minutes))


class TestCombinedWeighting:
    """Tests simulating combined decay + time weighting."""

    def test_combined_weight_behavior(self):
        """Simulate how decay and time factors combine."""
        from sparket.validator.scoring.aggregation.decay import compute_decay_weights

        now = 1000000.0

        submitted_ts = np.array([
            now - 5 * 86400,  # 5 days ago
            now - 2 * 86400,  # 2 days ago
            now - 3600,       # 1 hour ago
        ])

        minutes_to_close = np.array([
            7 * 24 * 60,   # 7 days to event
            1 * 24 * 60,   # 1 day to event
            60,            # 1 hour to event
        ])

        decay_weights = compute_decay_weights(submitted_ts, now, half_life_days=10)
        time_factors = compute_time_factors(minutes_to_close)
        combined = decay_weights * time_factors

        # C (copy-trading) should have lowest combined despite being most recent
        assert combined[2] < combined[0]

        # A (early prediction 5 days ago) should still have good value
        assert combined[0] > combined[1] * 0.5


__all__ = [
    "TestComputeTimeFactor",
    "TestComputeTimeFactors",
    "TestApplyTimeBonus",
    "TestApplyTimeBonusBatch",
    "TestApplyTimeWeighting",
    "TestIncentiveStructure",
    "TestEdgeCases",
    "TestBackwardsCompatibility",
    "TestCombinedWeighting",
]
