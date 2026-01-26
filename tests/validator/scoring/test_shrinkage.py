"""Tests for Bayesian shrinkage toward population mean."""

import numpy as np
import pytest

from sparket.validator.scoring.aggregation.shrinkage import (
    shrink_toward_mean,
    compute_population_mean,
    shrink_grouped,
)


class TestShrinkTowardMean:
    """Tests for shrinkage computation."""

    def test_large_sample_no_shrinkage(self):
        """Large n_eff should result in minimal shrinkage."""
        raw_values = np.array([0.8, 0.2, 0.9])
        n_effs = np.array([1000.0, 1000.0, 1000.0])
        pop_mean = 0.5
        k = 50.0

        shrunk = shrink_toward_mean(raw_values, n_effs, k, pop_mean)

        # With n_eff=1000, k=50: weight_raw = 1000/(1000+50) ≈ 0.95
        # So shrunk should be very close to raw
        np.testing.assert_array_almost_equal(shrunk, raw_values, decimal=1)

    def test_small_sample_more_shrinkage(self):
        """Small n_eff should result in more shrinkage than large n_eff."""
        raw_values = np.array([0.9])
        pop_mean = 0.5
        k = 50.0

        # Small n_eff
        shrunk_small = shrink_toward_mean(raw_values, np.array([10.0]), k, pop_mean)

        # Large n_eff
        shrunk_large = shrink_toward_mean(raw_values, np.array([500.0]), k, pop_mean)

        # Small sample should be shrunk more (closer to pop_mean)
        assert abs(shrunk_small[0] - pop_mean) < abs(shrunk_large[0] - pop_mean)
        # Both should be between raw and pop_mean
        assert pop_mean < shrunk_small[0] < raw_values[0]
        assert pop_mean < shrunk_large[0] < raw_values[0]

    def test_zero_sample_full_shrinkage(self):
        """Zero n_eff should result in full shrinkage to mean."""
        raw_values = np.array([0.9, 0.1])
        n_effs = np.array([0.0, 0.0])
        pop_mean = 0.5
        k = 50.0

        shrunk = shrink_toward_mean(raw_values, n_effs, k, pop_mean)

        np.testing.assert_array_almost_equal(shrunk, [0.5, 0.5])

    def test_shrinkage_direction(self):
        """Shrinkage should move values toward population mean."""
        raw_values = np.array([0.9, 0.1])  # One high, one low
        n_effs = np.array([20.0, 20.0])
        pop_mean = 0.5
        k = 50.0

        shrunk = shrink_toward_mean(raw_values, n_effs, k, pop_mean)

        # High value should decrease, low value should increase
        assert shrunk[0] < raw_values[0]
        assert shrunk[1] > raw_values[1]

    def test_auto_compute_pop_mean(self):
        """Should auto-compute population mean if not provided."""
        raw_values = np.array([0.2, 0.4, 0.6, 0.8])
        n_effs = np.array([100.0, 100.0, 100.0, 100.0])
        k = 50.0

        shrunk = shrink_toward_mean(raw_values, n_effs, k)

        # Population mean = 0.5
        # Values below mean should increase, values above should decrease
        assert shrunk[0] > raw_values[0]  # 0.2 -> higher
        assert shrunk[1] > raw_values[1]  # 0.4 -> higher
        assert shrunk[2] < raw_values[2]  # 0.6 -> lower
        assert shrunk[3] < raw_values[3]  # 0.8 -> lower

        # All should be pulled toward 0.5
        for i, s in enumerate(shrunk):
            assert abs(s - 0.5) < abs(raw_values[i] - 0.5)

    def test_custom_weights_for_pop_mean(self):
        """Should use custom weights for population mean computation."""
        raw_values = np.array([0.0, 1.0])
        n_effs = np.array([50.0, 50.0])
        k = 50.0
        custom_weights = np.array([1.0, 3.0])  # Weight 1.0 more heavily

        shrunk = shrink_toward_mean(raw_values, n_effs, k, weights=custom_weights)

        # Pop mean = (0*1 + 1*3) / 4 = 0.75
        # 0.0 should increase toward 0.75
        assert shrunk[0] > raw_values[0]
        # 1.0 should decrease toward 0.75
        assert shrunk[1] < raw_values[1]
        # Both should be closer to 0.75 than they started
        pop_mean = 0.75
        assert abs(shrunk[0] - pop_mean) < abs(raw_values[0] - pop_mean)
        assert abs(shrunk[1] - pop_mean) < abs(raw_values[1] - pop_mean)

    def test_weighted_pop_mean(self):
        """Population mean should be weighted by n_eff."""
        raw_values = np.array([0.0, 1.0])
        n_effs = np.array([100.0, 10.0])  # Weight 0.0 more heavily
        k = 50.0

        shrunk = shrink_toward_mean(raw_values, n_effs, k)

        # Weighted mean ≈ (0*100 + 1*10)/(100+10) = 10/110 ≈ 0.09
        # 0.0 should stay close to 0 (high n_eff + low pop_mean)
        assert shrunk[0] < 0.15
        # 1.0 should shrink toward the low pop_mean, but less than fully
        # because it has low n_eff with log scaling
        assert shrunk[1] < raw_values[1]  # Must decrease from 1.0

    def test_different_k_values(self):
        """Higher k should result in more shrinkage."""
        raw_values = np.array([0.9])
        n_effs = np.array([50.0])
        pop_mean = 0.5

        shrunk_low_k = shrink_toward_mean(raw_values, n_effs, k=10.0, population_mean=pop_mean)
        shrunk_high_k = shrink_toward_mean(raw_values, n_effs, k=200.0, population_mean=pop_mean)

        # Higher k -> more shrinkage -> closer to mean
        assert abs(shrunk_high_k[0] - pop_mean) < abs(shrunk_low_k[0] - pop_mean)

    def test_empty_array(self):
        """Empty array should return empty."""
        result = shrink_toward_mean(np.array([]), np.array([]), 50.0)
        assert len(result) == 0

    def test_all_zero_neff_with_no_pop_mean(self):
        """All zero n_eff with no pop_mean should handle gracefully."""
        raw_values = np.array([0.9, 0.1])
        n_effs = np.array([0.0, 0.0])
        k = 50.0

        # No explicit pop_mean, so it computes from data
        shrunk = shrink_toward_mean(raw_values, n_effs, k)

        # With all zero n_eff, should return population mean = 0.5 for each
        np.testing.assert_array_almost_equal(shrunk, [0.5, 0.5])


class TestComputePopulationMean:
    """Tests for population mean computation."""

    def test_simple_mean(self):
        """Without weights, should compute simple mean."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_population_mean(values) == pytest.approx(2.5, abs=1e-9)

    def test_weighted_mean(self):
        """With weights, should compute weighted mean."""
        values = np.array([0.0, 1.0])
        weights = np.array([3.0, 1.0])
        # (0*3 + 1*1) / (3+1) = 0.25
        assert compute_population_mean(values, weights) == pytest.approx(0.25, abs=1e-9)

    def test_empty_values(self):
        """Empty values should return 0."""
        assert compute_population_mean(np.array([])) == 0.0

    def test_zero_weights_fallback(self):
        """Zero weights should fall back to simple mean."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])
        # Falls back to simple mean = 2.0
        assert compute_population_mean(values, weights) == pytest.approx(2.0, abs=1e-9)


class TestShrinkGrouped:
    """Tests for grouped shrinkage."""

    def test_single_group(self):
        """Single group should work like regular shrinkage."""
        values = np.array([0.2, 0.4, 0.6, 0.8])
        n_effs = np.array([50.0, 50.0, 50.0, 50.0])
        group_ids = np.array([0, 0, 0, 0])
        k = 50.0

        result = shrink_grouped(values, n_effs, group_ids, k)

        expected = shrink_toward_mean(values, n_effs, k)
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_groups_independent(self):
        """Multiple groups should be shrunk independently."""
        # Group 0: high values, Group 1: low values
        values = np.array([0.8, 0.9, 0.1, 0.2])
        n_effs = np.array([20.0, 20.0, 20.0, 20.0])
        group_ids = np.array([0, 0, 1, 1])
        k = 50.0

        result = shrink_grouped(values, n_effs, group_ids, k)

        # Group 0 mean = 0.85, Group 1 mean = 0.15
        # Group 0 values should shrink toward 0.85
        # Group 1 values should shrink toward 0.15

        # Check that each group shrinks toward its own mean, not global
        # Group 0: values are high, so shrunk values should still be high
        assert result[0] > 0.7 and result[1] > 0.7
        # Group 1: values are low, so shrunk values should still be low
        assert result[2] < 0.3 and result[3] < 0.3

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        n = 100
        values = np.random.rand(n)
        n_effs = np.random.rand(n) * 100
        group_ids = np.random.randint(0, 5, n)

        result = shrink_grouped(values, n_effs, group_ids, k=50.0)

        assert result.shape == values.shape

    def test_values_bounded(self):
        """Shrunk values should stay within reasonable bounds."""
        # Values in [0, 1]
        values = np.random.rand(100)
        n_effs = np.random.rand(100) * 100
        group_ids = np.random.randint(0, 5, 100)

        result = shrink_grouped(values, n_effs, group_ids, k=50.0)

        # Shrunk values should be in [0, 1] since that's where inputs are
        assert np.all(result >= 0) and np.all(result <= 1)

