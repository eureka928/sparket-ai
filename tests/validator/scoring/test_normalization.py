"""Tests for score normalization utilities."""

import numpy as np
import pytest

from sparket.validator.scoring.aggregation.normalization import (
    normalize_zscore_logistic,
    normalize_percentile,
    normalize_minmax,
    normalize,
    normalize_grouped,
)


class TestNormalizeZscoreLogistic:
    """Tests for z-score + logistic normalization."""

    def test_empty_input(self):
        """Empty input should return empty array."""
        result = normalize_zscore_logistic(np.array([]))
        assert len(result) == 0

    def test_single_value(self):
        """Single value should normalize to 0.5."""
        result = normalize_zscore_logistic(np.array([42.0]))
        assert result[0] == pytest.approx(0.5, abs=1e-9)

    def test_mean_goes_to_half(self):
        """Mean value should normalize to ~0.5."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_zscore_logistic(values)
        # Mean is 3.0, so index 2 should be ~0.5
        assert result[2] == pytest.approx(0.5, abs=1e-9)

    def test_symmetric_around_mean(self):
        """Values equidistant from mean should be symmetric around 0.5."""
        values = np.array([0.0, 1.0, 2.0])  # Mean = 1.0
        result = normalize_zscore_logistic(values)
        # 0 and 2 are equidistant from mean
        assert result[0] + result[2] == pytest.approx(1.0, abs=1e-6)

    def test_constant_values(self):
        """Constant values should all normalize to 0.5."""
        values = np.array([5.0, 5.0, 5.0])
        result = normalize_zscore_logistic(values)
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_output_range(self):
        """Output should be in (0, 1)."""
        values = np.random.randn(100)
        result = normalize_zscore_logistic(values)
        assert np.all(result > 0) and np.all(result < 1)

    def test_alpha_steepness(self):
        """Higher alpha should give more extreme values."""
        values = np.array([0.0, 1.0, 2.0])
        result_low = normalize_zscore_logistic(values, alpha=0.5)
        result_high = normalize_zscore_logistic(values, alpha=2.0)

        # Higher alpha -> more extreme (further from 0.5)
        assert abs(result_high[0] - 0.5) > abs(result_low[0] - 0.5)


class TestNormalizePercentile:
    """Tests for percentile rank normalization."""

    def test_empty_input(self):
        """Empty input should return empty array."""
        result = normalize_percentile(np.array([]))
        assert len(result) == 0

    def test_single_value(self):
        """Single value should normalize to 0.5."""
        result = normalize_percentile(np.array([42.0]))
        assert result[0] == pytest.approx(0.5, abs=1e-9)

    def test_sorted_values(self):
        """Sorted values should give evenly spaced percentiles."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_percentile(values)

        # Percentiles should be roughly evenly spaced
        # Using rank/(n+1) formula: 1/6, 2/6, 3/6, 4/6, 5/6
        expected = np.array([1, 2, 3, 4, 5]) / 6
        np.testing.assert_array_almost_equal(result, expected)

    def test_handles_ties(self):
        """Tied values should get average rank."""
        values = np.array([1.0, 2.0, 2.0, 3.0])
        result = normalize_percentile(values)

        # Middle two values should have same percentile
        assert result[1] == result[2]

    def test_output_range(self):
        """Output should be in (0, 1)."""
        values = np.random.rand(100)
        result = normalize_percentile(values)
        assert np.all(result > 0) and np.all(result < 1)

    def test_preserves_order(self):
        """Higher values should have higher percentiles."""
        values = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = normalize_percentile(values)

        # 5.0 should have highest percentile
        assert result[4] == result.max()
        # 1.0s should have lowest percentiles
        assert result[1] == result.min() or result[3] == result.min()


class TestNormalizeMinmax:
    """Tests for min-max normalization."""

    def test_empty_input(self):
        """Empty input should return empty array."""
        result = normalize_minmax(np.array([]))
        assert len(result) == 0

    def test_single_value(self):
        """Single value should normalize to 0.5."""
        result = normalize_minmax(np.array([42.0]))
        assert result[0] == pytest.approx(0.5, abs=1e-9)

    def test_min_goes_to_zero(self):
        """Minimum value should normalize to 0."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_minmax(values)
        assert result[0] == pytest.approx(0.0, abs=1e-9)

    def test_max_goes_to_one(self):
        """Maximum value should normalize to 1."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_minmax(values)
        assert result[4] == pytest.approx(1.0, abs=1e-9)

    def test_linear_scaling(self):
        """Intermediate values should be linearly scaled."""
        values = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        result = normalize_minmax(values)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_constant_values(self):
        """Constant values should all normalize to 0.5."""
        values = np.array([5.0, 5.0, 5.0])
        result = normalize_minmax(values)
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_output_range(self):
        """Output should be in [0, 1]."""
        values = np.random.rand(100)
        result = normalize_minmax(values)
        assert np.all(result >= 0) and np.all(result <= 1)


class TestNormalize:
    """Tests for the generic normalize function."""

    def test_unknown_method_falls_back(self):
        """Unknown method should fall back to zscore_logistic."""
        values = np.array([1.0, 2.0, 3.0])
        # Note: implementation falls back to zscore_logistic for unknown methods
        result = normalize(values, method="unknown")  # type: ignore
        expected = normalize_zscore_logistic(values)
        np.testing.assert_array_almost_equal(result, expected)

    def test_zscore_logistic_method(self):
        """zscore_logistic method should work."""
        values = np.array([1.0, 2.0, 3.0])
        result = normalize(values, method="zscore_logistic")
        expected = normalize_zscore_logistic(values)
        np.testing.assert_array_almost_equal(result, expected)

    def test_percentile_method(self):
        """percentile method should work."""
        values = np.array([1.0, 2.0, 3.0])
        result = normalize(values, method="percentile")
        expected = normalize_percentile(values)
        np.testing.assert_array_almost_equal(result, expected)

    def test_minmax_method(self):
        """minmax method should work."""
        values = np.array([1.0, 2.0, 3.0])
        result = normalize(values, method="minmax")
        expected = normalize_minmax(values)
        np.testing.assert_array_almost_equal(result, expected)

    def test_default_method(self):
        """Default method should be zscore_logistic."""
        values = np.array([1.0, 2.0, 3.0])
        result_default = normalize(values)
        result_explicit = normalize(values, method="zscore_logistic")
        np.testing.assert_array_almost_equal(result_default, result_explicit)


class TestNormalizeGrouped:
    """Tests for grouped normalization."""

    def test_single_group(self):
        """Single group should normalize all together."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group_ids = np.array([0, 0, 0, 0, 0])

        result = normalize_grouped(values, group_ids, method="minmax")

        # Same as normalizing without groups
        expected = normalize_minmax(values)
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_groups(self):
        """Multiple groups should be normalized independently."""
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        group_ids = np.array([0, 0, 0, 1, 1, 1])

        result = normalize_grouped(values, group_ids, method="minmax")

        # Group 0: [1, 2, 3] -> [0, 0.5, 1]
        np.testing.assert_array_almost_equal(result[:3], [0.0, 0.5, 1.0])
        # Group 1: [10, 20, 30] -> [0, 0.5, 1]
        np.testing.assert_array_almost_equal(result[3:], [0.0, 0.5, 1.0])

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        n = 100
        values = np.random.rand(n)
        group_ids = np.random.randint(0, 5, n)

        result = normalize_grouped(values, group_ids)

        assert result.shape == values.shape

    def test_all_output_in_range(self):
        """All normalized values should be in valid range."""
        values = np.random.randn(100)
        group_ids = np.random.randint(0, 5, 100)

        result = normalize_grouped(values, group_ids, method="percentile")

        assert np.all(result >= 0) and np.all(result <= 1)

