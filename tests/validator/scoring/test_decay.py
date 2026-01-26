"""Tests for time decay weighting."""

import numpy as np
import pytest

from sparket.validator.scoring.aggregation.decay import (
    compute_decay_weight,
    compute_decay_weights,
    effective_sample_size,
    weighted_mean,
    weighted_std,
    weighted_aggregates,
    weighted_aggregates_batch,
)


class TestComputeDecayWeight:
    """Tests for single decay weight computation."""

    def test_zero_age(self):
        """Zero age should give weight = 1."""
        assert compute_decay_weight(0.0, 10.0) == pytest.approx(1.0, abs=1e-9)

    def test_half_life(self):
        """Age = half_life should give weight = 0.5."""
        assert compute_decay_weight(10.0, 10.0) == pytest.approx(0.5, abs=1e-9)

    def test_double_half_life(self):
        """Age = 2 * half_life should give weight = 0.25."""
        assert compute_decay_weight(20.0, 10.0) == pytest.approx(0.25, abs=1e-9)

    def test_negative_age_clamped(self):
        """Negative age should be clamped to 0."""
        assert compute_decay_weight(-5.0, 10.0) == pytest.approx(1.0, abs=1e-9)

    def test_very_old(self):
        """Very old observations should have low weight."""
        weight = compute_decay_weight(100.0, 10.0)  # 10 half-lives
        assert weight < 0.001

    def test_different_half_lives(self):
        """Different half-lives should scale decay."""
        w1 = compute_decay_weight(10.0, 10.0)  # 1 half-life
        w2 = compute_decay_weight(10.0, 20.0)  # 0.5 half-lives

        assert w1 == pytest.approx(0.5, abs=1e-9)
        assert w2 == pytest.approx(0.5 ** 0.5, abs=1e-6)  # sqrt(0.5)


class TestComputeDecayWeights:
    """Tests for batch decay weight computation."""

    def test_empty_input(self):
        """Empty input should return empty array."""
        weights = compute_decay_weights(np.array([]), 1000.0, 10.0)
        assert len(weights) == 0

    def test_single_timestamp(self):
        """Single timestamp should work."""
        # 1 day ago with 10-day half-life
        timestamps = np.array([0.0])
        ref = 86400.0  # 1 day later

        weights = compute_decay_weights(timestamps, ref, 10.0)

        assert len(weights) == 1
        # 1 day = 1/10 half-life
        expected = 0.5 ** 0.1
        assert weights[0] == pytest.approx(expected, abs=1e-6)

    def test_multiple_timestamps(self):
        """Multiple timestamps should each get correct weight."""
        # 0, 5, 10 days ago with 10-day half-life
        ref = 10 * 86400  # Reference is 10 days from epoch
        timestamps = np.array([10 * 86400, 5 * 86400, 0])  # 0, 5, 10 days ago

        weights = compute_decay_weights(timestamps, ref, 10.0)

        assert weights[0] == pytest.approx(1.0, abs=1e-6)  # 0 days
        assert weights[1] == pytest.approx(0.5 ** 0.5, abs=1e-6)  # 5 days
        assert weights[2] == pytest.approx(0.5, abs=1e-6)  # 10 days

    def test_weights_monotonically_decrease(self):
        """Older timestamps should have smaller weights."""
        ref = 30 * 86400
        timestamps = np.arange(0, 30 * 86400, 86400)  # 0 to 29 days

        weights = compute_decay_weights(timestamps, ref, 10.0)

        # Weights should decrease as timestamps get older
        assert np.all(np.diff(weights) >= 0)  # Older = smaller

    def test_all_weights_positive(self):
        """All weights should be positive."""
        ref = 100 * 86400
        timestamps = np.random.uniform(0, ref, 100)

        weights = compute_decay_weights(timestamps, ref, 10.0)

        assert np.all(weights > 0)


class TestEffectiveSampleSize:
    """Tests for effective sample size computation."""

    def test_equal_weights(self):
        """Equal weights should give n_eff = sum(weights)."""
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        assert effective_sample_size(weights) == pytest.approx(4.0, abs=1e-9)

    def test_decaying_weights(self):
        """Decaying weights should give n_eff < n."""
        weights = np.array([1.0, 0.5, 0.25, 0.125])
        assert effective_sample_size(weights) == pytest.approx(1.875, abs=1e-9)

    def test_empty_weights(self):
        """Empty weights should give n_eff = 0."""
        weights = np.array([])
        assert effective_sample_size(weights) == 0.0


class TestWeightedMean:
    """Tests for weighted mean computation."""

    def test_empty_values(self):
        """Empty values should return 0."""
        values = np.array([])
        weights = np.array([])
        assert weighted_mean(values, weights) == 0.0

    def test_equal_weights(self):
        """Equal weights should give simple mean."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        assert weighted_mean(values, weights) == pytest.approx(2.5, abs=1e-9)

    def test_unequal_weights(self):
        """Unequal weights should bias toward high-weight values."""
        values = np.array([1.0, 2.0])
        weights = np.array([1.0, 3.0])  # Weight 2.0 more heavily
        # (1*1 + 2*3) / 4 = 7/4 = 1.75
        assert weighted_mean(values, weights) == pytest.approx(1.75, abs=1e-9)

    def test_single_value(self):
        """Single value should return that value."""
        assert weighted_mean(np.array([5.0]), np.array([1.0])) == 5.0

    def test_zero_weights(self):
        """All zero weights should return 0."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])
        assert weighted_mean(values, weights) == 0.0


class TestWeightedStd:
    """Tests for weighted standard deviation."""

    def test_zero_weights(self):
        """Zero weights should return 0."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])
        assert weighted_std(values, weights) == 0.0

    def test_constant_values(self):
        """Constant values should have std = 0."""
        values = np.array([5.0, 5.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0])
        assert weighted_std(values, weights) == pytest.approx(0.0, abs=1e-9)

    def test_simple_case(self):
        """Test with known std."""
        values = np.array([0.0, 1.0])
        weights = np.array([1.0, 1.0])
        # mean = 0.5, var = ((0-0.5)^2 + (1-0.5)^2)/2 = 0.25, std = 0.5
        assert weighted_std(values, weights) == pytest.approx(0.5, abs=1e-9)

    def test_with_precomputed_mean(self):
        """Should use precomputed mean if provided."""
        values = np.array([0.0, 1.0])
        weights = np.array([1.0, 1.0])
        std = weighted_std(values, weights, mean=0.5)
        assert std == pytest.approx(0.5, abs=1e-9)

    def test_unequal_weights(self):
        """Unequal weights should affect std."""
        values = np.array([0.0, 1.0])
        weights = np.array([3.0, 1.0])  # Weight 0 more heavily
        # weighted mean = (0*3 + 1*1)/4 = 0.25
        # weighted var = (3*(0-0.25)^2 + 1*(1-0.25)^2)/4 = (0.1875 + 0.5625)/4 = 0.1875
        # std = sqrt(0.1875) ≈ 0.433
        std = weighted_std(values, weights)
        assert std == pytest.approx(np.sqrt(0.1875), abs=1e-6)


class TestWeightedAggregates:
    """Tests for combined weighted aggregates."""

    def test_returns_all_three(self):
        """Should return (mean, std, n_eff)."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])

        mean, std, n_eff = weighted_aggregates(values, weights)

        assert mean == pytest.approx(2.0, abs=1e-9)
        assert n_eff == pytest.approx(3.0, abs=1e-9)
        assert std >= 0

    def test_zero_weights(self):
        """Zero weights should return all zeros."""
        values = np.array([1.0, 2.0])
        weights = np.array([0.0, 0.0])

        mean, std, n_eff = weighted_aggregates(values, weights)

        assert mean == 0.0
        assert std == 0.0
        assert n_eff == 0.0


class TestWeightedAggregatesBatch:
    """Tests for batch weighted aggregates by group."""

    def test_single_group(self):
        """Single group should work like regular aggregates."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        group_ids = np.array([0, 0, 0])

        unique_ids, means, stds, n_effs = weighted_aggregates_batch(
            values, weights, group_ids
        )

        assert len(unique_ids) == 1
        assert unique_ids[0] == 0
        assert means[0] == pytest.approx(2.0, abs=1e-9)

    def test_multiple_groups(self):
        """Multiple groups should be computed separately."""
        values = np.array([1.0, 2.0, 10.0, 20.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        group_ids = np.array([0, 0, 1, 1])

        unique_ids, means, stds, n_effs = weighted_aggregates_batch(
            values, weights, group_ids
        )

        assert len(unique_ids) == 2
        # Group 0: mean = 1.5
        idx0 = np.where(unique_ids == 0)[0][0]
        assert means[idx0] == pytest.approx(1.5, abs=1e-9)
        # Group 1: mean = 15.0
        idx1 = np.where(unique_ids == 1)[0][0]
        assert means[idx1] == pytest.approx(15.0, abs=1e-9)

    def test_returns_correct_shapes(self):
        """Output arrays should have correct shapes."""
        n_groups = 5
        n_per_group = 20
        values = np.random.rand(n_groups * n_per_group)
        weights = np.random.rand(n_groups * n_per_group)
        group_ids = np.repeat(np.arange(n_groups), n_per_group)

        unique_ids, means, stds, n_effs = weighted_aggregates_batch(
            values, weights, group_ids
        )

        assert len(unique_ids) == n_groups
        assert len(means) == n_groups
        assert len(stds) == n_groups
        assert len(n_effs) == n_groups

