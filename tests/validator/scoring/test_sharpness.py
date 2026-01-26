"""Tests for sharpness metrics."""

import numpy as np
import pytest

from sparket.validator.scoring.metrics.sharpness import (
    compute_variance,
    compute_sharpness,
    compute_sharpness_batch,
)


class TestComputeVariance:
    """Tests for variance computation."""

    def test_constant_values(self):
        """Constant values should have variance 0."""
        values = np.array([0.5, 0.5, 0.5, 0.5])
        assert compute_variance(values) == pytest.approx(0.0, abs=1e-9)

    def test_simple_variance(self):
        """Simple case with known variance."""
        # [0, 1] -> mean=0.5, var = ((0-0.5)^2 + (1-0.5)^2)/2 = 0.25
        values = np.array([0.0, 1.0])
        assert compute_variance(values) == pytest.approx(0.25, abs=1e-9)

    def test_larger_sample(self):
        """Test with larger sample."""
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # mean = 0.3, var = sum((x-0.3)^2)/5
        expected = np.var(values, ddof=0)
        assert compute_variance(values) == pytest.approx(expected, abs=1e-9)

    def test_empty_array(self):
        """Empty array should return 0."""
        assert compute_variance(np.array([])) == 0.0

    def test_single_value(self):
        """Single value should return 0."""
        assert compute_variance(np.array([0.5])) == 0.0


class TestComputeSharpness:
    """Tests for sharpness score computation."""

    def test_constant_predictions_low_sharpness(self):
        """Constant predictions (always 0.5) should have low sharpness."""
        probs = np.full(100, 0.5)
        sharp = compute_sharpness(probs, target_variance=0.05, min_samples=30)
        assert sharp == pytest.approx(0.0, abs=1e-9)

    def test_varied_predictions_high_sharpness(self):
        """Varied predictions should have higher sharpness."""
        probs = np.array([0.1, 0.9] * 50)  # Very varied
        sharp = compute_sharpness(probs, target_variance=0.05, min_samples=30)
        # variance of [0.1, 0.9, ...] is 0.16, which is >> 0.05
        # so sharpness should be capped at 1.0
        assert sharp == pytest.approx(1.0, abs=1e-9)

    def test_moderate_variance(self):
        """Moderate variance should give intermediate sharpness."""
        # Create data with variance ≈ 0.025
        probs = np.array([0.4, 0.6] * 50)  # var ≈ 0.01
        sharp = compute_sharpness(probs, target_variance=0.05, min_samples=30)
        # sharp = min(1, 0.01/0.05) = 0.2
        assert sharp == pytest.approx(0.2, abs=1e-2)

    def test_insufficient_samples(self):
        """Too few samples should return 0.5 (neutral)."""
        probs = np.array([0.3, 0.7])  # Only 2 samples
        sharp = compute_sharpness(probs, target_variance=0.05, min_samples=30)
        assert sharp == 0.5

    def test_zero_target_variance(self):
        """Zero target variance edge case."""
        probs = np.array([0.1, 0.9] * 50)
        sharp = compute_sharpness(probs, target_variance=0.0, min_samples=30)
        # Any variance > 0 should give 1.0
        assert sharp == 1.0

    def test_sharpness_capped_at_one(self):
        """Sharpness should never exceed 1.0."""
        probs = np.array([0.01, 0.99] * 50)  # Very high variance
        sharp = compute_sharpness(probs, target_variance=0.01, min_samples=30)
        assert sharp == 1.0

    def test_sharpness_range(self):
        """Sharpness should always be in [0, 1]."""
        for _ in range(20):
            probs = np.random.rand(100)
            sharp = compute_sharpness(probs, target_variance=0.05, min_samples=30)
            assert 0 <= sharp <= 1


class TestComputeSharpnessBatch:
    """Tests for batch sharpness computation."""

    def test_batch_output_shape(self):
        """Should return arrays of correct shape."""
        probs = np.random.rand(500)
        miner_ids = np.repeat(np.arange(10), 50)

        unique_ids, scores = compute_sharpness_batch(
            probs, miner_ids, target_variance=0.05, min_samples=30
        )

        assert len(unique_ids) == 10
        assert len(scores) == 10

    def test_batch_per_miner(self):
        """Each miner should get their own sharpness."""
        # Create miners with different prediction patterns
        all_probs = []
        all_ids = []

        # Miner 0: constant predictions (low sharpness)
        all_probs.extend([0.5] * 100)
        all_ids.extend([0] * 100)

        # Miner 1: varied predictions (high sharpness)
        all_probs.extend([0.1, 0.9] * 50)
        all_ids.extend([1] * 100)

        unique_ids, scores = compute_sharpness_batch(
            np.array(all_probs),
            np.array(all_ids),
            target_variance=0.05,
            min_samples=30,
        )

        assert len(unique_ids) == 2
        assert scores[0] < scores[1]  # Miner 0 should be less sharp

    def test_insufficient_samples_per_miner(self):
        """Miners with too few samples should get default score."""
        probs = np.array([0.5] * 60)
        miner_ids = np.array([0] * 50 + [1] * 5 + [2] * 5)

        unique_ids, scores = compute_sharpness_batch(
            probs, miner_ids, target_variance=0.05, min_samples=30
        )

        assert len(unique_ids) == 3
        # Miners 1 and 2 should have default score
        assert scores[1] == 0.5
        assert scores[2] == 0.5

    def test_scores_in_range(self):
        """All scores should be in [0, 1]."""
        probs = np.random.rand(500)
        miner_ids = np.repeat(np.arange(10), 50)

        unique_ids, scores = compute_sharpness_batch(
            probs, miner_ids, target_variance=0.05, min_samples=30
        )

        assert np.all(scores >= 0) and np.all(scores <= 1)

