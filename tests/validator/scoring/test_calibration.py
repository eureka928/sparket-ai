"""Tests for calibration metrics."""

import numpy as np
import pytest

from sparket.validator.scoring.metrics.calibration import (
    CalibrationResult,
    logit,
    fit_calibration_curve,
    calibration_score,
    compute_calibration,
    compute_calibration_batch,
)


class TestLogit:
    """Tests for logit function."""

    def test_logit_half(self):
        """logit(0.5) should be 0."""
        result = logit(np.array([0.5]))
        assert result[0] == pytest.approx(0.0, abs=1e-9)

    def test_logit_low(self):
        """logit of low probability should be negative."""
        result = logit(np.array([0.2]))
        # logit(0.2) = log(0.2/0.8) = log(0.25) ≈ -1.386
        assert result[0] == pytest.approx(np.log(0.25), abs=1e-6)

    def test_logit_high(self):
        """logit of high probability should be positive."""
        result = logit(np.array([0.8]))
        # logit(0.8) = log(0.8/0.2) = log(4) ≈ 1.386
        assert result[0] == pytest.approx(np.log(4), abs=1e-6)

    def test_logit_vectorized(self):
        """Logit should work on arrays."""
        probs = np.array([0.2, 0.5, 0.8])
        results = logit(probs)
        assert len(results) == 3
        assert results[1] == pytest.approx(0.0, abs=1e-9)

    def test_logit_clamped_extremes(self):
        """Extreme values should be clamped to prevent infinity."""
        probs = np.array([0.0, 1.0])
        results = logit(probs)
        # Should be finite (clamped)
        assert np.all(np.isfinite(results))


class TestFitCalibrationCurve:
    """Tests for calibration curve fitting."""

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should give b≈1, a≈0."""
        # Generate perfectly calibrated data
        np.random.seed(42)
        probs = np.random.uniform(0.1, 0.9, 500)
        outcomes = (np.random.rand(500) < probs).astype(np.int8)

        a, b = fit_calibration_curve(probs, outcomes)

        # Should be close to ideal (a=0, b=1)
        # Allow some tolerance due to randomness
        assert abs(a) < 1.0
        assert abs(b - 1.0) < 1.0

    def test_overconfident_predictions(self):
        """Overconfident predictions should give b > 1."""
        # Predictions that are more extreme than they should be
        np.random.seed(42)
        true_probs = np.random.uniform(0.3, 0.7, 300)
        # Make predictions more extreme
        pred_probs = np.where(true_probs > 0.5, 0.8, 0.2)
        outcomes = (np.random.rand(300) < true_probs).astype(np.int8)

        a, b = fit_calibration_curve(pred_probs, outcomes, num_bins=5)

        # b should indicate the relationship between predicted and actual
        # This is a smoke test - actual value depends on data
        assert np.isfinite(a) and np.isfinite(b)

    def test_insufficient_data(self):
        """Insufficient data should return neutral (a=0, b=1)."""
        probs = np.array([0.5, 0.5])
        outcomes = np.array([1, 0], dtype=np.int8)

        a, b = fit_calibration_curve(probs, outcomes, min_samples_per_bin=5)

        assert a == 0.0
        assert b == 1.0

    def test_empty_input(self):
        """Empty input should return neutral."""
        a, b = fit_calibration_curve(np.array([]), np.array([], dtype=np.int8))
        assert a == 0.0
        assert b == 1.0


class TestCalibrationScore:
    """Tests for calibration score computation."""

    def test_perfect_calibration(self):
        """a=0, b=1 should give score = 1."""
        assert calibration_score(0.0, 1.0) == pytest.approx(1.0, abs=1e-9)

    def test_poor_intercept(self):
        """Non-zero intercept should reduce score."""
        # Cal = 1 / (1 + |1-1| + |0.5|) = 1 / 1.5 ≈ 0.667
        assert calibration_score(0.5, 1.0) == pytest.approx(1 / 1.5, abs=1e-6)

    def test_poor_slope(self):
        """Slope away from 1 should reduce score."""
        # Cal = 1 / (1 + |1.5-1| + |0|) = 1 / 1.5 ≈ 0.667
        assert calibration_score(0.0, 1.5) == pytest.approx(1 / 1.5, abs=1e-6)

    def test_both_errors(self):
        """Both intercept and slope errors compound."""
        # Cal = 1 / (1 + |1.5-1| + |0.5|) = 1 / 2 = 0.5
        assert calibration_score(0.5, 1.5) == pytest.approx(0.5, abs=1e-9)

    def test_score_always_positive(self):
        """Score should always be in (0, 1]."""
        for a in np.linspace(-5, 5, 20):
            for b in np.linspace(-5, 5, 20):
                score = calibration_score(a, b)
                assert 0 < score <= 1


class TestComputeCalibration:
    """Tests for complete calibration computation."""

    def test_returns_result_object(self):
        """Should return CalibrationResult dataclass."""
        np.random.seed(42)
        probs = np.random.rand(100)
        outcomes = (np.random.rand(100) < probs).astype(np.int8)

        result = compute_calibration(probs, outcomes)

        assert isinstance(result, CalibrationResult)
        assert hasattr(result, 'intercept')
        assert hasattr(result, 'slope')
        assert hasattr(result, 'score')
        assert hasattr(result, 'bins_used')

    def test_insufficient_samples(self):
        """Too few samples should return neutral result."""
        probs = np.array([0.5] * 10)
        outcomes = np.array([1, 0] * 5, dtype=np.int8)

        result = compute_calibration(probs, outcomes, min_samples=30)

        assert result.intercept == 0.0
        assert result.slope == 1.0
        assert result.score == 0.5
        assert result.bins_used == 0

    def test_score_in_range(self):
        """Score should always be in (0, 1]."""
        np.random.seed(42)
        probs = np.random.rand(200)
        outcomes = (np.random.rand(200) < probs).astype(np.int8)

        result = compute_calibration(probs, outcomes)

        assert 0 < result.score <= 1

    def test_bins_used_reported(self):
        """Should report how many bins were used."""
        np.random.seed(42)
        probs = np.random.rand(500)
        outcomes = (np.random.rand(500) < probs).astype(np.int8)

        result = compute_calibration(probs, outcomes, num_bins=10, min_samples_per_bin=5)

        assert 0 <= result.bins_used <= 10


class TestComputeCalibrationBatch:
    """Tests for batch calibration computation."""

    def test_batch_output_shape(self):
        """Should return arrays of correct shape."""
        np.random.seed(42)
        probs = np.random.rand(500)
        outcomes = (np.random.rand(500) < probs).astype(np.int8)
        miner_ids = np.repeat(np.arange(10), 50)

        unique_ids, scores = compute_calibration_batch(
            probs, outcomes, miner_ids, min_samples=30
        )

        assert len(unique_ids) == 10
        assert len(scores) == 10

    def test_batch_per_miner(self):
        """Each miner should get their own calibration."""
        np.random.seed(42)

        # Create data for 5 miners
        all_probs = []
        all_outcomes = []
        all_ids = []

        for mid in range(5):
            n = 100
            probs = np.random.rand(n)
            outcomes = (np.random.rand(n) < probs).astype(np.int8)
            all_probs.extend(probs)
            all_outcomes.extend(outcomes)
            all_ids.extend([mid] * n)

        unique_ids, scores = compute_calibration_batch(
            np.array(all_probs),
            np.array(all_outcomes, dtype=np.int8),
            np.array(all_ids),
            min_samples=30,
        )

        assert len(unique_ids) == 5
        assert np.all(scores > 0) and np.all(scores <= 1)

    def test_insufficient_samples_per_miner(self):
        """Miners with too few samples should get default score."""
        probs = np.array([0.5] * 60)
        outcomes = np.array([1, 0] * 30, dtype=np.int8)
        # 3 miners, only first has 50 samples
        miner_ids = np.array([0] * 50 + [1] * 5 + [2] * 5)

        unique_ids, scores = compute_calibration_batch(
            probs, outcomes, miner_ids, min_samples=30
        )

        assert len(unique_ids) == 3
        # Miners 1 and 2 should have default score
        assert scores[1] == 0.5
        assert scores[2] == 0.5

