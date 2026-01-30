"""Tests for isotonic calibration."""

import pytest
from sparket.miner.custom.models.calibration.isotonic import IsotonicCalibrator


class TestIsotonicCalibrator:
    """Tests for IsotonicCalibrator."""

    def test_not_fitted_returns_raw(self):
        """When not fitted, should return raw probability."""
        calibrator = IsotonicCalibrator(min_samples=100)

        assert not calibrator.is_fitted
        assert calibrator.calibrate(0.65) == 0.65

    def test_fit_requires_min_samples(self):
        """Fit should fail if not enough samples."""
        calibrator = IsotonicCalibrator(min_samples=100)

        # Add only 50 samples
        for i in range(50):
            calibrator.add_sample(predicted=0.5, actual=1.0)

        result = calibrator.fit()
        assert result is False
        assert not calibrator.is_fitted

    def test_fit_succeeds_with_enough_samples(self):
        """Fit should succeed with enough samples."""
        calibrator = IsotonicCalibrator(min_samples=10)

        # Add 20 samples
        for i in range(20):
            pred = i / 20
            actual = 1.0 if i > 10 else 0.0
            calibrator.add_sample(predicted=pred, actual=actual)

        result = calibrator.fit()
        assert result is True
        assert calibrator.is_fitted

    def test_calibration_is_monotonic(self):
        """Calibrated values should be monotonically increasing."""
        calibrator = IsotonicCalibrator(min_samples=10)

        # Add samples with clear pattern
        for i in range(100):
            pred = i / 100
            actual = 1.0 if i > 50 else 0.0
            calibrator.add_sample(predicted=pred, actual=actual)

        calibrator.fit()

        # Check monotonicity
        prev = 0.0
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            curr = calibrator.calibrate(p)
            assert curr >= prev, f"Not monotonic: {prev} -> {curr}"
            prev = curr

    def test_calibrate_pair_sums_to_one(self):
        """Calibrated pair should sum to 1."""
        calibrator = IsotonicCalibrator(min_samples=10)

        # Add samples
        for i in range(20):
            calibrator.add_sample(predicted=i / 20, actual=1.0 if i > 10 else 0.0)

        calibrator.fit()

        home, away = calibrator.calibrate_pair(0.65, 0.35)
        assert home + away == pytest.approx(1.0, abs=0.001)

    def test_get_calibration_stats(self):
        """Should return calibration statistics."""
        calibrator = IsotonicCalibrator(min_samples=10)

        # Add samples with known outcomes
        for i in range(50):
            pred = i / 50
            actual = 1.0 if i > 25 else 0.0
            calibrator.add_sample(predicted=pred, actual=actual)

        stats = calibrator.get_calibration_stats()

        assert stats["sample_count"] == 50
        assert "brier_score" in stats
        assert "calibration_slope" in stats
        assert "reliability_diagram" in stats

    def test_clear_resets_state(self):
        """Clear should reset all calibration state."""
        calibrator = IsotonicCalibrator(min_samples=10)

        for i in range(20):
            calibrator.add_sample(predicted=i / 20, actual=1.0)
        calibrator.fit()

        assert calibrator.is_fitted
        assert calibrator.sample_count > 0

        calibrator.clear()

        assert not calibrator.is_fitted
        assert calibrator.sample_count == 0
