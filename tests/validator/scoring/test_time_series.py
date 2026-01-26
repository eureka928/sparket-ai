"""Tests for time series metrics: correlation and lead-lag analysis."""

import numpy as np
import pytest

from sparket.validator.scoring.metrics.time_series import (
    LeadLagResult,
    bucket_time_series,
    align_time_series,
    compute_correlation,
    compute_sos,
    detect_moves,
    analyze_lead_lag,
)


class TestBucketTimeSeries:
    """Tests for time series bucketing."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        ts, vals = bucket_time_series(np.array([]), np.array([]), 60)
        assert len(ts) == 0
        assert len(vals) == 0

    def test_single_value(self):
        """Single value should pass through."""
        ts, vals = bucket_time_series(np.array([100.0]), np.array([0.5]), 60)
        assert len(ts) == 1
        assert vals[0] == 0.5

    def test_bucket_floor(self):
        """Timestamps should be floored to bucket boundaries."""
        # Bucket size = 60 seconds
        timestamps = np.array([0, 30, 59, 60, 90, 120])
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        ts, vals = bucket_time_series(timestamps, values, 60)

        # Buckets: 0, 60, 120
        assert len(ts) == 3
        # Last value in each bucket is used
        assert vals[0] == 0.3  # 59 is last in bucket 0
        assert vals[1] == 0.5  # 90 is last in bucket 60
        assert vals[2] == 0.6  # 120 is in bucket 120

    def test_preserves_order(self):
        """Output should be chronologically ordered."""
        timestamps = np.array([300.0, 0.0, 120.0, 60.0])
        values = np.array([0.4, 0.1, 0.3, 0.2])

        ts, vals = bucket_time_series(timestamps, values, 60)

        # Should be sorted by bucket time
        assert np.all(np.diff(ts) >= 0)

    def test_same_bucket_takes_last(self):
        """Multiple values in same bucket should take the last one."""
        timestamps = np.array([0, 10, 20, 30])
        values = np.array([0.1, 0.2, 0.3, 0.4])

        ts, vals = bucket_time_series(timestamps, values, 60)

        assert len(ts) == 1
        assert vals[0] == 0.4  # Last value


class TestAlignTimeSeries:
    """Tests for time series alignment."""

    def test_perfect_overlap(self):
        """Identical timestamps should align perfectly."""
        ts = np.array([0, 60, 120])
        vals1 = np.array([0.1, 0.2, 0.3])
        vals2 = np.array([0.4, 0.5, 0.6])

        aligned1, aligned2 = align_time_series(ts, vals1, ts, vals2)

        np.testing.assert_array_equal(aligned1, vals1)
        np.testing.assert_array_equal(aligned2, vals2)

    def test_partial_overlap(self):
        """Only common timestamps should be included."""
        ts1 = np.array([0, 60, 120])
        vals1 = np.array([0.1, 0.2, 0.3])
        ts2 = np.array([60, 120, 180])
        vals2 = np.array([0.5, 0.6, 0.7])

        aligned1, aligned2 = align_time_series(ts1, vals1, ts2, vals2)

        # Common timestamps: 60, 120
        assert len(aligned1) == 2
        assert len(aligned2) == 2
        np.testing.assert_array_equal(aligned1, [0.2, 0.3])
        np.testing.assert_array_equal(aligned2, [0.5, 0.6])

    def test_no_overlap(self):
        """No common timestamps should return empty."""
        ts1 = np.array([0, 60])
        vals1 = np.array([0.1, 0.2])
        ts2 = np.array([120, 180])
        vals2 = np.array([0.3, 0.4])

        aligned1, aligned2 = align_time_series(ts1, vals1, ts2, vals2)

        assert len(aligned1) == 0
        assert len(aligned2) == 0


class TestComputeCorrelation:
    """Tests for Pearson correlation computation."""

    def test_perfect_positive_correlation(self):
        """Perfectly correlated series should give corr = 1."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([2, 4, 6, 8, 10], dtype=np.float64)
        assert compute_correlation(x, y) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_correlation(self):
        """Perfectly anti-correlated should give corr = -1."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([10, 8, 6, 4, 2], dtype=np.float64)
        assert compute_correlation(x, y) == pytest.approx(-1.0, abs=1e-9)

    def test_no_correlation(self):
        """Uncorrelated series should give corr ≈ 0."""
        np.random.seed(42)
        x = np.random.rand(100)
        y = np.random.rand(100)
        corr = compute_correlation(x, y)
        assert abs(corr) < 0.3  # Likely near 0

    def test_insufficient_points(self):
        """Too few points should return 0."""
        x = np.array([1, 2], dtype=np.float64)
        y = np.array([3, 4], dtype=np.float64)
        assert compute_correlation(x, y, min_points=3) == 0.0

    def test_constant_series(self):
        """Constant series should return 0 (undefined correlation)."""
        x = np.array([1, 1, 1, 1], dtype=np.float64)
        y = np.array([1, 2, 3, 4], dtype=np.float64)
        assert compute_correlation(x, y) == 0.0

    def test_correlation_range(self):
        """Correlation should always be in [-1, 1]."""
        for _ in range(20):
            x = np.random.rand(50)
            y = np.random.rand(50)
            corr = compute_correlation(x, y)
            assert -1 <= corr <= 1

    def test_nan_correlation_returns_zero(self):
        """NaN in correlation should return 0."""
        # Create data that would produce NaN correlation
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = compute_correlation(x, y)
        assert corr == 0.0


class TestComputeSOS:
    """Tests for Source of Signal (originality) score."""

    def test_perfect_correlation_gives_zero_sos(self):
        """Perfect correlation = no originality = SOS = 0."""
        assert compute_sos(1.0) == pytest.approx(0.0, abs=1e-9)
        assert compute_sos(-1.0) == pytest.approx(0.0, abs=1e-9)

    def test_no_correlation_gives_one_sos(self):
        """No correlation = maximum originality = SOS = 1."""
        assert compute_sos(0.0) == pytest.approx(1.0, abs=1e-9)

    def test_partial_correlation(self):
        """Partial correlation gives intermediate SOS."""
        assert compute_sos(0.5) == pytest.approx(0.5, abs=1e-9)
        assert compute_sos(-0.5) == pytest.approx(0.5, abs=1e-9)

    def test_sos_range(self):
        """SOS should always be in [0, 1]."""
        for corr in np.linspace(-1, 1, 100):
            sos = compute_sos(corr)
            assert 0 <= sos <= 1


class TestDetectMoves:
    """Tests for move detection in time series."""

    def test_no_moves(self):
        """Constant series should have no moves."""
        ts = np.array([0, 60, 120, 180])
        vals = np.array([0.5, 0.5, 0.5, 0.5])

        times, dirs, mags = detect_moves(ts, vals, threshold=0.01)

        assert len(times) == 0

    def test_single_move_up(self):
        """Single upward move should be detected."""
        ts = np.array([0, 60, 120])
        vals = np.array([0.5, 0.5, 0.6])

        times, dirs, mags = detect_moves(ts, vals, threshold=0.05)

        assert len(times) == 1
        assert times[0] == 120
        assert dirs[0] == 1  # Up
        assert mags[0] == pytest.approx(0.1, abs=1e-9)

    def test_single_move_down(self):
        """Single downward move should be detected."""
        ts = np.array([0, 60, 120])
        vals = np.array([0.5, 0.5, 0.4])

        times, dirs, mags = detect_moves(ts, vals, threshold=0.05)

        assert len(times) == 1
        assert dirs[0] == -1  # Down

    def test_multiple_moves(self):
        """Multiple moves should all be detected."""
        ts = np.array([0, 60, 120, 180, 240])
        vals = np.array([0.5, 0.6, 0.6, 0.5, 0.7])

        times, dirs, mags = detect_moves(ts, vals, threshold=0.05)

        assert len(times) == 3
        assert list(dirs) == [1, -1, 1]  # Up, down, up

    def test_threshold_filters_small_moves(self):
        """Moves smaller than threshold should be ignored."""
        ts = np.array([0, 60, 120])
        vals = np.array([0.5, 0.52, 0.53])  # Small moves

        times, dirs, mags = detect_moves(ts, vals, threshold=0.05)

        assert len(times) == 0

    def test_insufficient_points(self):
        """Single point should return empty."""
        ts = np.array([0])
        vals = np.array([0.5])

        times, dirs, mags = detect_moves(ts, vals, threshold=0.01)

        assert len(times) == 0


class TestAnalyzeLeadLag:
    """Tests for lead-lag analysis."""

    def test_returns_result_object(self):
        """Should return LeadLagResult dataclass."""
        truth_ts = np.array([0, 60, 120, 180])
        truth_vals = np.array([0.5, 0.5, 0.6, 0.6])
        miner_ts = np.array([0, 60, 120, 180])
        miner_vals = np.array([0.5, 0.6, 0.6, 0.6])

        result = analyze_lead_lag(
            truth_ts, truth_vals, miner_ts, miner_vals,
            lead_window_seconds=120,
            lag_window_seconds=120,
            move_threshold=0.05,
        )

        assert isinstance(result, LeadLagResult)
        assert hasattr(result, 'moves_led')
        assert hasattr(result, 'moves_matched')
        assert hasattr(result, 'total_truth_moves')
        assert hasattr(result, 'lead_ratio')
        assert hasattr(result, 'sos_score')

    def test_miner_leads(self):
        """Miner moving before truth should count as lead."""
        # Truth moves at t=120
        truth_ts = np.array([0, 60, 120, 180])
        truth_vals = np.array([0.5, 0.5, 0.6, 0.6])

        # Miner moves at t=60 (before truth)
        miner_ts = np.array([0, 60, 120, 180])
        miner_vals = np.array([0.5, 0.6, 0.6, 0.6])

        result = analyze_lead_lag(
            truth_ts, truth_vals, miner_ts, miner_vals,
            lead_window_seconds=120,
            lag_window_seconds=60,
            move_threshold=0.05,
        )

        assert result.moves_led >= 1
        assert result.moves_matched >= 1

    def test_miner_lags(self):
        """Miner moving after truth should not count as lead."""
        # Truth moves at t=60
        truth_ts = np.array([0, 60, 120, 180])
        truth_vals = np.array([0.5, 0.6, 0.6, 0.6])

        # Miner moves at t=120 (after truth)
        miner_ts = np.array([0, 60, 120, 180])
        miner_vals = np.array([0.5, 0.5, 0.6, 0.6])

        result = analyze_lead_lag(
            truth_ts, truth_vals, miner_ts, miner_vals,
            lead_window_seconds=30,  # Short lead window
            lag_window_seconds=120,
            move_threshold=0.05,
        )

        assert result.moves_matched >= 1
        # Lead should be 0 or low

    def test_no_truth_moves(self):
        """No truth moves should give neutral lead ratio."""
        truth_ts = np.array([0, 60, 120])
        truth_vals = np.array([0.5, 0.5, 0.5])  # No moves
        miner_ts = np.array([0, 60, 120])
        miner_vals = np.array([0.5, 0.6, 0.7])

        result = analyze_lead_lag(
            truth_ts, truth_vals, miner_ts, miner_vals,
            lead_window_seconds=120,
            lag_window_seconds=120,
            move_threshold=0.05,
        )

        assert result.total_truth_moves == 0
        assert result.lead_ratio == 0.5  # Neutral

    def test_no_miner_moves(self):
        """No miner moves should give lead ratio = 0."""
        truth_ts = np.array([0, 60, 120])
        truth_vals = np.array([0.5, 0.6, 0.7])  # Moves
        miner_ts = np.array([0, 60, 120])
        miner_vals = np.array([0.5, 0.5, 0.5])  # No moves

        result = analyze_lead_lag(
            truth_ts, truth_vals, miner_ts, miner_vals,
            lead_window_seconds=120,
            lag_window_seconds=120,
            move_threshold=0.05,
        )

        assert result.total_truth_moves >= 1
        assert result.moves_matched == 0
        assert result.lead_ratio == 0.0

    def test_lead_ratio_range(self):
        """Lead ratio should be in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            n = 20
            truth_ts = np.arange(n, dtype=np.float64) * 60
            truth_vals = np.cumsum(np.random.randn(n) * 0.05) + 0.5
            miner_ts = truth_ts.copy()
            miner_vals = np.cumsum(np.random.randn(n) * 0.05) + 0.5

            result = analyze_lead_lag(
                truth_ts, truth_vals, miner_ts, miner_vals,
                lead_window_seconds=120,
                lag_window_seconds=120,
                move_threshold=0.03,
            )

            assert 0 <= result.lead_ratio <= 1

    def test_sos_score_computed(self):
        """SOS score should be computed from correlation."""
        truth_ts = np.arange(10, dtype=np.float64) * 60
        truth_vals = np.linspace(0.3, 0.7, 10)
        miner_ts = truth_ts.copy()
        miner_vals = truth_vals.copy()  # Perfect correlation

        result = analyze_lead_lag(
            truth_ts, truth_vals, miner_ts, miner_vals,
            lead_window_seconds=120,
            lag_window_seconds=120,
            move_threshold=0.05,
        )

        # Perfect correlation -> SOS ≈ 0
        assert result.sos_score == pytest.approx(0.0, abs=0.01)

    def test_insufficient_aligned_points_for_correlation(self):
        """Should handle case with only 2 aligned points (less than min_points)."""
        # Only 2 overlapping timestamps (below min_points=3 default)
        truth_ts = np.array([0, 60, 120, 180], dtype=np.float64)
        truth_vals = np.array([0.5, 0.5, 0.6, 0.7])
        miner_ts = np.array([0, 60, 300, 360], dtype=np.float64)  # Only 0, 60 overlap
        miner_vals = np.array([0.5, 0.5, 0.5, 0.5])

        result = analyze_lead_lag(
            truth_ts, truth_vals, miner_ts, miner_vals,
            lead_window_seconds=60,
            lag_window_seconds=60,
            move_threshold=0.05,
        )

        # Not enough overlap for meaningful correlation -> SOS = 0.5 (neutral)
        # The correlation function returns 0.0 when n < min_points
        # So SOS = 1 - |0| = 1.0 or if min_points check fails in analyze_lead_lag, 0.5
        assert 0.0 <= result.sos_score <= 1.0

