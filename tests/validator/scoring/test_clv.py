"""Tests for CLV (Closing Line Value) and CLE (Closing Line Edge) calculations."""

import numpy as np
import pytest

from sparket.validator.scoring.metrics.clv import (
    CLVResult,
    compute_clv,
    compute_clv_batch,
    compute_mes,
    compute_mes_batch,
)


class TestComputeCLV:
    """Tests for single CLV/CLE computation."""

    def test_returns_result_object(self):
        """Should return CLVResult dataclass."""
        result = compute_clv(
            miner_odds=2.0,
            miner_prob=0.5,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=0.0,
            event_start_ts=3600.0,
        )
        assert isinstance(result, CLVResult)

    def test_equal_odds_clv_zero(self):
        """Equal miner and truth odds should give CLV = 0."""
        result = compute_clv(
            miner_odds=2.0,
            miner_prob=0.5,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=0.0,
            event_start_ts=3600.0,
        )
        assert result.clv_odds == pytest.approx(0.0, abs=1e-9)
        assert result.clv_prob == pytest.approx(0.0, abs=1e-9)

    def test_better_odds_positive_clv(self):
        """Miner offering better odds should give positive CLV_odds."""
        # Miner: 2.2, Truth: 2.0
        # CLV_odds = (2.2 - 2.0) / 2.0 = 0.1
        result = compute_clv(
            miner_odds=2.2,
            miner_prob=1 / 2.2,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=0.0,
            event_start_ts=3600.0,
        )
        assert result.clv_odds == pytest.approx(0.1, abs=1e-9)

    def test_worse_odds_negative_clv(self):
        """Miner offering worse odds should give negative CLV_odds."""
        # Miner: 1.8, Truth: 2.0
        # CLV_odds = (1.8 - 2.0) / 2.0 = -0.1
        result = compute_clv(
            miner_odds=1.8,
            miner_prob=1 / 1.8,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=0.0,
            event_start_ts=3600.0,
        )
        assert result.clv_odds == pytest.approx(-0.1, abs=1e-9)

    def test_cle_positive_edge(self):
        """Positive CLE means profitable bet at miner's odds."""
        # Miner odds: 2.2, Truth prob: 0.5
        # CLE = 2.2 * 0.5 - 1 = 1.1 - 1 = 0.1
        result = compute_clv(
            miner_odds=2.2,
            miner_prob=1 / 2.2,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=0.0,
            event_start_ts=3600.0,
        )
        assert result.cle == pytest.approx(0.1, abs=1e-9)

    def test_cle_negative_edge(self):
        """Negative CLE means unprofitable bet."""
        # Miner odds: 1.8, Truth prob: 0.5
        # CLE = 1.8 * 0.5 - 1 = 0.9 - 1 = -0.1
        result = compute_clv(
            miner_odds=1.8,
            miner_prob=1 / 1.8,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=0.0,
            event_start_ts=3600.0,
        )
        assert result.cle == pytest.approx(-0.1, abs=1e-9)

    def test_cle_clamping(self):
        """CLE should be clamped to [cle_min, cle_max]."""
        # Very high odds would give CLE > 1
        result = compute_clv(
            miner_odds=10.0,
            miner_prob=0.1,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=0.0,
            event_start_ts=3600.0,
            cle_min=-1.0,
            cle_max=1.0,
        )
        # CLE = 10 * 0.5 - 1 = 4, but clamped to 1
        assert result.cle == pytest.approx(1.0, abs=1e-9)

    def test_minutes_to_close(self):
        """Minutes to close should be computed correctly."""
        result = compute_clv(
            miner_odds=2.0,
            miner_prob=0.5,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=0.0,
            event_start_ts=7200.0,  # 2 hours = 120 minutes
        )
        assert result.minutes_to_close == 120

    def test_minutes_to_close_negative_clamped(self):
        """Negative time difference should be clamped to 0."""
        result = compute_clv(
            miner_odds=2.0,
            miner_prob=0.5,
            truth_odds=2.0,
            truth_prob=0.5,
            submitted_ts=1000.0,
            event_start_ts=500.0,  # Event already started
        )
        assert result.minutes_to_close == 0

    def test_zero_truth_odds(self):
        """Zero or invalid truth odds should give CLV = 0."""
        result = compute_clv(
            miner_odds=2.0,
            miner_prob=0.5,
            truth_odds=1.0,  # Invalid odds (≤1)
            truth_prob=0.5,
            submitted_ts=0.0,
            event_start_ts=3600.0,
        )
        assert result.clv_odds == 0.0

    def test_zero_truth_prob(self):
        """Zero truth prob should give CLV_prob = 0."""
        result = compute_clv(
            miner_odds=2.0,
            miner_prob=0.5,
            truth_odds=2.0,
            truth_prob=0.0,
            submitted_ts=0.0,
            event_start_ts=3600.0,
        )
        assert result.clv_prob == 0.0


class TestComputeCLVBatch:
    """Tests for batch CLV/CLE computation."""

    def test_batch_matches_individual(self):
        """Batch results should match individual calculations."""
        miner_odds = np.array([2.0, 2.2, 1.8])
        miner_probs = 1.0 / miner_odds
        truth_odds = np.array([2.0, 2.0, 2.0])
        truth_probs = np.array([0.5, 0.5, 0.5])
        submitted = np.array([0.0, 0.0, 0.0])
        event_start = np.array([3600.0, 3600.0, 3600.0])

        clv_o, clv_p, cle, mins = compute_clv_batch(
            miner_odds, miner_probs, truth_odds, truth_probs,
            submitted, event_start
        )

        for i in range(len(miner_odds)):
            individual = compute_clv(
                miner_odds[i], miner_probs[i], truth_odds[i], truth_probs[i],
                submitted[i], event_start[i]
            )
            assert clv_o[i] == pytest.approx(individual.clv_odds, abs=1e-9)
            assert clv_p[i] == pytest.approx(individual.clv_prob, abs=1e-9)
            assert cle[i] == pytest.approx(individual.cle, abs=1e-9)
            assert mins[i] == individual.minutes_to_close

    def test_batch_shape(self):
        """Batch outputs should have correct shapes."""
        N = 100
        miner_odds = np.random.uniform(1.5, 3.0, N)
        miner_probs = 1.0 / miner_odds
        truth_odds = np.full(N, 2.0)
        truth_probs = np.full(N, 0.5)
        submitted = np.zeros(N)
        event_start = np.full(N, 3600.0)

        clv_o, clv_p, cle, mins = compute_clv_batch(
            miner_odds, miner_probs, truth_odds, truth_probs,
            submitted, event_start
        )

        assert clv_o.shape == (N,)
        assert clv_p.shape == (N,)
        assert cle.shape == (N,)
        assert mins.shape == (N,)


class TestComputeMES:
    """Tests for Market Efficiency Score."""

    def test_perfect_efficiency(self):
        """CLV = 0 should give MES = 1."""
        assert compute_mes(0.0) == pytest.approx(1.0, abs=1e-9)

    def test_slight_deviation(self):
        """Small CLV should give high MES."""
        # MES = 1 - |0.05| = 0.95
        assert compute_mes(0.05) == pytest.approx(0.95, abs=1e-9)
        assert compute_mes(-0.05) == pytest.approx(0.95, abs=1e-9)

    def test_large_deviation(self):
        """Large CLV should give low MES."""
        # MES = 1 - |0.5| = 0.5
        assert compute_mes(0.5) == pytest.approx(0.5, abs=1e-9)

    def test_max_deviation_clamped(self):
        """CLV > 1 should still give MES = 0 (clamped)."""
        # MES = 1 - min(1, |1.5|) = 1 - 1 = 0
        assert compute_mes(1.5) == pytest.approx(0.0, abs=1e-9)
        assert compute_mes(-1.5) == pytest.approx(0.0, abs=1e-9)


class TestComputeMESBatch:
    """Tests for batch MES computation."""

    def test_batch_matches_individual(self):
        """Batch results should match individual calculations."""
        clv_probs = np.array([0.0, 0.05, -0.05, 0.5, 1.5])

        batch = compute_mes_batch(clv_probs)

        for i in range(len(clv_probs)):
            individual = compute_mes(clv_probs[i])
            assert batch[i] == pytest.approx(individual, abs=1e-9)

    def test_batch_range(self):
        """Batch MES should always be in [0, 1]."""
        clv_probs = np.random.uniform(-2, 2, 100)
        mes = compute_mes_batch(clv_probs)
        assert np.all(mes >= 0) and np.all(mes <= 1)

