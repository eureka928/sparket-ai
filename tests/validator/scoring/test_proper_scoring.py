"""Tests for proper scoring rules: Brier score, log-loss, PSS."""

import numpy as np
import pytest

from sparket.validator.scoring.metrics.proper_scoring import (
    brier_score,
    brier_score_batch,
    log_loss,
    log_loss_batch,
    pss,
    pss_batch,
    compute_proper_scoring,
    compute_proper_scoring_batch,
    outcome_to_vector,
    ProperScoringResult,
)


class TestBrierScore:
    """Tests for Brier score calculation."""

    def test_perfect_prediction(self):
        """Perfect prediction should give Brier = 0."""
        probs = np.array([1.0, 0.0])
        outcome = np.array([1, 0], dtype=np.int8)
        assert brier_score(probs, outcome) == pytest.approx(0.0, abs=1e-9)

    def test_worst_prediction(self):
        """Completely wrong prediction should give Brier = 2."""
        probs = np.array([0.0, 1.0])
        outcome = np.array([1, 0], dtype=np.int8)
        assert brier_score(probs, outcome) == pytest.approx(2.0, abs=1e-9)

    def test_even_prediction(self):
        """50-50 prediction should give Brier = 0.5 for binary."""
        probs = np.array([0.5, 0.5])
        outcome = np.array([1, 0], dtype=np.int8)
        # (0.5 - 1)^2 + (0.5 - 0)^2 = 0.25 + 0.25 = 0.5
        assert brier_score(probs, outcome) == pytest.approx(0.5, abs=1e-9)

    def test_typical_prediction(self):
        """Test a typical prediction scenario."""
        probs = np.array([0.7, 0.3])
        outcome = np.array([1, 0], dtype=np.int8)
        # (0.7 - 1)^2 + (0.3 - 0)^2 = 0.09 + 0.09 = 0.18
        assert brier_score(probs, outcome) == pytest.approx(0.18, abs=1e-9)

    def test_three_way_outcome(self):
        """Test 3-way outcome (e.g., home/draw/away)."""
        probs = np.array([0.5, 0.3, 0.2])
        outcome = np.array([0, 1, 0], dtype=np.int8)  # draw wins
        # (0.5 - 0)^2 + (0.3 - 1)^2 + (0.2 - 0)^2 = 0.25 + 0.49 + 0.04 = 0.78
        assert brier_score(probs, outcome) == pytest.approx(0.78, abs=1e-9)

    def test_normalization(self):
        """Probabilities should be normalized before scoring."""
        probs = np.array([1.4, 0.6])  # Sum = 2.0, should normalize
        outcome = np.array([1, 0], dtype=np.int8)
        # After normalization: [0.7, 0.3]
        # (0.7 - 1)^2 + (0.3 - 0)^2 = 0.18
        assert brier_score(probs, outcome) == pytest.approx(0.18, abs=1e-9)

    def test_zero_sum_gets_penalty(self):
        """Zero sum probabilities should get worst score (not crash)."""
        probs = np.array([0.0, 0.0])
        outcome = np.array([1, 0], dtype=np.int8)
        # Zero sum gets worst possible Brier score
        result = brier_score(probs, outcome)
        assert result == 2.0


class TestBrierScoreBatch:
    """Tests for batch Brier score calculation."""

    def test_batch_matches_individual(self):
        """Batch results should match individual calculations."""
        forecasts = np.array([
            [0.7, 0.3],
            [0.5, 0.5],
            [0.9, 0.1],
        ])
        outcomes = np.array([
            [1, 0],
            [1, 0],
            [0, 1],
        ], dtype=np.int8)

        batch = brier_score_batch(forecasts, outcomes)

        for i in range(len(forecasts)):
            individual = brier_score(forecasts[i], outcomes[i])
            assert batch[i] == pytest.approx(individual, abs=1e-9)

    def test_batch_shape(self):
        """Batch output should have correct shape."""
        N, K = 100, 4
        forecasts = np.random.dirichlet(np.ones(K), size=N)
        outcomes = np.eye(K, dtype=np.int8)[np.random.randint(0, K, N)]

        batch = brier_score_batch(forecasts, outcomes)
        assert batch.shape == (N,)

    def test_large_batch_performance(self):
        """Large batch should complete quickly (smoke test)."""
        N, K = 10000, 3
        forecasts = np.random.dirichlet(np.ones(K), size=N)
        outcomes = np.eye(K, dtype=np.int8)[np.random.randint(0, K, N)]

        batch = brier_score_batch(forecasts, outcomes)
        assert batch.shape == (N,)
        assert np.all(batch >= 0) and np.all(batch <= 2)


class TestLogLoss:
    """Tests for log loss calculation."""

    def test_perfect_prediction(self):
        """Perfect prediction should give log loss ≈ 0."""
        probs = np.array([0.999999, 0.000001])
        outcome = np.array([1, 0], dtype=np.int8)
        assert log_loss(probs, outcome) == pytest.approx(0.0, abs=1e-4)

    def test_worst_prediction(self):
        """Very wrong prediction should give high log loss."""
        probs = np.array([0.001, 0.999])
        outcome = np.array([1, 0], dtype=np.int8)
        # -log(0.001) ≈ 6.9
        assert log_loss(probs, outcome) > 5.0

    def test_even_prediction(self):
        """50-50 prediction should give log loss = log(2) ≈ 0.693."""
        probs = np.array([0.5, 0.5])
        outcome = np.array([1, 0], dtype=np.int8)
        assert log_loss(probs, outcome) == pytest.approx(np.log(2), abs=1e-6)

    def test_typical_prediction(self):
        """Test typical prediction scenario."""
        probs = np.array([0.7, 0.3])
        outcome = np.array([1, 0], dtype=np.int8)
        # -log(0.7) ≈ 0.357
        assert log_loss(probs, outcome) == pytest.approx(-np.log(0.7), abs=1e-6)

    def test_three_way(self):
        """Test 3-way outcome."""
        probs = np.array([0.5, 0.3, 0.2])
        outcome = np.array([0, 1, 0], dtype=np.int8)
        # -log(0.3) ≈ 1.204
        assert log_loss(probs, outcome) == pytest.approx(-np.log(0.3), abs=1e-6)


class TestLogLossBatch:
    """Tests for batch log loss calculation."""

    def test_batch_matches_individual(self):
        """Batch results should match individual calculations."""
        forecasts = np.array([
            [0.7, 0.3],
            [0.5, 0.5],
            [0.9, 0.1],
        ])
        outcomes = np.array([
            [1, 0],
            [1, 0],
            [0, 1],
        ], dtype=np.int8)

        batch = log_loss_batch(forecasts, outcomes)

        for i in range(len(forecasts)):
            individual = log_loss(forecasts[i], outcomes[i])
            assert batch[i] == pytest.approx(individual, abs=1e-9)

    def test_batch_shape(self):
        """Batch output should have correct shape."""
        N, K = 100, 4
        forecasts = np.random.dirichlet(np.ones(K), size=N)
        outcomes = np.eye(K, dtype=np.int8)[np.random.randint(0, K, N)]

        batch = log_loss_batch(forecasts, outcomes)
        assert batch.shape == (N,)


class TestPSS:
    """Tests for Probability Skill Score."""

    def test_equal_scores(self):
        """Equal miner and truth scores should give PSS = 0."""
        assert pss(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)

    def test_better_than_truth(self):
        """Lower miner score should give positive PSS."""
        # PSS = 1 - (0.3 / 0.5) = 1 - 0.6 = 0.4
        assert pss(0.3, 0.5) == pytest.approx(0.4, abs=1e-9)

    def test_worse_than_truth(self):
        """Higher miner score should give negative PSS."""
        # PSS = 1 - (0.7 / 0.5) = 1 - 1.4 = -0.4
        assert pss(0.7, 0.5) == pytest.approx(-0.4, abs=1e-9)

    def test_truth_zero(self):
        """Zero truth score should return 0 (edge case)."""
        assert pss(0.5, 0.0) == 0.0

    def test_perfect_miner(self):
        """Perfect miner (score=0) should give PSS = 1."""
        assert pss(0.0, 0.5) == pytest.approx(1.0, abs=1e-9)


class TestPSSBatch:
    """Tests for batch PSS calculation."""

    def test_batch_matches_individual(self):
        """Batch results should match individual calculations."""
        miner = np.array([0.3, 0.5, 0.7])
        truth = np.array([0.5, 0.5, 0.5])

        batch = pss_batch(miner, truth)

        for i in range(len(miner)):
            individual = pss(miner[i], truth[i])
            assert batch[i] == pytest.approx(individual, abs=1e-9)

    def test_handles_zero_truth(self):
        """Batch should handle zero truth values."""
        miner = np.array([0.3, 0.5])
        truth = np.array([0.0, 0.5])

        batch = pss_batch(miner, truth)
        assert batch[0] == 0.0  # Zero truth
        assert batch[1] == pytest.approx(0.0, abs=1e-9)  # Equal


class TestComputeProperScoring:
    """Tests for combined proper scoring computation."""

    def test_returns_result_object(self):
        """Should return ProperScoringResult dataclass."""
        miner = np.array([0.7, 0.3])
        truth = np.array([0.6, 0.4])
        outcome = np.array([1, 0], dtype=np.int8)

        result = compute_proper_scoring(miner, truth, outcome)

        assert isinstance(result, ProperScoringResult)
        assert hasattr(result, 'brier_miner')
        assert hasattr(result, 'brier_truth')
        assert hasattr(result, 'logloss_miner')
        assert hasattr(result, 'logloss_truth')
        assert hasattr(result, 'pss_brier')
        assert hasattr(result, 'pss_log')

    def test_consistent_results(self):
        """Results should be internally consistent."""
        miner = np.array([0.7, 0.3])
        truth = np.array([0.6, 0.4])
        outcome = np.array([1, 0], dtype=np.int8)

        result = compute_proper_scoring(miner, truth, outcome)

        # PSS should be computed from Brier scores
        expected_pss_brier = 1.0 - (result.brier_miner / result.brier_truth)
        assert result.pss_brier == pytest.approx(expected_pss_brier, abs=1e-9)


class TestComputeProperScoringBatch:
    """Tests for batch proper scoring computation."""

    def test_batch_shape(self):
        """Batch outputs should have correct shapes."""
        N = 50
        miner = np.random.dirichlet([1, 1], size=N)
        truth = np.random.dirichlet([1, 1], size=N)
        outcomes = np.eye(2, dtype=np.int8)[np.random.randint(0, 2, N)]

        brier, logloss, pss_b, pss_l = compute_proper_scoring_batch(
            miner, truth, outcomes
        )

        assert brier.shape == (N,)
        assert logloss.shape == (N,)
        assert pss_b.shape == (N,)
        assert pss_l.shape == (N,)


class TestOutcomeToVector:
    """Tests for outcome to one-hot vector conversion."""

    def test_first_outcome(self):
        """First outcome should give [1, 0, ...]."""
        vec = outcome_to_vector(0, 3)
        np.testing.assert_array_equal(vec, [1, 0, 0])

    def test_middle_outcome(self):
        """Middle outcome should work correctly."""
        vec = outcome_to_vector(1, 3)
        np.testing.assert_array_equal(vec, [0, 1, 0])

    def test_last_outcome(self):
        """Last outcome should give [..., 0, 1]."""
        vec = outcome_to_vector(2, 3)
        np.testing.assert_array_equal(vec, [0, 0, 1])

    def test_dtype(self):
        """Output should be int8."""
        vec = outcome_to_vector(0, 2)
        assert vec.dtype == np.int8


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_probabilities(self):
        """Very small probabilities should be handled."""
        probs = np.array([1e-10, 1.0 - 1e-10])
        outcome = np.array([1, 0], dtype=np.int8)

        # Should not raise, should clamp internally
        ll = log_loss(probs, outcome)
        assert np.isfinite(ll)
        assert ll > 0

    def test_probabilities_near_one(self):
        """Probabilities near 1 should be handled."""
        probs = np.array([1.0 - 1e-15, 1e-15])
        outcome = np.array([1, 0], dtype=np.int8)

        brier = brier_score(probs, outcome)
        ll = log_loss(probs, outcome)

        assert np.isfinite(brier)
        assert np.isfinite(ll)

    def test_nan_handling(self):
        """NaN in input should get worst score (not crash)."""
        probs = np.array([np.nan, 0.5])
        outcome = np.array([1, 0], dtype=np.int8)

        # NaN gets worst possible Brier score (2.0)
        brier = brier_score(probs, outcome)
        assert brier == 2.0

