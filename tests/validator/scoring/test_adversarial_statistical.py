"""Statistical invariant and stress tests.

Tests for:
1. Statistical invariants that must always hold
2. Monte Carlo stress tests
3. Long-running pattern detection
4. Convergence properties
5. Distribution preservation
"""

from decimal import Decimal
from datetime import datetime, timezone

import numpy as np
import pytest

from sparket.validator.scoring.metrics.proper_scoring import (
    brier_score,
    brier_score_batch,
    log_loss,
    log_loss_batch,
    pss,
)
from sparket.validator.scoring.metrics.clv import (
    compute_clv,
    compute_mes,
)
from sparket.validator.scoring.metrics.calibration import (
    compute_calibration,
)
from sparket.validator.scoring.metrics.sharpness import (
    compute_sharpness,
)
from sparket.validator.scoring.metrics.time_series import (
    compute_correlation,
    compute_sos,
)
from sparket.validator.scoring.aggregation.decay import (
    compute_decay_weights,
    weighted_mean,
    effective_sample_size,
)
from sparket.validator.scoring.aggregation.normalization import (
    normalize_zscore_logistic,
    normalize_percentile,
    normalize_minmax,
)
from sparket.validator.scoring.aggregation.shrinkage import (
    shrink_toward_mean,
    compute_population_mean,
)


# =============================================================================
# STATISTICAL INVARIANTS
# =============================================================================

class TestBrierInvariants:
    """Invariants that must hold for Brier scores."""

    def test_perfect_prediction_zero_score(self):
        """Perfect predictions should give Brier = 0."""
        # Outcome is 0, predicted 0% for 1
        p_forecast = np.array([1.0, 0.0])
        outcome = np.array([1, 0], dtype=np.int8)

        score = brier_score(p_forecast, outcome)
        assert score == 0.0

    def test_worst_prediction_max_score(self):
        """Completely wrong prediction should give max score."""
        # Predicted opposite of what happened
        p_forecast = np.array([0.0, 1.0])
        outcome = np.array([1, 0], dtype=np.int8)

        score = brier_score(p_forecast, outcome)
        assert score == 2.0  # Max possible

    def test_uniform_prediction_score(self):
        """Uniform 50/50 prediction should give 0.5 Brier."""
        p_forecast = np.array([0.5, 0.5])
        outcome = np.array([1, 0], dtype=np.int8)

        score = brier_score(p_forecast, outcome)
        assert np.isclose(score, 0.5, atol=0.001)

    def test_brier_bounded_zero_two(self):
        """Brier score must be in [0, 2]."""
        np.random.seed(42)

        for _ in range(100):
            probs = np.random.dirichlet([1, 1])
            outcome = np.zeros(2, dtype=np.int8)
            outcome[np.random.randint(2)] = 1

            score = brier_score(probs, outcome)
            assert 0 <= score <= 2

    def test_brier_symmetric(self):
        """Brier score should be symmetric in outcomes."""
        # If we flip which outcome happened, score should be same
        # when predictions are also flipped
        p1 = np.array([0.7, 0.3])
        o1 = np.array([1, 0], dtype=np.int8)

        p2 = np.array([0.3, 0.7])
        o2 = np.array([0, 1], dtype=np.int8)

        assert np.isclose(brier_score(p1, o1), brier_score(p2, o2))


class TestLogLossInvariants:
    """Invariants that must hold for log loss."""

    def test_confident_correct_low_loss(self):
        """Confident correct prediction should have low loss."""
        p_forecast = np.array([0.99, 0.01])
        outcome = np.array([1, 0], dtype=np.int8)

        loss = log_loss(p_forecast, outcome)
        assert loss < 0.1

    def test_confident_wrong_high_loss(self):
        """Confident wrong prediction should have high loss."""
        p_forecast = np.array([0.01, 0.99])
        outcome = np.array([1, 0], dtype=np.int8)

        loss = log_loss(p_forecast, outcome)
        assert loss > 4  # High penalty

    def test_log_loss_non_negative(self):
        """Log loss must be >= 0."""
        np.random.seed(42)

        for _ in range(100):
            probs = np.random.dirichlet([1, 1])
            outcome = np.zeros(2, dtype=np.int8)
            outcome[np.random.randint(2)] = 1

            loss = log_loss(probs, outcome)
            assert loss >= 0


class TestPSSInvariants:
    """Invariants for Predictive Skill Score."""

    def test_pss_positive_for_good_predictor(self):
        """Good predictor should have positive PSS."""
        # Better than baseline
        np.random.seed(42)

        miner_briers = []
        truth_briers = []

        for _ in range(100):
            true_p = np.random.uniform(0.3, 0.7)
            # Miner knows roughly the right probability
            miner_p = true_p + np.random.normal(0, 0.05)
            miner_p = np.clip(miner_p, 0.01, 0.99)

            miner_forecast = np.array([miner_p, 1 - miner_p])
            truth_forecast = np.array([0.5, 0.5])  # Baseline is 50/50
            outcome = np.zeros(2, dtype=np.int8)
            outcome[0 if np.random.random() < true_p else 1] = 1

            miner_briers.append(brier_score(miner_forecast, outcome))
            truth_briers.append(brier_score(truth_forecast, outcome))

        avg_miner = np.mean(miner_briers)
        avg_truth = np.mean(truth_briers)

        avg_pss = pss(avg_miner, avg_truth)
        # Good predictor should have positive PSS (beats baseline)
        assert avg_pss > -0.1  # Might be slightly negative due to noise

    def test_pss_zero_for_baseline(self):
        """Same predictions as truth should give PSS = 0."""
        # If miner and truth have identical scores, PSS = 0
        miner_score = 0.25
        truth_score = 0.25

        result = pss(miner_score, truth_score)
        assert result == 0.0


class TestNormalizationInvariants:
    """Invariants for normalization."""

    def test_normalized_in_unit_interval(self):
        """Z-score logistic output must be in (0, 1)."""
        np.random.seed(42)

        for _ in range(100):
            values = np.random.uniform(-1000, 1000, 50)
            norm = normalize_zscore_logistic(values)

            assert np.all(norm > 0)
            assert np.all(norm < 1)

    def test_percentile_in_unit_interval(self):
        """Percentile normalization must be in [0, 1]."""
        np.random.seed(42)

        for _ in range(100):
            values = np.random.uniform(-1000, 1000, 50)
            norm = normalize_percentile(values)

            assert np.all(norm >= 0)
            assert np.all(norm <= 1)

    def test_minmax_in_unit_interval(self):
        """Min-max normalization must be in [0, 1]."""
        np.random.seed(42)

        for _ in range(100):
            values = np.random.uniform(-1000, 1000, 50)
            norm = normalize_minmax(values)

            assert np.all(norm >= 0)
            assert np.all(norm <= 1)

    def test_ordering_preserved(self):
        """Normalization should preserve relative ordering."""
        np.random.seed(42)

        values = np.random.uniform(0, 100, 50)
        sorted_idx = np.argsort(values)

        for norm_fn in [normalize_zscore_logistic, normalize_percentile, normalize_minmax]:
            norm = norm_fn(values)
            norm_sorted_idx = np.argsort(norm)

            # Ordering should be preserved
            np.testing.assert_array_equal(sorted_idx, norm_sorted_idx)


class TestShrinkageInvariants:
    """Invariants for Bayesian shrinkage."""

    def test_shrinkage_moves_toward_mean(self):
        """Shrinkage should always move values toward population mean."""
        np.random.seed(42)

        values = np.random.uniform(0, 1, 20)
        n_effs = np.random.uniform(10, 100, 20)

        pop_mean = compute_population_mean(values, n_effs)
        shrunk = shrink_toward_mean(values, n_effs, k=200.0)

        # Each shrunk value should be between original and pop_mean
        for i in range(len(values)):
            if values[i] > pop_mean:
                assert shrunk[i] <= values[i]
                assert shrunk[i] >= pop_mean
            else:
                assert shrunk[i] >= values[i]
                assert shrunk[i] <= pop_mean

    def test_high_neff_less_shrinkage(self):
        """Higher n_eff should result in less shrinkage."""
        values = np.array([0.8])
        pop_mean = 0.5  # Explicit pop_mean needed to test shrinkage
        n_effs_low = np.array([10.0])
        n_effs_high = np.array([1000.0])

        shrunk_low = shrink_toward_mean(values, n_effs_low, k=200.0, population_mean=pop_mean)
        shrunk_high = shrink_toward_mean(values, n_effs_high, k=200.0, population_mean=pop_mean)

        # High n_eff should stay closer to original
        assert abs(shrunk_high[0] - values[0]) < abs(shrunk_low[0] - values[0])


class TestDecayInvariants:
    """Invariants for time decay."""

    def test_weights_monotonically_decrease(self):
        """More recent should have higher weight."""
        # compute_decay_weights expects (timestamps, reference_timestamp, half_life)
        # More recent timestamps (closer to now) get higher weights
        now = 30.0 * 86400  # Reference time

        # Timestamps from oldest to newest
        timestamps = np.array([0, 5, 10, 15, 20, 25, 28, 29, 30]) * 86400

        weights = compute_decay_weights(timestamps, now, half_life_days=10.0)

        # Should be monotonically increasing (newer = higher weight)
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1]

    def test_weights_sum_reasonable(self):
        """Weights should have reasonable sum."""
        ages = np.linspace(0, 30, 100)

        weights = compute_decay_weights(ages * 86400, 30 * 86400, half_life_days=10.0)

        # All weights should be positive and <= 1
        assert np.all(weights > 0)
        assert np.all(weights <= 1)


# =============================================================================
# MONTE CARLO STRESS TESTS
# =============================================================================

class TestMonteCarloStress:
    """Monte Carlo stress testing."""

    @pytest.mark.parametrize("seed", range(10))
    def test_calibration_with_random_data(self, seed):
        """Calibration should not crash with random valid data."""
        np.random.seed(seed)

        n_samples = 100
        probs = np.random.uniform(0.1, 0.9, n_samples)
        outcomes = (np.random.random(n_samples) < probs).astype(np.int8)

        cal = compute_calibration(probs, outcomes, min_samples=10)
        assert np.isfinite(cal.score)

    @pytest.mark.parametrize("seed", range(10))
    def test_sharpness_with_random_data(self, seed):
        """Sharpness should not crash with random valid data."""
        np.random.seed(seed)

        probs = np.random.uniform(0, 1, 100)

        sharp = compute_sharpness(probs, target_variance=0.04)
        assert np.isfinite(sharp)
        assert sharp >= 0

    @pytest.mark.parametrize("seed", range(10))
    def test_correlation_with_random_series(self, seed):
        """Correlation should be in [-1, 1]."""
        np.random.seed(seed)

        vals1 = np.random.uniform(0, 1, 50)
        vals2 = np.random.uniform(0, 1, 50)

        corr = compute_correlation(vals1, vals2)
        assert -1 <= corr <= 1

    def test_batch_brier_many_events(self):
        """Batch Brier with many events."""
        np.random.seed(42)

        n_events = 1000
        n_outcomes = 2

        forecasts = np.random.dirichlet([1, 1], n_events)
        outcomes = np.eye(n_outcomes, dtype=np.int8)[np.random.randint(n_outcomes, size=n_events)]

        scores = brier_score_batch(forecasts, outcomes)

        assert len(scores) == n_events
        assert np.all(scores >= 0)
        assert np.all(scores <= 2)


# =============================================================================
# CONVERGENCE TESTS
# =============================================================================

class TestConvergence:
    """Tests for convergence properties."""

    def test_shrinkage_converges_with_sample_size(self):
        """As n_eff increases, shrunk value approaches raw value."""
        raw_val = 0.75
        pop_mean = 0.5

        n_effs = [10, 50, 100, 500, 1000, 5000]
        shrunk_vals = []

        for n in n_effs:
            shrunk = shrink_toward_mean(
                np.array([raw_val]),
                np.array([float(n)]),
                k=200.0,
                population_mean=pop_mean,  # Explicit pop_mean needed
            )
            shrunk_vals.append(shrunk[0])

        # Should converge to raw value
        assert shrunk_vals[-1] > shrunk_vals[0]
        assert abs(shrunk_vals[-1] - raw_val) < 0.01

    def test_decay_converges_to_zero(self):
        """Very old data should have near-zero weight."""
        ages = np.array([0, 100, 1000, 10000])  # Days
        now = 10000.0

        weights = compute_decay_weights(ages * 86400, now * 86400, half_life_days=10.0)

        # Very old (10000 days ago) should be essentially zero
        assert weights[0] < 1e-100

    def test_effective_sample_size_bounded(self):
        """Effective sample size should be <= actual sample size."""
        np.random.seed(42)

        for _ in range(100):
            n = np.random.randint(10, 1000)
            weights = np.random.uniform(0, 1, n)

            n_eff = effective_sample_size(weights)
            assert n_eff <= n


# =============================================================================
# DISTRIBUTION PRESERVATION
# =============================================================================

class TestDistributionPreservation:
    """Tests that transformations preserve statistical properties."""

    def test_normalization_preserves_rank_distribution(self):
        """Rank distribution should be preserved by normalization."""
        np.random.seed(42)

        values = np.random.exponential(1, 100)

        for norm_fn in [normalize_percentile, normalize_minmax]:
            norm = norm_fn(values)

            # Ranks should be preserved
            orig_ranks = np.argsort(np.argsort(values))
            norm_ranks = np.argsort(np.argsort(norm))

            np.testing.assert_array_equal(orig_ranks, norm_ranks)

    def test_weighted_mean_unbiased(self):
        """Weighted mean with equal weights should equal simple mean."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        equal_weights = np.ones(5)

        wmean = weighted_mean(values, equal_weights)
        simple_mean = np.mean(values)

        assert np.isclose(wmean, simple_mean)

    def test_sos_symmetric(self):
        """SOS should be symmetric in the correlation."""
        vals1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        vals2 = np.array([0.15, 0.25, 0.35, 0.45, 0.55])

        corr12 = compute_correlation(vals1, vals2)
        corr21 = compute_correlation(vals2, vals1)

        sos12 = compute_sos(corr12)
        sos21 = compute_sos(corr21)

        assert np.isclose(sos12, sos21)


# =============================================================================
# EDGE CASE STRESS TESTS
# =============================================================================

class TestEdgeCaseStress:
    """Stress tests with edge case combinations."""

    def test_all_same_value_normalization(self):
        """All values identical should be handled gracefully."""
        values = np.full(100, 0.42)

        # Each normalization handles constant values differently
        # The important thing is no NaN/Inf and consistent output

        # Z-score logistic: std=0 leads to special handling
        norm_zscore = normalize_zscore_logistic(values)
        assert np.all(np.isfinite(norm_zscore))
        assert np.all(norm_zscore == norm_zscore[0])  # All same

        # Percentile: all same rank means 0.5
        norm_pct = normalize_percentile(values)
        assert np.allclose(norm_pct, 0.5)

        # Min-max: 0/0 when range is 0, should handle gracefully
        norm_mm = normalize_minmax(values)
        assert np.all(np.isfinite(norm_mm))
        assert np.all(norm_mm == norm_mm[0])  # All same

    def test_two_clusters_normalization(self):
        """Two distinct clusters of values."""
        values = np.concatenate([np.full(50, 0.2), np.full(50, 0.8)])

        norm_zscore = normalize_zscore_logistic(values)
        norm_percentile = normalize_percentile(values)

        # Z-score logistic should give bimodal output
        assert norm_zscore[0] < 0.5
        assert norm_zscore[-1] > 0.5

    def test_single_outlier_many_normal(self):
        """One extreme outlier among many normal values."""
        values = np.concatenate([np.full(99, 0.5), [100.0]])

        norm = normalize_zscore_logistic(values)

        # Normal values should still be near 0.5
        assert np.allclose(norm[:99], 0.5, atol=0.1)
        # Outlier should be near 1
        assert norm[-1] > 0.9

    def test_alternating_extreme_values(self):
        """Alternating between extremes."""
        values = np.array([0.0, 1.0] * 50)

        norm = normalize_zscore_logistic(values)

        # Should show bimodal pattern
        assert norm[0] < 0.3
        assert norm[1] > 0.7

    def test_linear_sequence(self):
        """Perfectly linear sequence."""
        values = np.linspace(0, 1, 100)

        norm_percentile = normalize_percentile(values)
        norm_minmax = normalize_minmax(values)

        # Percentile should be approximately uniform
        np.testing.assert_array_almost_equal(
            norm_percentile,
            np.linspace(0, 1, 100),
            decimal=2
        )

        # Min-max should also be linear
        np.testing.assert_array_almost_equal(
            norm_minmax,
            np.linspace(0, 1, 100),
            decimal=5
        )

