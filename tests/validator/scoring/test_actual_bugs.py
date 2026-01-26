"""Tests that expose actual bugs and inconsistencies in scoring code.

These tests should FAIL if the code has the bug, PASS once fixed.
"""

import numpy as np
import pytest

from sparket.validator.scoring.metrics.proper_scoring import (
    brier_score,
    brier_score_batch,
    log_loss,
    log_loss_batch,
)
from sparket.validator.scoring.metrics.time_series import (
    align_time_series,
)
from sparket.validator.scoring.aggregation.shrinkage import (
    shrink_toward_mean,
)


class TestBrierLogLossConsistency:
    """Brier and log-loss should treat unnormalized forecasts consistently."""

    def test_unnormalized_forecast_treated_same(self):
        """Both metrics should handle unnormalized forecasts the same way.

        Current bug: Brier normalizes, log-loss doesn't.
        """
        # Forecast that sums to 1.2 (overround)
        forecast = np.array([0.6, 0.6])
        outcome = np.array([1, 0], dtype=np.int8)

        brier = brier_score(forecast, outcome)
        logloss = log_loss(forecast, outcome)

        # If normalized to [0.5, 0.5]:
        # - Brier = (0.5-1)^2 + (0.5-0)^2 = 0.5
        # - LogLoss = -log(0.5) ≈ 0.693

        # If NOT normalized (using 0.6):
        # - LogLoss = -log(0.6) ≈ 0.511

        # Current behavior: Brier normalizes to 0.5, but log-loss uses 0.6
        # This test exposes the inconsistency

        # Expected: both should either normalize or not
        # For proper scoring rules, normalization is debatable, but must be consistent!

        # Calculate what each would give if properly normalized
        normalized_forecast = forecast / forecast.sum()
        expected_logloss_normalized = -np.log(normalized_forecast[0])

        # Check if log_loss actually normalized (it should but doesn't!)
        # This assertion will FAIL with current code, exposing the bug
        assert np.isclose(logloss, expected_logloss_normalized, atol=0.01), (
            f"Log-loss should normalize forecast like Brier does. "
            f"Got {logloss}, expected {expected_logloss_normalized}"
        )

    def test_batch_consistency_with_single(self):
        """Batch versions should give same results as single versions."""
        forecasts = np.array([[0.6, 0.6], [0.9, 0.1]])
        outcomes = np.array([[1, 0], [1, 0]], dtype=np.int8)

        # Single computations
        brier_singles = [brier_score(f, o) for f, o in zip(forecasts, outcomes)]
        logloss_singles = [log_loss(f, o) for f, o in zip(forecasts, outcomes)]

        # Batch computations
        brier_batch = brier_score_batch(forecasts, outcomes)
        logloss_batch = log_loss_batch(forecasts, outcomes)

        # Should match
        np.testing.assert_allclose(brier_singles, brier_batch)
        np.testing.assert_allclose(logloss_singles, logloss_batch)


class TestAlignTimeSeriesRequiresSorted:
    """align_time_series uses searchsorted which requires sorted input."""

    def test_unsorted_input_gives_wrong_result(self):
        """If timestamps aren't sorted, align gives wrong values.

        This test exposes the bug - it will FAIL with correct behavior.
        """
        # Unsorted timestamps
        ts1 = np.array([300.0, 100.0, 200.0])  # NOT sorted!
        vals1 = np.array([0.3, 0.1, 0.2])  # Values correspond to ts1

        ts2 = np.array([100.0, 200.0, 300.0])  # Sorted
        vals2 = np.array([0.1, 0.2, 0.3])

        aligned1, aligned2 = align_time_series(ts1, vals1, ts2, vals2)

        # Common timestamps are all three
        assert len(aligned1) == 3, "Should find 3 common timestamps"

        # If alignment worked correctly:
        # At ts=100: vals1 should be 0.1 (second element of ts1)
        # At ts=200: vals1 should be 0.2 (third element of ts1)
        # At ts=300: vals1 should be 0.3 (first element of ts1)

        # But searchsorted on unsorted array gives wrong indices!
        # This will FAIL, exposing the bug

        # The function should either:
        # 1. Sort internally before searchsorted
        # 2. Document that input must be sorted
        # 3. Use a different algorithm (like set intersection + lookup)

        expected1 = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(aligned1, expected1, err_msg=(
            "align_time_series gives wrong values with unsorted input"
        ))


class TestShrinkageGamingVulnerability:
    """Shrinkage can be gamed by inflating n_eff."""

    def test_huge_neff_dominates_population_mean(self):
        """A miner with massive n_eff can shift the population mean.

        This isn't necessarily a bug, but it's a design concern.
        """
        # 10 honest miners with reasonable n_eff and moderate scores
        honest_vals = np.full(10, 0.5)
        honest_neffs = np.full(10, 100.0)

        # 1 attacker with huge n_eff and extreme score
        attacker_val = np.array([0.9])
        attacker_neff = np.array([100000.0])  # 1000x larger

        all_vals = np.concatenate([honest_vals, attacker_val])
        all_neffs = np.concatenate([honest_neffs, attacker_neff])

        shrunk = shrink_toward_mean(all_vals, all_neffs, k=200.0)

        # Population mean is n_eff-weighted, so attacker dominates
        # Honest miners get pulled toward attacker's value

        honest_shrunk = shrunk[:10]
        attacker_shrunk = shrunk[10]

        # Attacker barely shrinks (high n_eff)
        assert abs(attacker_shrunk - 0.9) < 0.01, "Attacker resists shrinkage"

        # Honest miners get pulled toward attacker
        # This is the vulnerability - honest scores decrease!
        assert honest_shrunk.mean() > 0.5, (
            f"Honest miners pulled toward attacker: {honest_shrunk.mean()}"
        )

    def test_neff_inflation_resistance(self):
        """There should be a cap or diminishing returns on n_eff.

        Currently there isn't - this test documents the vulnerability.
        """
        raw_val = 0.8
        pop_mean = 0.5

        # Test increasing n_eff
        n_effs = [10, 100, 1000, 10000, 100000, 1000000]
        shrink_amounts = []

        for n_eff in n_effs:
            shrunk = shrink_toward_mean(
                np.array([raw_val]),
                np.array([float(n_eff)]),
                k=200.0,
                population_mean=pop_mean,
            )[0]
            shrink_amounts.append(abs(shrunk - raw_val))

        # At some point, increasing n_eff shouldn't matter much
        # (diminishing returns)

        # Currently: shrinkage goes to 0 as n_eff increases without bound
        # Ideally: there should be some cap or log-scaling

        # This documents the current behavior
        print(f"\nShrinkage vs n_eff: {list(zip(n_effs, shrink_amounts))}")

        # At n_eff=10: shrunk = 10/(10+200) * 0.8 + 200/210 * 0.5 ≈ 0.51
        # At n_eff=1M: shrunk ≈ 0.8 (almost no shrinkage)

        assert shrink_amounts[-1] < 0.001, (
            "Very high n_eff eliminates shrinkage completely"
        )


class TestPSSEdgeCases:
    """PSS (Probability Skill Score) edge cases."""

    def test_pss_when_truth_is_perfect(self):
        """PSS when ground truth has score = 0 (perfect prediction).

        If truth_score = 0 and miner_score > 0:
        PSS = 1 - (miner/0) = undefined

        Currently returns 0, which might not be right.
        """
        from sparket.validator.scoring.metrics.proper_scoring import pss

        # Miner has score 0.5, truth has score 0 (perfect)
        result = pss(0.5, 0.0)

        # What should this be?
        # - 0 means "miner equals truth" (wrong - truth is better)
        # - negative infinity means "infinitely worse" (mathematically correct)
        # - Some large negative value as a cap?

        # Current behavior: returns 0
        # This is debatable - documenting the choice
        assert result == 0.0, "PSS returns 0 when truth_score is 0"

    def test_pss_negative_skill(self):
        """PSS when miner is worse than truth."""
        from sparket.validator.scoring.metrics.proper_scoring import pss

        # Miner has score 0.8, truth has score 0.2
        result = pss(0.8, 0.2)

        # PSS = 1 - (0.8/0.2) = 1 - 4 = -3
        # This is unbounded negative!

        assert result == -3.0
        # There's no lower bound on PSS - is this intentional?
        # Very bad miners could have PSS of -100 or worse


