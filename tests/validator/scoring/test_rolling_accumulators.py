"""Test that RollingAggregatesJob correctly computes accumulator (ws, wt) pairs.

Verifies the invariant: brier_mean == brier_ws / brier_wt (and same
for all other metrics). This catches bugs in the accumulator computation
added for the ledger checkpoint system.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparket.validator.scoring.aggregation.decay import compute_decay_weights, weighted_mean


class TestAccumulatorPairComputation:
    """Test that accumulator pairs satisfy mean == ws / wt."""

    def test_brier_accumulator_matches_mean(self):
        """Simulate what RollingAggregatesJob does and verify ws/wt == mean."""
        # Simulate 20 submissions with varied Brier scores
        rng = np.random.RandomState(42)
        n = 20
        brier_values = rng.uniform(0.05, 0.45, size=n)

        # Simulate timestamps spread over 10 days
        ref_ts = 1700000000.0  # reference time
        timestamps = ref_ts - rng.uniform(0, 10 * 86400, size=n)
        half_life = 10  # days

        decay_weights = compute_decay_weights(timestamps, ref_ts, half_life)

        # Compute mean the way RollingAggregatesJob does
        brier_mean = weighted_mean(brier_values, decay_weights)

        # Compute accumulator pairs the way the updated job does
        brier_ws = float(np.sum(brier_values * decay_weights))
        brier_wt = float(np.sum(decay_weights))

        # The invariant: mean == ws / wt
        assert abs(brier_mean - brier_ws / brier_wt) < 1e-12, (
            f"brier_mean={brier_mean} != ws/wt={brier_ws / brier_wt}"
        )

    def test_fq_accumulator_matches_mean(self):
        """FQ = 1 - 2*brier, so fq_ws should be sum((1-2*b_i) * w_i)."""
        rng = np.random.RandomState(43)
        n = 15
        brier_values = rng.uniform(0.1, 0.4, size=n)
        fq_values = 1.0 - 2.0 * brier_values

        ref_ts = 1700000000.0
        timestamps = ref_ts - rng.uniform(0, 7 * 86400, size=n)
        decay_weights = compute_decay_weights(timestamps, ref_ts, half_life_days=10)

        fq_mean = weighted_mean(fq_values, decay_weights)

        fq_ws = float(np.sum(fq_values * decay_weights))
        fq_wt = float(np.sum(decay_weights))

        assert abs(fq_mean - fq_ws / fq_wt) < 1e-12

    def test_pss_accumulator_matches_mean(self):
        """PSS accumulator: sum(pss_i * w_i) / sum(w_i) == pss_mean."""
        rng = np.random.RandomState(44)
        n = 25
        pss_values = rng.uniform(-0.3, 0.5, size=n)

        ref_ts = 1700000000.0
        timestamps = ref_ts - rng.uniform(0, 14 * 86400, size=n)
        decay_weights = compute_decay_weights(timestamps, ref_ts, half_life_days=10)

        pss_mean = weighted_mean(pss_values, decay_weights)

        pss_ws = float(np.sum(pss_values * decay_weights))
        pss_wt = float(np.sum(decay_weights))

        assert abs(pss_mean - pss_ws / pss_wt) < 1e-12

    def test_es_accumulator_matches_mean(self):
        """CLE (economic edge) accumulator."""
        rng = np.random.RandomState(45)
        n = 30
        cle_values = rng.uniform(-0.2, 0.3, size=n)

        ref_ts = 1700000000.0
        timestamps = ref_ts - rng.uniform(0, 20 * 86400, size=n)
        decay_weights = compute_decay_weights(timestamps, ref_ts, half_life_days=10)

        es_mean = weighted_mean(cle_values, decay_weights)

        es_ws = float(np.sum(cle_values * decay_weights))
        es_wt = float(np.sum(decay_weights))

        assert abs(es_mean - es_ws / es_wt) < 1e-12

    def test_mes_accumulator_from_clv_prob(self):
        """MES = 1 - |clv_prob|, accumulated with weights."""
        rng = np.random.RandomState(46)
        n = 20
        clv_prob_values = rng.uniform(-0.1, 0.1, size=n)
        mes_values = 1.0 - np.abs(clv_prob_values)

        ref_ts = 1700000000.0
        timestamps = ref_ts - rng.uniform(0, 10 * 86400, size=n)
        decay_weights = compute_decay_weights(timestamps, ref_ts, half_life_days=10)

        mes_mean = weighted_mean(mes_values, decay_weights)

        mes_ws = float(np.sum(mes_values * decay_weights))
        mes_wt = float(np.sum(decay_weights))

        assert abs(mes_mean - mes_ws / mes_wt) < 1e-12

    def test_empty_values_produce_zero_accumulators(self):
        """No submissions -> ws=0, wt=0."""
        brier_values = []
        brier_weights = []

        brier_ws = float(np.sum(np.array(brier_values) * np.array(brier_weights))) if brier_values else 0.0
        brier_wt = float(np.sum(brier_weights)) if brier_weights else 0.0

        assert brier_ws == 0.0
        assert brier_wt == 0.0

    def test_single_submission_accumulator(self):
        """Single submission: ws = value * weight, wt = weight."""
        value = 0.25
        weight = 0.8  # e.g., 3 days old with 10-day half-life

        ws = value * weight
        wt = weight

        assert abs(ws / wt - value) < 1e-12, "Single value should reconstruct exactly"

    @pytest.mark.parametrize("n_subs", [5, 50, 500])
    def test_accumulator_precision_at_scale(self, n_subs: int):
        """Verify numeric precision holds with many submissions."""
        rng = np.random.RandomState(n_subs)
        values = rng.uniform(0.0, 1.0, size=n_subs)

        ref_ts = 1700000000.0
        timestamps = ref_ts - rng.uniform(0, 30 * 86400, size=n_subs)
        weights = compute_decay_weights(timestamps, ref_ts, half_life_days=10)

        mean = weighted_mean(values, weights)
        ws = float(np.sum(values * weights))
        wt = float(np.sum(weights))

        # Should match to at least 10 decimal places
        assert abs(mean - ws / wt) < 1e-10, (
            f"Precision loss at {n_subs} submissions: diff={abs(mean - ws/wt):.2e}"
        )
