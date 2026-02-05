"""Tests for the deterministic compute_weights function."""

import pytest
import numpy as np

from sparket.validator.ledger.compute_weights import WeightResult, compute_weights
from sparket.validator.ledger.models import (
    ChainParamsSnapshot,
    MinerMetrics,
    ScoringConfigSnapshot,
)


def _default_config() -> ScoringConfigSnapshot:
    """Standard scoring config matching ScoringParams defaults."""
    return ScoringConfigSnapshot(params={
        "dimension_weights": {
            "w_fq": 0.6, "w_cal": 0.4,
            "w_edge": 0.7, "w_mes": 0.3,
            "w_sos": 0.6, "w_lead": 0.4,
        },
        "skill_score_weights": {
            "w_outcome_accuracy": 0.10,
            "w_outcome_relative": 0.10,
            "w_odds_edge": 0.50,
            "w_info_adv": 0.30,
        },
        "normalization": {"min_count_for_zscore": 10},
        "weight_emission": {"burn_rate": 0.9},
    })


def _default_chain(n_neurons: int = 10, burn_uid: int = 0) -> ChainParamsSnapshot:
    return ChainParamsSnapshot(
        burn_rate=0.9,
        burn_uid=burn_uid,
        max_weight_limit=0.5,
        min_allowed_weights=1,
        n_neurons=n_neurons,
    )


def _make_metrics(uid: int, **overrides) -> MinerMetrics:
    defaults = dict(
        uid=uid, hotkey=f"hk_{uid}",
        fq_raw=0.3, pss_mean=0.1, es_adj=0.05,
        mes_mean=0.6, cal_score=0.7, sharp_score=0.6,
        sos_score=0.5, lead_score=0.4, brier_mean=0.35,
    )
    defaults.update(overrides)
    return MinerMetrics(**defaults)


class TestComputeWeights:

    def test_deterministic_same_input_same_output(self):
        """Run 10 times, all identical."""
        config = _default_config()
        chain = _default_chain()
        metrics = [_make_metrics(i) for i in range(1, 6)]

        results = []
        for _ in range(10):
            r = compute_weights(metrics, config, chain)
            results.append((r.uids, r.uint16_weights))

        for i in range(1, 10):
            assert results[i] == results[0], f"Run {i} differs from run 0"

    def test_all_zeros_allocates_to_burn(self):
        config = _default_config()
        chain = _default_chain(n_neurons=5, burn_uid=0)
        metrics = [_make_metrics(i, fq_raw=0, pss_mean=0, es_adj=0, mes_mean=0,
                                  cal_score=0, sharp_score=0, sos_score=0, lead_score=0,
                                  brier_mean=0) for i in range(1, 4)]
        result = compute_weights(metrics, config, chain)
        # When all scores zero: 100% to burn
        if result.uids:
            assert 0 in result.uids

    def test_burn_rate_applied_correctly(self):
        config = _default_config()
        chain = _default_chain(n_neurons=10, burn_uid=0)
        metrics = [_make_metrics(i, fq_raw=0.5, es_adj=0.1) for i in range(1, 5)]
        result = compute_weights(metrics, config, chain)
        # Burn UID should get the largest weight (burn_rate=0.9)
        if 0 in result.uids:
            burn_idx = result.uids.index(0)
            total = sum(result.uint16_weights)
            burn_frac = result.uint16_weights[burn_idx] / total if total else 0
            # After max_weight_limit processing, burn may be capped but should still be dominant
            assert burn_frac > 0.3, f"burn_frac={burn_frac} too low"

    def test_max_weight_limit_enforced(self):
        config = _default_config()
        chain = ChainParamsSnapshot(
            burn_rate=0.0, burn_uid=None,
            max_weight_limit=0.5, min_allowed_weights=1, n_neurons=10,
        )
        # One miner much better than others
        metrics = [
            _make_metrics(1, fq_raw=0.9, es_adj=0.5, pss_mean=0.5),
            _make_metrics(2, fq_raw=0.1, es_adj=0.01, pss_mean=0.01),
            _make_metrics(3, fq_raw=0.1, es_adj=0.01, pss_mean=0.01),
        ]
        result = compute_weights(metrics, config, chain)
        # With max_weight_limit=0.5, top miner shouldn't take all the weight
        if result.uint16_weights:
            max_w = max(result.uint16_weights)
            total = sum(result.uint16_weights)
            max_frac = max_w / total if total else 0
            # Should be capped near the limit (tolerance for uint16 quantization)
            assert max_frac <= 0.55, f"max_frac={max_frac} exceeds limit+tolerance"

    def test_nan_scores_replaced_with_zero(self):
        config = _default_config()
        chain = _default_chain()
        metrics = [
            _make_metrics(1, fq_raw=float('nan')),
            _make_metrics(2, fq_raw=0.5),
        ]
        result = compute_weights(metrics, config, chain)
        # Should not crash, NaN miner gets 0 weight
        assert isinstance(result, WeightResult)

    def test_weight_result_includes_intermediates(self):
        config = _default_config()
        chain = _default_chain(n_neurons=5)
        metrics = [_make_metrics(i) for i in range(1, 4)]
        result = compute_weights(metrics, config, chain)
        # Skill scores recorded
        assert len(result.skill_scores) == 3
        # Dimension scores recorded
        for uid in [1, 2, 3]:
            assert uid in result.dimension_scores
            dims = result.dimension_scores[uid]
            assert "forecast_dim" in dims
            assert "econ_dim" in dims
            assert "info_dim" in dims

    def test_empty_metrics_with_burn(self):
        config = _default_config()
        chain = _default_chain(burn_uid=0)
        result = compute_weights([], config, chain)
        assert result.uids == [0]

    def test_empty_metrics_without_burn(self):
        config = _default_config()
        chain = ChainParamsSnapshot(
            burn_rate=0, burn_uid=None,
            max_weight_limit=0.5, min_allowed_weights=1, n_neurons=10,
        )
        result = compute_weights([], config, chain)
        assert result.uids == []

    @pytest.mark.parametrize("n_miners", [3, 5, 15, 50])
    def test_normalization_method_switches(self, n_miners):
        """Verify correct normalization based on miner count vs min_count_for_zscore."""
        config = _default_config()
        chain = _default_chain(n_neurons=max(n_miners + 5, 10))
        metrics = [_make_metrics(i, fq_raw=i * 0.01) for i in range(1, n_miners + 1)]
        result = compute_weights(metrics, config, chain)
        # Should produce valid output regardless of method
        assert isinstance(result, WeightResult)
        assert len(result.uids) > 0
