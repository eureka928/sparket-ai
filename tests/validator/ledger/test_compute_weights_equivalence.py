"""Equivalence test: compute_weights() vs original SkillScoreJob normalization.

This is the critical "did we break scoring?" test. It feeds identical data
through both the old inline normalization path and the new shared
compute_weights() function, asserting identical skill_score outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparket.validator.ledger.compute_weights import compute_weights
from sparket.validator.ledger.models import ChainParamsSnapshot, MinerMetrics, ScoringConfigSnapshot
from sparket.validator.scoring.aggregation.normalization import (
    normalize_percentile,
    normalize_zscore_logistic,
)
from sparket.validator.config.scoring_params import get_scoring_params


def _original_skill_score_logic(rows: list[dict], params) -> dict[str, float]:
    """Reproduce the ORIGINAL SkillScoreJob normalization logic verbatim.

    This is a copy of the pre-refactor inline code from skill_score.py,
    used as the ground-truth reference for equivalence testing.
    """
    fq_raw_list = [r["fq_raw"] for r in rows]
    pss_list = [r["pss_mean"] for r in rows]
    cal_list = [r["cal_score"] for r in rows]
    es_adj_list = [r["es_adj"] for r in rows]
    mes_list = [r["mes_mean"] for r in rows]
    sos_list = [r["sos_score"] for r in rows]
    lead_list = [r["lead_score"] for r in rows]

    fq_raw = np.array(fq_raw_list)
    pss = np.array(pss_list)
    cal = np.array(cal_list)
    es_adj = np.array(es_adj_list)
    mes = np.array(mes_list)
    sos = np.array(sos_list)
    lead = np.array(lead_list)

    # Original normalization
    fq_norm = (fq_raw + 1) / 2
    fq_norm = np.clip(fq_norm, 0, 1)

    min_count = int(params.normalization.min_count_for_zscore)
    use_zscore = len(rows) >= min_count
    if use_zscore:
        pss_norm = normalize_zscore_logistic(pss)
        es_norm = normalize_zscore_logistic(es_adj)
    else:
        pss_norm = normalize_percentile(pss)
        es_norm = normalize_percentile(es_adj)

    cal_norm = np.clip(cal, 0, 1)
    mes_norm = np.clip(mes, 0, 1)
    sos_norm = np.clip(sos, 0, 1)
    lead_norm = np.clip(lead, 0, 1)

    dim_weights = params.dimension_weights
    skill_weights = params.skill_score_weights

    w_fq = float(dim_weights.w_fq)
    w_cal = float(dim_weights.w_cal)
    forecast_dim = w_fq * fq_norm + w_cal * cal_norm

    skill_dim = pss_norm

    w_edge = float(dim_weights.w_edge)
    w_mes = float(dim_weights.w_mes)
    econ_dim = w_edge * es_norm + w_mes * mes_norm

    w_sos = float(dim_weights.w_sos)
    w_lead = float(dim_weights.w_lead)
    info_dim = w_sos * sos_norm + w_lead * lead_norm

    w_outcome_accuracy = float(skill_weights.w_outcome_accuracy)
    w_outcome_relative = float(skill_weights.w_outcome_relative)
    w_odds_edge = float(skill_weights.w_odds_edge)
    w_info_adv = float(skill_weights.w_info_adv)

    skill_score = (
        w_outcome_accuracy * forecast_dim
        + w_outcome_relative * skill_dim
        + w_odds_edge * econ_dim
        + w_info_adv * info_dim
    )

    return {r["hotkey"]: float(skill_score[i]) for i, r in enumerate(rows)}


def _build_test_miners(n: int = 15, seed: int = 42) -> list[dict]:
    """Generate N miners with varied, realistic metrics."""
    rng = np.random.RandomState(seed)
    miners = []
    for i in range(n):
        miners.append({
            "uid": i + 1,
            "hotkey": f"hk_{i+1}",
            "fq_raw": float(rng.uniform(-0.5, 0.8)),
            "pss_mean": float(rng.uniform(-0.3, 0.5)),
            "es_adj": float(rng.uniform(-0.2, 0.4)),
            "mes_mean": float(rng.uniform(0.3, 0.9)),
            "cal_score": float(rng.uniform(0.2, 0.9)),
            "sharp_score": float(rng.uniform(0.2, 0.8)),
            "sos_score": float(rng.uniform(0.1, 0.9)),
            "lead_score": float(rng.uniform(0.1, 0.8)),
            "brier_mean": float(rng.uniform(0.1, 0.45)),
        })
    return miners


class TestComputeWeightsEquivalence:
    """Prove compute_weights normalization matches the original SkillScoreJob."""

    @pytest.mark.parametrize("n_miners", [3, 5, 10, 15, 50])
    def test_skill_scores_match_original(self, n_miners: int):
        """Feed same data through old logic and new function, assert identical."""
        params = get_scoring_params()
        miners = _build_test_miners(n=n_miners, seed=n_miners)

        # -- Old path: original inline normalization --
        old_scores = _original_skill_score_logic(miners, params)

        # -- New path: compute_weights() --
        metrics = [
            MinerMetrics(
                uid=m["uid"], hotkey=m["hotkey"],
                fq_raw=m["fq_raw"], pss_mean=m["pss_mean"],
                es_adj=m["es_adj"], mes_mean=m["mes_mean"],
                cal_score=m["cal_score"], sharp_score=m["sharp_score"],
                sos_score=m["sos_score"], lead_score=m["lead_score"],
                brier_mean=m["brier_mean"],
            )
            for m in miners
        ]

        config = ScoringConfigSnapshot(params=params.model_dump(mode="json"))
        chain = ChainParamsSnapshot(
            burn_rate=0.0, burn_uid=None,
            max_weight_limit=1.0, min_allowed_weights=1,
            n_neurons=max(m["uid"] for m in miners) + 5,
        )

        result = compute_weights(metrics, config, chain)

        # Compare skill scores (before weight processing)
        for m in miners:
            hk = m["hotkey"]
            uid = m["uid"]
            old_val = old_scores[hk]
            new_val = result.skill_scores.get(uid)
            assert new_val is not None, f"Missing uid {uid} in compute_weights output"
            assert abs(old_val - new_val) < 1e-10, (
                f"Miner {hk}: old={old_val:.12f} vs new={new_val:.12f}, "
                f"diff={abs(old_val - new_val):.2e}"
            )

    def test_dimension_scores_match_original(self):
        """Verify individual dimension scores (forecast, econ, info) match."""
        params = get_scoring_params()
        miners = _build_test_miners(n=20, seed=99)

        # Old path
        rows = miners
        fq_raw = np.array([r["fq_raw"] for r in rows])
        pss = np.array([r["pss_mean"] for r in rows])
        cal = np.array([r["cal_score"] for r in rows])
        es_adj = np.array([r["es_adj"] for r in rows])
        mes = np.array([r["mes_mean"] for r in rows])
        sos = np.array([r["sos_score"] for r in rows])
        lead = np.array([r["lead_score"] for r in rows])

        fq_norm = np.clip((fq_raw + 1) / 2, 0, 1)
        pss_norm = normalize_zscore_logistic(pss)
        es_norm = normalize_zscore_logistic(es_adj)
        cal_norm = np.clip(cal, 0, 1)
        mes_norm = np.clip(mes, 0, 1)
        sos_norm = np.clip(sos, 0, 1)
        lead_norm = np.clip(lead, 0, 1)

        dim_w = params.dimension_weights
        old_forecast = float(dim_w.w_fq) * fq_norm + float(dim_w.w_cal) * cal_norm
        old_econ = float(dim_w.w_edge) * es_norm + float(dim_w.w_mes) * mes_norm
        old_info = float(dim_w.w_sos) * sos_norm + float(dim_w.w_lead) * lead_norm

        # New path
        metrics = [
            MinerMetrics(uid=m["uid"], hotkey=m["hotkey"], **{
                k: m[k] for k in ("fq_raw", "pss_mean", "es_adj", "mes_mean",
                                    "cal_score", "sharp_score", "sos_score",
                                    "lead_score", "brier_mean")
            })
            for m in miners
        ]
        config = ScoringConfigSnapshot(params=params.model_dump(mode="json"))
        chain = ChainParamsSnapshot(
            burn_rate=0.0, burn_uid=None,
            max_weight_limit=1.0, min_allowed_weights=1, n_neurons=30,
        )
        result = compute_weights(metrics, config, chain)

        # Compare dimension scores for each miner
        sorted_miners = sorted(miners, key=lambda m: m["uid"])
        for i, m in enumerate(sorted_miners):
            uid = m["uid"]
            dims = result.dimension_scores.get(uid, {})
            assert abs(dims["forecast_dim"] - float(old_forecast[i])) < 1e-10
            assert abs(dims["econ_dim"] - float(old_econ[i])) < 1e-10
            assert abs(dims["info_dim"] - float(old_info[i])) < 1e-10

    def test_edge_case_single_miner(self):
        """Single miner: both paths should give score 0.5 from normalization."""
        params = get_scoring_params()
        miners = _build_test_miners(n=1, seed=1)

        old_scores = _original_skill_score_logic(miners, params)
        metrics = [MinerMetrics(
            uid=miners[0]["uid"], hotkey=miners[0]["hotkey"],
            fq_raw=miners[0]["fq_raw"], pss_mean=miners[0]["pss_mean"],
            es_adj=miners[0]["es_adj"], mes_mean=miners[0]["mes_mean"],
            cal_score=miners[0]["cal_score"], sharp_score=miners[0]["sharp_score"],
            sos_score=miners[0]["sos_score"], lead_score=miners[0]["lead_score"],
            brier_mean=miners[0]["brier_mean"],
        )]

        config = ScoringConfigSnapshot(params=params.model_dump(mode="json"))
        chain = ChainParamsSnapshot(
            burn_rate=0.0, burn_uid=None,
            max_weight_limit=1.0, min_allowed_weights=1, n_neurons=10,
        )
        result = compute_weights(metrics, config, chain)

        old_val = old_scores[miners[0]["hotkey"]]
        new_val = result.skill_scores.get(miners[0]["uid"])
        assert abs(old_val - new_val) < 1e-10

    def test_edge_case_identical_miners(self):
        """All miners identical: both paths should give equal scores."""
        params = get_scoring_params()
        base = {
            "fq_raw": 0.3, "pss_mean": 0.1, "es_adj": 0.05,
            "mes_mean": 0.6, "cal_score": 0.7, "sharp_score": 0.6,
            "sos_score": 0.5, "lead_score": 0.4, "brier_mean": 0.35,
        }
        miners = [{"uid": i + 1, "hotkey": f"hk_{i+1}", **base} for i in range(10)]

        old_scores = _original_skill_score_logic(miners, params)
        metrics = [
            MinerMetrics(uid=m["uid"], hotkey=m["hotkey"], **base)
            for m in miners
        ]

        config = ScoringConfigSnapshot(params=params.model_dump(mode="json"))
        chain = ChainParamsSnapshot(
            burn_rate=0.0, burn_uid=None,
            max_weight_limit=1.0, min_allowed_weights=1, n_neurons=20,
        )
        result = compute_weights(metrics, config, chain)

        # All scores should be equal
        scores = list(result.skill_scores.values())
        assert all(abs(s - scores[0]) < 1e-10 for s in scores)

        # And match old path
        old_vals = list(old_scores.values())
        assert all(abs(o - old_vals[0]) < 1e-10 for o in old_vals)
        assert abs(scores[0] - old_vals[0]) < 1e-10
