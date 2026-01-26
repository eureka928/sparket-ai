"""Final skill score computation job.

Combines all dimension scores into the final SkillScore using 4 dimensions:

1. ForecastDim (Accuracy): How accurate are predictions vs outcome?
   - FQ: 1 - 2*brier_mean (transforms Brier to [0,1] scale)
   - CAL: Calibration score

2. SkillDim (Relative Skill): How well does miner beat the market?
   - PSS: Time-adjusted PSS vs matched snapshot

3. EconDim (Economic Edge): Does miner beat the closing line?
   - EDGE: CLE-based edge score
   - MES: Market efficiency score

4. InfoDim (Information Value): Does miner have information advantage?
   - SOS: Source of signal (independence)
   - LEAD: Lead-lag ratio

SkillScore = w_forecast * ForecastDim + w_skill * SkillDim + w_econ * EconDim + w_info * InfoDim
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import text

from sparket.validator.config.scoring_params import get_scoring_params

from ..aggregation.normalization import normalize_zscore_logistic, normalize_percentile
from ..determinism import get_canonical_window_bounds
from .base import ScoringJob


# SQL queries
_SELECT_ROLLING_SCORES = text(
    """
    SELECT
        miner_id,
        miner_hotkey,
        fq_raw,
        brier_mean,
        pss_mean,
        cal_score,
        sharp_score,
        es_adj,
        mes_mean,
        sos_score,
        lead_score
    FROM miner_rolling_score
    WHERE as_of = :as_of
      AND window_days = :window_days
    ORDER BY miner_id, miner_hotkey
    """
)

_UPDATE_SKILL_SCORE = text(
    """
    UPDATE miner_rolling_score
    SET fq_score = :fq_score,
        edge_score = :edge_score,
        mes_score = :mes_score,
        sos_score = :sos_score_norm,
        lead_score = :lead_score_norm,
        forecast_dim = :forecast_dim,
        econ_dim = :econ_dim,
        info_dim = :info_dim,
        skill_score = :skill_score,
        score_version = score_version + 1
    WHERE miner_id = :miner_id
      AND miner_hotkey = :miner_hotkey
      AND as_of = :as_of
      AND window_days = :window_days
    """
)


MinerKey = str  # "miner_id:miner_hotkey"


def make_miner_key(miner_id: int, miner_hotkey: str) -> MinerKey:
    """Create a miner lookup key."""
    return f"{miner_id}:{miner_hotkey}"


class SkillScoreJob(ScoringJob):
    """Compute final skill score for all miners.

    Normalizes metrics across all miners and computes 4-dimension composite.
    """

    JOB_ID = "skill_score_v2"

    def __init__(
        self,
        db: Any,
        logger: Any,
        *,
        as_of: datetime | None = None,
        job_id_override: str | None = None,
    ):
        """Initialize the job."""
        super().__init__(db, logger, job_id_override=job_id_override)
        self.params = get_scoring_params()
        self.as_of = as_of

    async def execute(self) -> None:
        """Execute the skill score computation."""
        window_days = self.params.windows.rolling_window_days
        if self.as_of is not None:
            as_of = self.as_of
        else:
            _, as_of = get_canonical_window_bounds(window_days)

        self.logger.info(f"Computing skill scores as of {as_of}")

        # Fetch all rolling scores
        rows = await self.db.read(
            _SELECT_ROLLING_SCORES,
            params={"as_of": as_of, "window_days": window_days},
            mappings=True,
        )

        self.items_total = len(rows)
        self.logger.info(f"Found {self.items_total} miners with rolling scores")

        if not rows:
            return

        # Collect raw metrics by miner
        keys: List[MinerKey] = []
        miner_info: Dict[MinerKey, Dict[str, Any]] = {}

        fq_raw_list = []
        pss_list = []
        cal_list = []
        es_adj_list = []
        mes_list = []
        sos_list = []
        lead_list = []

        for row in rows:
            key = make_miner_key(row["miner_id"], row["miner_hotkey"])
            keys.append(key)
            miner_info[key] = {
                "miner_id": row["miner_id"],
                "miner_hotkey": row["miner_hotkey"],
            }

            # FQ = 1 - 2*brier (already computed in rolling_aggregates)
            fq_raw_list.append(self._to_float_safe(row["fq_raw"], 0.0))
            # PSS (time-adjusted from rolling_aggregates)
            pss_list.append(self._to_float_safe(row["pss_mean"], 0.0))
            # Calibration
            cal_list.append(self._to_float_safe(row["cal_score"], 0.5))
            # Economic edge
            es_adj_list.append(self._to_float_safe(row["es_adj"], 0.0))
            mes_list.append(self._to_float_safe(row["mes_mean"], 0.5))
            # Information value
            sos_list.append(self._to_float_safe(row["sos_score"], 0.5))
            lead_list.append(self._to_float_safe(row["lead_score"], 0.5))

        # Convert to arrays
        fq_raw = np.array(fq_raw_list)
        pss = np.array(pss_list)
        cal = np.array(cal_list)
        es_adj = np.array(es_adj_list)
        mes = np.array(mes_list)
        sos = np.array(sos_list)
        lead = np.array(lead_list)

        # Normalize metrics that need normalization
        # FQ is already [-1, 1] from 1 - 2*brier, normalize to [0, 1]
        fq_norm = (fq_raw + 1) / 2  # Linear map from [-1,1] to [0,1]
        fq_norm = np.clip(fq_norm, 0, 1)
        
        # PSS can be negative (worse than baseline), normalize
        min_count = int(self.params.normalization.min_count_for_zscore)
        use_zscore = len(rows) >= min_count
        if use_zscore:
            pss_norm = normalize_zscore_logistic(pss)
            es_norm = normalize_zscore_logistic(es_adj)
        else:
            pss_norm = normalize_percentile(pss)
            es_norm = normalize_percentile(es_adj)

        # Others are already in [0, 1] range
        cal_norm = np.clip(cal, 0, 1)
        mes_norm = np.clip(mes, 0, 1)
        sos_norm = np.clip(sos, 0, 1)
        lead_norm = np.clip(lead, 0, 1)

        # Get dimension weights
        dim_weights = self.params.dimension_weights
        skill_weights = self.params.skill_score_weights

        # ForecastDim = w_fq * FQ + w_cal * CAL (accuracy-based)
        w_fq = float(dim_weights.w_fq)
        w_cal = float(dim_weights.w_cal)
        forecast_dim = w_fq * fq_norm + w_cal * cal_norm

        # SkillDim = PSS (relative skill vs market, time-adjusted)
        # This is a single metric dimension
        skill_dim = pss_norm

        # EconDim = w_edge * EDGE + w_mes * MES
        w_edge = float(dim_weights.w_edge)
        w_mes = float(dim_weights.w_mes)
        econ_dim = w_edge * es_norm + w_mes * mes_norm

        # InfoDim = w_sos * SOS + w_lead * LEAD
        w_sos = float(dim_weights.w_sos)
        w_lead = float(dim_weights.w_lead)
        info_dim = w_sos * sos_norm + w_lead * lead_norm

        # SkillScore = weighted combination of all dimensions (task-mapped)
        w_outcome_accuracy = float(skill_weights.w_outcome_accuracy)
        w_outcome_relative = float(skill_weights.w_outcome_relative)
        w_odds_edge = float(skill_weights.w_odds_edge)
        w_info_adv = float(skill_weights.w_info_adv)
        
        skill_score = (
            w_outcome_accuracy * forecast_dim +
            w_outcome_relative * skill_dim +
            w_odds_edge * econ_dim +
            w_info_adv * info_dim
        )

        # Persist results
        for i, key in enumerate(keys):
            info = miner_info[key]
            await self.db.write(
                _UPDATE_SKILL_SCORE,
                params={
                    "miner_id": info["miner_id"],
                    "miner_hotkey": info["miner_hotkey"],
                    "as_of": as_of,
                    "window_days": window_days,
                    "fq_score": float(fq_norm[i]),
                    "edge_score": float(es_norm[i]),
                    "mes_score": float(mes_norm[i]),
                    "sos_score_norm": float(sos_norm[i]),
                    "lead_score_norm": float(lead_norm[i]),
                    "forecast_dim": float(forecast_dim[i]),
                    "econ_dim": float(econ_dim[i]),
                    "info_dim": float(info_dim[i]),
                    "skill_score": float(skill_score[i]),
                },
            )
            self.items_processed += 1

    def _to_float_safe(self, val: Any, default: float) -> float:
        """Safely convert to float, returning default on failure."""
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default


__all__ = ["SkillScoreJob"]
