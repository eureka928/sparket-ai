"""Originality and lead-lag analysis job.

Computes:
- SOS (Source of Signal): How original/independent are miner predictions?
- LeadRatio: Does miner anticipate market moves?
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from sqlalchemy import text

from sparket.validator.config.scoring_params import get_scoring_params

from ..determinism import get_canonical_window_bounds
from ..metrics.time_series import (
    analyze_lead_lag,
    bucket_time_series,
    align_time_series,
    compute_correlation,
    compute_sos,
)
from .base import ScoringJob


# SQL templates
_SELECT_MINERS_WITH_SUBMISSIONS = """
    SELECT DISTINCT miner_id, miner_hotkey
    FROM miner_submission
    WHERE submitted_at >= :window_start
      AND submitted_at < :window_end
      {miner_filter}
    ORDER BY miner_id, miner_hotkey
"""

_SELECT_MINER_QUOTES = text(
    """
    SELECT
        ms.market_id,
        ms.side,
        ms.submitted_at,
        ms.imp_prob
    FROM miner_submission ms
    WHERE ms.miner_id = :miner_id
      AND ms.miner_hotkey = :miner_hotkey
      AND ms.submitted_at >= :window_start
      AND ms.submitted_at < :window_end
    ORDER BY ms.market_id, ms.side, ms.submitted_at
    """
)

_SELECT_GROUND_TRUTH_QUOTES = text(
    """
    SELECT
        pq.market_id,
        pq.side,
        pq.ts,
        pq.imp_prob
    FROM provider_quote pq
    WHERE pq.market_id = ANY(:market_ids)
      AND pq.ts >= :window_start
      AND pq.ts < :window_end
    ORDER BY pq.market_id, pq.side, pq.ts
    """
)

_UPDATE_ORIGINALITY_LEAD = text(
    """
    UPDATE miner_rolling_score
    SET sos_mean = :sos_mean,
        sos_score = :sos_score,
        lead_ratio = :lead_ratio,
        lead_score = :lead_score
    WHERE miner_id = :miner_id
      AND miner_hotkey = :miner_hotkey
      AND as_of = :as_of
      AND window_days = :window_days
    """
)


MarketSideKey = Tuple[int, str]  # (market_id, side)


class OriginalityLeadLagJob(ScoringJob):
    """Compute originality (SOS) and lead-lag metrics for all miners.

    SOS: 1 - |correlation| between miner and truth time series
    LeadRatio: Fraction of truth moves where miner moved first
    """

    JOB_ID = "originality_lead_lag_v1"
    CHECKPOINT_INTERVAL = 20  # Lower since this job is heavier

    def __init__(
        self,
        db: Any,
        logger: Any,
        *,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        miner_id_min: int | None = None,
        miner_id_max: int | None = None,
        job_id_override: str | None = None,
    ):
        """Initialize the job."""
        super().__init__(db, logger, job_id_override=job_id_override)
        self.params = get_scoring_params()
        self.window_start = window_start
        self.window_end = window_end
        self.miner_id_min = miner_id_min
        self.miner_id_max = miner_id_max

    async def execute(self) -> None:
        """Execute the originality and lead-lag job."""
        window_days = self.params.windows.rolling_window_days
        if self.window_start is not None and self.window_end is not None:
            window_start, window_end = self.window_start, self.window_end
        else:
            window_start, window_end = get_canonical_window_bounds(
                window_days,
                reference_time=self.window_end,
            )

        self.logger.info(f"Computing originality/lead-lag for {window_days} day window")

        # Get active miners
        miner_filter = ""
        params: Dict[str, Any] = {"window_start": window_start, "window_end": window_end}
        if self.miner_id_min is not None and self.miner_id_max is not None:
            miner_filter = "AND miner_id BETWEEN :miner_id_min AND :miner_id_max"
            params["miner_id_min"] = self.miner_id_min
            params["miner_id_max"] = self.miner_id_max
        miners = await self.db.read(
            text(_SELECT_MINERS_WITH_SUBMISSIONS.format(miner_filter=miner_filter)),
            params=params,
            mappings=True,
        )

        self.items_total = len(miners)
        self.logger.info(f"Found {self.items_total} active miners")

        if not miners:
            return

        start_idx = self.state.get("last_miner_idx", 0)

        for idx, miner in enumerate(miners[start_idx:], start=start_idx):
            await self._process_miner(
                miner_id=miner["miner_id"],
                miner_hotkey=miner["miner_hotkey"],
                window_start=window_start,
                window_end=window_end,
            )

            self.items_processed = idx + 1
            self.state["last_miner_idx"] = idx + 1
            await self.checkpoint_if_due()

    async def _process_miner(
        self,
        miner_id: int,
        miner_hotkey: str,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        """Compute originality and lead-lag for a single miner."""
        # Get miner quotes
        miner_quotes = await self.db.read(
            _SELECT_MINER_QUOTES,
            params={
                "miner_id": miner_id,
                "miner_hotkey": miner_hotkey,
                "window_start": window_start,
                "window_end": window_end,
            },
            mappings=True,
        )

        if not miner_quotes:
            return

        min_samples = self.params.windows.min_samples_originality
        if len(miner_quotes) < min_samples:
            return

        # Group miner quotes by (market_id, side)
        miner_by_market: Dict[MarketSideKey, List[Tuple[float, float]]] = {}
        market_ids = set()

        for q in miner_quotes:
            key = (q["market_id"], q["side"])
            market_ids.add(q["market_id"])

            if key not in miner_by_market:
                miner_by_market[key] = []
            miner_by_market[key].append(
                (q["submitted_at"].timestamp(), float(q["imp_prob"]))
            )

        # Get ground truth quotes for these markets
        truth_quotes = await self.db.read(
            _SELECT_GROUND_TRUTH_QUOTES,
            params={
                "market_ids": list(market_ids),
                "window_start": window_start,
                "window_end": window_end,
            },
            mappings=True,
        )

        # Group truth quotes by (market_id, side)
        truth_by_market: Dict[MarketSideKey, List[Tuple[float, float]]] = {}
        for q in truth_quotes:
            key = (q["market_id"], q["side"])
            if key not in truth_by_market:
                truth_by_market[key] = []
            truth_by_market[key].append(
                (q["ts"].timestamp(), float(q["imp_prob"]))
            )

        # Compute metrics per market, then aggregate
        sos_scores = []
        lead_ratios = []

        lead_lag_params = self.params.lead_lag
        bucket_seconds = lead_lag_params.bucket_minutes * 60
        move_threshold = float(lead_lag_params.move_threshold)
        lead_window_seconds = lead_lag_params.lead_window_minutes * 60
        lag_window_seconds = lead_lag_params.lag_window_minutes * 60

        for market_key, miner_points in miner_by_market.items():
            truth_points = truth_by_market.get(market_key)
            if not truth_points:
                continue

            # Convert to numpy arrays
            miner_times = np.array([p[0] for p in miner_points])
            miner_vals = np.array([p[1] for p in miner_points])
            truth_times = np.array([p[0] for p in truth_points])
            truth_vals = np.array([p[1] for p in truth_points])

            # Bucket both series
            miner_buck_ts, miner_buck_vals = bucket_time_series(
                miner_times, miner_vals, bucket_seconds
            )
            truth_buck_ts, truth_buck_vals = bucket_time_series(
                truth_times, truth_vals, bucket_seconds
            )

            if len(miner_buck_ts) < 2 or len(truth_buck_ts) < 2:
                continue

            # Analyze lead-lag
            result = analyze_lead_lag(
                truth_times=truth_buck_ts,
                truth_values=truth_buck_vals,
                miner_times=miner_buck_ts,
                miner_values=miner_buck_vals,
                lead_window_seconds=lead_window_seconds,
                lag_window_seconds=lag_window_seconds,
                move_threshold=move_threshold,
            )

            sos_scores.append(result.sos_score)
            if result.moves_matched > 0:
                lead_ratios.append(result.lead_ratio)

        # Aggregate scores
        sos_score = None
        if sos_scores:
            sos_score = sum(sos_scores) / len(sos_scores)

        lead_ratio = None
        lead_score = None
        if lead_ratios:
            lead_ratio = sum(lead_ratios) / len(lead_ratios)
            lead_score = lead_ratio

        # Update if we have at least one metric
        if sos_score is not None or lead_ratio is not None:
            _, as_of = get_canonical_window_bounds(
                self.params.windows.rolling_window_days
            )

            await self.db.write(
                _UPDATE_ORIGINALITY_LEAD,
                params={
                    "miner_id": miner_id,
                    "miner_hotkey": miner_hotkey,
                    "as_of": as_of,
                    "window_days": self.params.windows.rolling_window_days,
                    "sos_mean": float(sos_score) if sos_score else None,
                    "sos_score": float(sos_score) if sos_score else None,
                    "lead_ratio": float(lead_ratio) if lead_ratio else None,
                    "lead_score": float(lead_score) if lead_score else None,
                },
            )


__all__ = ["OriginalityLeadLagJob"]
