"""Calibration and sharpness batch job.

Computes calibration and sharpness scores for all miners with
enough settled outcomes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
from sqlalchemy import text

from sparket.validator.config.scoring_params import get_scoring_params

from ..determinism import get_canonical_window_bounds
from ..metrics.calibration import compute_calibration
from ..metrics.sharpness import compute_sharpness
from .base import ScoringJob


# SQL templates
_SELECT_MINERS_WITH_OUTCOMES = """
    SELECT DISTINCT ms.miner_id, ms.miner_hotkey
    FROM miner_submission ms
    JOIN submission_outcome_score sos ON ms.submission_id = sos.submission_id
    WHERE sos.settled_at >= :window_start
      AND sos.settled_at < :window_end
      {miner_filter}
    ORDER BY ms.miner_id, ms.miner_hotkey
"""

_SELECT_MINER_PREDICTIONS = text(
    """
    SELECT
        ms.submission_id,
        ms.side,
        ms.imp_prob,
        sos.outcome_vector
    FROM miner_submission ms
    JOIN submission_outcome_score sos ON ms.submission_id = sos.submission_id
    WHERE ms.miner_id = :miner_id
      AND ms.miner_hotkey = :miner_hotkey
      AND sos.settled_at >= :window_start
      AND sos.settled_at < :window_end
    ORDER BY ms.submission_id
    """
)

_UPDATE_CALIBRATION_SHARPNESS = text(
    """
    UPDATE miner_rolling_score
    SET cal_score = :cal_score,
        sharp_score = :sharp_score
    WHERE miner_id = :miner_id
      AND miner_hotkey = :miner_hotkey
      AND as_of = :as_of
      AND window_days = :window_days
    """
)


# Map side to outcome vector index
SIDE_TO_INDEX: Dict[str, int] = {
    "home": 0,
    "away": 1,
    "draw": 2,
    "over": 0,
    "under": 1,
}


class CalibrationSharpnessJob(ScoringJob):
    """Compute calibration and sharpness for all miners.

    Calibration: How well do predicted probabilities match actual frequencies?
    Sharpness: How decisive are the predictions?
    """

    JOB_ID = "calibration_sharpness_v1"
    CHECKPOINT_INTERVAL = 50

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
        """Execute the calibration and sharpness job."""
        # Use calibration window (longer for more samples)
        window_days = self.params.windows.calibration_window_days
        if self.window_start is not None and self.window_end is not None:
            window_start, window_end = self.window_start, self.window_end
        else:
            window_start, window_end = get_canonical_window_bounds(
                window_days,
                reference_time=self.window_end,
            )

        self.logger.info(f"Computing calibration/sharpness for {window_days} day window")

        # Get miners with settled outcomes
        miner_filter = ""
        params: Dict[str, Any] = {"window_start": window_start, "window_end": window_end}
        if self.miner_id_min is not None and self.miner_id_max is not None:
            miner_filter = "AND ms.miner_id BETWEEN :miner_id_min AND :miner_id_max"
            params["miner_id_min"] = self.miner_id_min
            params["miner_id_max"] = self.miner_id_max
        miners = await self.db.read(
            text(_SELECT_MINERS_WITH_OUTCOMES.format(miner_filter=miner_filter)),
            params=params,
            mappings=True,
        )

        self.items_total = len(miners)
        self.logger.info(f"Found {self.items_total} miners with settled outcomes")

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
        """Compute calibration and sharpness for a single miner."""
        predictions = await self.db.read(
            _SELECT_MINER_PREDICTIONS,
            params={
                "miner_id": miner_id,
                "miner_hotkey": miner_hotkey,
                "window_start": window_start,
                "window_end": window_end,
            },
            mappings=True,
        )

        if not predictions:
            return

        # Extract probabilities and outcomes
        probs_list: List[float] = []
        outcomes_list: List[int] = []

        for pred in predictions:
            try:
                prob = float(pred["imp_prob"])
                side = pred["side"]
                outcome_vec = pred["outcome_vector"]

                if outcome_vec is None:
                    continue

                # Parse outcome vector
                if isinstance(outcome_vec, str):
                    outcome_vec = json.loads(outcome_vec)

                # Get index for this side
                side_idx = SIDE_TO_INDEX.get(side.lower())
                if side_idx is None or side_idx >= len(outcome_vec):
                    continue

                # Did this side win?
                outcome_hit = int(outcome_vec[side_idx])

                probs_list.append(prob)
                outcomes_list.append(outcome_hit)

            except Exception:
                continue

        if not probs_list:
            return

        # Convert to numpy arrays
        probs = np.array(probs_list, dtype=np.float64)
        outcomes = np.array(outcomes_list, dtype=np.int8)

        # Compute calibration
        cal_params = self.params.calibration
        cal_result = compute_calibration(
            probs,
            outcomes,
            num_bins=cal_params.num_bins,
            min_samples_per_bin=cal_params.min_samples_per_bin,
        )
        cal_score = cal_result.score

        # Compute sharpness (using all probs)
        sharp_params = self.params.sharpness
        sharp_score = compute_sharpness(
            probs,
            target_variance=float(sharp_params.target_variance),
        )

        # Update rolling score
        rolling_window = self.params.windows.rolling_window_days
        _, as_of = get_canonical_window_bounds(rolling_window)

        await self.db.write(
            _UPDATE_CALIBRATION_SHARPNESS,
            params={
                "miner_id": miner_id,
                "miner_hotkey": miner_hotkey,
                "as_of": as_of,
                "window_days": rolling_window,
                "cal_score": float(cal_score),
                "sharp_score": float(sharp_score),
            },
        )


__all__ = ["CalibrationSharpnessJob"]
