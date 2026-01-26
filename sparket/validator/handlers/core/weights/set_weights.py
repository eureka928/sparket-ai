"""Handler for setting weights on chain.

Reads skill scores from the database and emits them to the chain.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import bittensor as bt
from sqlalchemy import text

from sparket.validator.config.scoring_params import get_scoring_params
from sparket.validator.scoring.determinism import get_canonical_window_bounds

from .weights_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)


# SQL query to fetch latest skill scores
_SELECT_SKILL_SCORES = text(
    """
    SELECT
        m.uid,
        m.hotkey,
        mrs.skill_score
    FROM miner m
    JOIN miner_rolling_score mrs ON m.miner_id = mrs.miner_id AND m.hotkey = mrs.miner_hotkey
    WHERE m.netuid = :netuid
      AND m.active = 1
      AND mrs.as_of = :as_of
      AND mrs.window_days = :window_days
      AND mrs.skill_score IS NOT NULL
    ORDER BY m.uid
    """
)


class SetWeightsHandler:
    """Handler for setting weights on chain based on skill scores."""

    def __init__(self, database: Any):
        """Initialize the handler.

        Args:
            database: Database manager
        """
        self.database = database
        self.params = get_scoring_params()

    async def load_scores_from_db(
        self,
        netuid: int,
        n_neurons: int,
    ) -> np.ndarray:
        """Load skill scores from database for all active miners.

        Args:
            netuid: Network UID
            n_neurons: Total number of neurons in metagraph

        Returns:
            Array of scores indexed by UID (zeros for missing miners)
        """
        window_days = self.params.windows.rolling_window_days
        _, as_of = get_canonical_window_bounds(window_days)

        try:
            rows = await self.database.read(
                _SELECT_SKILL_SCORES,
                params={
                    "netuid": netuid,
                    "as_of": as_of,
                    "window_days": window_days,
                },
                mappings=True,
            )

            # Initialize scores array with zeros
            scores = np.zeros(n_neurons, dtype=np.float32)

            # Fill in scores from database
            for row in rows:
                uid = row["uid"]
                skill_score = row["skill_score"]

                if uid is not None and 0 <= uid < n_neurons and skill_score is not None:
                    scores[uid] = float(skill_score)

            bt.logging.info({
                "load_scores": {
                    "miners_found": len(rows),
                    "total_neurons": n_neurons,
                    "non_zero_scores": int(np.count_nonzero(scores)),
                }
            })

            return scores

        except Exception as e:
            bt.logging.warning({"load_scores_error": str(e)})
            return np.zeros(n_neurons, dtype=np.float32)

    def set_weights(self, validator: Any) -> None:
        """Set weights on chain from in-memory scores.

        This is the synchronous interface for compatibility.
        Uses validator.scores if set, otherwise loads from database.

        Args:
            validator: Validator instance
        """
        scores = getattr(validator, "scores", None)

        if scores is None:
            bt.logging.warning("set_weights(): validator has no scores; skipping")
            return

        self._emit_weights(validator, scores)

    async def set_weights_from_db(self, validator: Any) -> None:
        """Set weights on chain from database scores.

        This is the primary method that loads skill scores from the
        database and emits them to the chain.

        Args:
            validator: Validator instance
        """
        netuid = validator.config.netuid
        n_neurons = validator.metagraph.n

        # Load scores from database
        scores = await self.load_scores_from_db(netuid, n_neurons)

        # Update validator's in-memory scores
        validator.scores = scores

        # Emit to chain
        self._emit_weights(validator, scores)

    def _emit_weights(self, validator: Any, scores: np.ndarray) -> None:
        """Emit weights to chain.

        Args:
            validator: Validator instance
            scores: Score array indexed by UID
        """
        # Check if scores contains any NaN values
        if np.isnan(scores).any():
            bt.logging.warning(
                "Scores contain NaN values. Replacing with zeros."
            )
            scores = np.nan_to_num(scores, nan=0.0)

        # Calculate norm and normalize
        norm = np.linalg.norm(scores, ord=1, axis=0, keepdims=True)
        if np.any(norm == 0) or np.isnan(norm).any():
            bt.logging.warning("All scores are zero or invalid; skipping weight emission")
            return

        raw_weights = scores / norm

        bt.logging.debug({"raw_weights": raw_weights.tolist()[:10]})  # First 10
        bt.logging.debug({"raw_weight_uids": validator.metagraph.uids.tolist()[:10]})

        processed_weight_uids, processed_weights = process_weights_for_netuid(
            uids=validator.metagraph.uids,
            weights=raw_weights,
            netuid=validator.config.netuid,
            subtensor=validator.subtensor,
            metagraph=validator.metagraph,
        )

        bt.logging.debug({"processed_weights_sample": processed_weights[:10].tolist()})
        bt.logging.debug({"processed_weight_uids_sample": processed_weight_uids[:10].tolist()})

        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )

        bt.logging.debug({"uint_weights_sample": uint_weights[:10]})
        bt.logging.debug({"uint_uids_sample": uint_uids[:10]})

        # Set the weights on chain
        try:
            result, msg = validator.subtensor.set_weights(
                wallet=validator.wallet,
                netuid=validator.config.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=False,
                wait_for_inclusion=False,
                version_key=validator.spec_version,
            )

            if result is True:
                bt.logging.info({
                    "set_weights": "success",
                    "n_weights": len(uint_weights),
                })
            else:
                bt.logging.error({"set_weights": "failed", "message": str(msg)})

        except Exception as e:
            bt.logging.error({"set_weights_error": str(e)})


__all__ = ["SetWeightsHandler"]
