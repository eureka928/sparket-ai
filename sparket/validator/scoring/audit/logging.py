"""Structured audit logging for scoring operations.

Provides logging utilities for debugging consensus divergence
between validators.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .hashing import compute_batch_hash, compute_hash


class ScoringAuditLogger:
    """Structured logger for scoring audit trail.

    Logs key events and hashes during scoring for debugging
    consensus divergence.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the audit logger.

        Args:
            logger: Optional logger instance (creates one if not provided)
        """
        self.logger = logger or logging.getLogger("scoring.audit")

    def log_job_start(
        self,
        job_id: str,
        window_start: datetime,
        window_end: datetime,
        params_hash: str,
    ) -> None:
        """Log job start with parameters hash.

        Args:
            job_id: Job identifier
            window_start: Window start time
            window_end: Window end time
            params_hash: Hash of scoring parameters
        """
        self.logger.info({
            "event": "job_start",
            "job_id": job_id,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "params_hash": params_hash[:16] + "...",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def log_job_complete(
        self,
        job_id: str,
        items_processed: int,
        output_hash: str,
        duration_seconds: float,
    ) -> None:
        """Log job completion with output hash.

        Args:
            job_id: Job identifier
            items_processed: Number of items processed
            output_hash: Hash of job outputs
            duration_seconds: Job duration
        """
        self.logger.info({
            "event": "job_complete",
            "job_id": job_id,
            "items_processed": items_processed,
            "output_hash": output_hash[:16] + "...",
            "duration_seconds": round(duration_seconds, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def log_miner_score(
        self,
        miner_id: int,
        miner_hotkey: str,
        scores: Dict[str, Any],
        score_hash: str,
    ) -> None:
        """Log individual miner score computation.

        Args:
            miner_id: Miner identifier
            miner_hotkey: Miner hotkey (truncated for privacy)
            scores: Computed scores
            score_hash: Hash of scores
        """
        self.logger.debug({
            "event": "miner_score",
            "miner_id": miner_id,
            "miner_hotkey": miner_hotkey[:16] + "...",
            "skill_score": scores.get("skill_score"),
            "score_hash": score_hash[:16] + "...",
        })

    def log_bias_update(
        self,
        n_entries: int,
        version: int,
        bias_hash: str,
    ) -> None:
        """Log bias estimate update.

        Args:
            n_entries: Number of bias entries updated
            version: New bias version
            bias_hash: Hash of bias entries
        """
        self.logger.info({
            "event": "bias_update",
            "n_entries": n_entries,
            "version": version,
            "bias_hash": bias_hash[:16] + "...",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def log_ground_truth_snapshot(
        self,
        n_markets: int,
        n_sides: int,
        snapshot_hash: str,
    ) -> None:
        """Log ground truth closing snapshot.

        Args:
            n_markets: Number of markets
            n_sides: Number of side entries
            snapshot_hash: Hash of snapshot
        """
        self.logger.info({
            "event": "ground_truth_snapshot",
            "n_markets": n_markets,
            "n_sides": n_sides,
            "snapshot_hash": snapshot_hash[:16] + "...",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def log_consensus_check(
        self,
        as_of: datetime,
        n_miners: int,
        batch_hash: str,
    ) -> None:
        """Log consensus verification point.

        This should be logged by all validators to enable comparison.

        Args:
            as_of: Score timestamp
            n_miners: Number of miners scored
            batch_hash: Hash of full batch
        """
        self.logger.info({
            "event": "consensus_check",
            "as_of": as_of.isoformat(),
            "n_miners": n_miners,
            "batch_hash": batch_hash,  # Full hash for comparison
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# Default instance
_audit_logger: Optional[ScoringAuditLogger] = None


def get_audit_logger() -> ScoringAuditLogger:
    """Get or create the default audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = ScoringAuditLogger()
    return _audit_logger


__all__ = [
    "ScoringAuditLogger",
    "get_audit_logger",
]

