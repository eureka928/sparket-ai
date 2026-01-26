"""Ground truth construction from sportsbook consensus.

This module handles:
- Sportsbook bias estimation (iterative calibration)
- Weighted consensus probability computation
- Periodic snapshot pipeline
- Time-matched snapshot lookup
"""

from __future__ import annotations

from .bias import BiasEstimator, BiasState, BiasKey, make_bias_key
from .consensus import ConsensusComputer, BookQuote
from .snapshot_pipeline import (
    SnapshotPipeline,
    SnapshotResult,
    MatchedSnapshot,
    find_matched_snapshot,
    find_closing_snapshot,
    find_matched_snapshots_batch,
)

__all__ = [
    "BiasEstimator",
    "BiasState",
    "BiasKey",
    "make_bias_key",
    "ConsensusComputer",
    "BookQuote",
    "SnapshotPipeline",
    "SnapshotResult",
    "MatchedSnapshot",
    "find_matched_snapshot",
    "find_closing_snapshot",
    "find_matched_snapshots_batch",
]

