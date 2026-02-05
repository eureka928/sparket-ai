"""Scoring ledger export system for primary+auditor validator model.

The ledger module handles exporting license-safe scoring data from the
primary validator so auditor validators can independently verify and
reproduce weights without a SportsDataIO subscription.

Exports use a checkpoint+delta model:
- Checkpoints: full accumulator state per miner (every scoring cycle)
- Deltas: settled submission outcome scores for independent Brier verification
"""

from .models import (
    AccumulatorEntry,
    ChainParamsSnapshot,
    CheckpointWindow,
    DeltaWindow,
    LedgerManifest,
    MinerMetrics,
    MinerRosterEntry,
    OutcomeEntry,
    RecomputeRecord,
    ScoringConfigSnapshot,
    SettledSubmissionEntry,
)
from .redaction import DataTier, redact
from .signer import sign_manifest, verify_manifest

__all__ = [
    "AccumulatorEntry",
    "ChainParamsSnapshot",
    "CheckpointWindow",
    "DataTier",
    "DeltaWindow",
    "LedgerManifest",
    "MinerMetrics",
    "MinerRosterEntry",
    "OutcomeEntry",
    "RecomputeRecord",
    "ScoringConfigSnapshot",
    "SettledSubmissionEntry",
    "redact",
    "sign_manifest",
    "verify_manifest",
]
