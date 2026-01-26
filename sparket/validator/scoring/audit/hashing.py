"""Deterministic hashing for scoring inputs and outputs.

Provides hash functions for verifying that validators produce
identical scoring results from identical inputs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List


def _serialize_value(val: Any) -> Any:
    """Serialize a value for deterministic hashing."""
    if val is None:
        return None
    elif isinstance(val, Decimal):
        return str(val)
    elif isinstance(val, datetime):
        return val.isoformat()
    elif isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in sorted(val.items())}
    elif isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    elif isinstance(val, (int, float, str, bool)):
        return val
    else:
        return str(val)


def compute_hash(data: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of a dictionary.

    The hash is computed from a canonical JSON representation
    with sorted keys and consistent formatting.

    Args:
        data: Dictionary to hash

    Returns:
        Hex-encoded SHA256 hash (64 characters)
    """
    serialized = _serialize_value(data)
    canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_miner_score_hash(
    miner_id: int,
    miner_hotkey: str,
    as_of: datetime,
    window_days: int,
    scores: Dict[str, Any],
) -> str:
    """Compute hash of a miner's rolling scores.

    Args:
        miner_id: Miner identifier
        miner_hotkey: Miner hotkey
        as_of: Score timestamp
        window_days: Rolling window size
        scores: Score values

    Returns:
        Hex-encoded SHA256 hash
    """
    payload = {
        "miner_id": miner_id,
        "miner_hotkey": miner_hotkey,
        "as_of": as_of.isoformat(),
        "window_days": window_days,
        "scores": scores,
    }
    return compute_hash(payload)


def compute_batch_hash(
    as_of: datetime,
    window_days: int,
    miner_scores: List[Dict[str, Any]],
) -> str:
    """Compute hash of a full batch of miner scores.

    Args:
        as_of: Score timestamp
        window_days: Rolling window size
        miner_scores: List of miner score dicts

    Returns:
        Hex-encoded SHA256 hash
    """
    # Sort by miner_id for determinism
    sorted_scores = sorted(miner_scores, key=lambda x: (x.get("miner_id", 0), x.get("miner_hotkey", "")))

    payload = {
        "as_of": as_of.isoformat(),
        "window_days": window_days,
        "n_miners": len(sorted_scores),
        "scores": sorted_scores,
    }
    return compute_hash(payload)


def compute_bias_hash(
    bias_entries: List[Dict[str, Any]],
) -> str:
    """Compute hash of sportsbook bias estimates.

    Args:
        bias_entries: List of bias estimate dicts

    Returns:
        Hex-encoded SHA256 hash
    """
    # Sort by composite key for determinism
    sorted_entries = sorted(
        bias_entries,
        key=lambda x: (
            x.get("sportsbook_id", 0),
            x.get("sport_id", 0),
            x.get("market_kind", ""),
        ),
    )

    payload = {
        "n_entries": len(sorted_entries),
        "entries": sorted_entries,
    }
    return compute_hash(payload)


def compute_ground_truth_hash(
    closing_entries: List[Dict[str, Any]],
) -> str:
    """Compute hash of ground truth closing snapshots.

    Args:
        closing_entries: List of closing snapshot dicts

    Returns:
        Hex-encoded SHA256 hash
    """
    sorted_entries = sorted(
        closing_entries,
        key=lambda x: (x.get("market_id", 0), x.get("side", "")),
    )

    payload = {
        "n_entries": len(sorted_entries),
        "entries": sorted_entries,
    }
    return compute_hash(payload)


__all__ = [
    "compute_hash",
    "compute_miner_score_hash",
    "compute_batch_hash",
    "compute_bias_hash",
    "compute_ground_truth_hash",
]

