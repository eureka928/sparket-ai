from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Tuple


def floor_time_to_bucket(dt: datetime, bucket_seconds: int) -> datetime:
    """Floor a datetime to the start of its bucket (UTC).

    - Treats naive datetimes as UTC.
    - Returns a naive UTC datetime aligned to bucket_seconds.
    """
    if dt.tzinfo is None:
        aware = dt.replace(tzinfo=timezone.utc)
    else:
        aware = dt.astimezone(timezone.utc)
    epoch = int(aware.timestamp())
    bucket_epoch = (epoch // bucket_seconds) * bucket_seconds
    return datetime.fromtimestamp(bucket_epoch, tz=timezone.utc).replace(tzinfo=None)


def miner_submission_idempotency_key(
    miner_id: int,
    miner_hotkey: str,
    market_id: int,
    side: str,
    submitted_at: datetime,
    bucket_seconds: int,
) -> Tuple[int, str, int, str, datetime]:
    """Build the Task 1 idempotency key, returning storage-ready values.

    The returned datetime is bucketed; use it as `submitted_at` when inserting,
    so the existing unique constraint dedupes within the bucket.
    """
    ts_bucket = floor_time_to_bucket(submitted_at, bucket_seconds)
    return (miner_id, miner_hotkey, market_id, side, ts_bucket)


def stable_payload_hash(payload: dict) -> str:
    """Deterministic SHA-256 over canonical JSON of payload."""
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(data).hexdigest()


def inbox_outcome_dedupe_key(event_id: str | int, miner_hotkey: str, ts: datetime, bucket_seconds: int) -> str:
    """Task 3 outcome envelope key: outcome:{event_id}:{miner_hotkey}:{ts_bucket}.

    Use this as `Inbox.dedupe_key`.
    """
    ts_bucket = floor_time_to_bucket(ts, bucket_seconds)
    return f"outcome:{event_id}:{miner_hotkey}:{int(ts_bucket.timestamp())}"


__all__ = [
    "floor_time_to_bucket",
    "miner_submission_idempotency_key",
    "stable_payload_hash",
    "inbox_outcome_dedupe_key",
]


