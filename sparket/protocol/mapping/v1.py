"""API v1 → storage mapping layer.

Thin adapters from public request models to internal storage row shapes.
Idempotency bucketing is applied on write to align with database uniqueness
constraints.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List

from sparket.protocol.mapping.idempotency import (
    floor_time_to_bucket,
    miner_submission_idempotency_key,
    inbox_outcome_dedupe_key,
)
from sparket.protocol.models.v1.odds import SubmitOddsRequest, MarketSubmission
from sparket.protocol.models.v1.common import PriceSide
from sparket.protocol.models.v1.outcomes import SubmitOutcomeRequest
from sparket.shared.rows import MinerSubmissionRow, InboxRow


def _ensure_imp_prob(odds_eu: float, imp_prob: float | None) -> float:
    if imp_prob is not None:
        return float(imp_prob)
    return 1.0 / float(odds_eu)


def map_submit_odds_to_miner_submission_rows(
    request: SubmitOddsRequest,
    received_at: datetime,
    bucket_seconds: int,
) -> List[MinerSubmissionRow]:
    """Flatten SubmitOddsRequest into miner_submission-shaped rows.

    Uses server `received_at` (bucketed) for idempotency against the
    `(miner_id, miner_hotkey, market_id, side, submitted_at)` unique index.
    """
    rows: List[MinerSubmissionRow] = []
    submitted_at_bucket = floor_time_to_bucket(received_at, bucket_seconds)
    for sub in request.submissions:
        for price in sub.prices:
            rows.append({
                "miner_id": request.miner_id,
                "miner_hotkey": request.miner_hotkey,
                "market_id": sub.market_id,
                "side": price.side,
                "submitted_at": submitted_at_bucket,
                "priced_at": sub.priced_at,
                "odds_eu": price.odds_eu,
                "imp_prob": _ensure_imp_prob(price.odds_eu, price.imp_prob),
                "payload": {"kind": sub.kind},
            })
    return rows


def map_submit_outcome_to_inbox_row(
    request: SubmitOutcomeRequest,
    received_at: datetime,
    bucket_seconds: int,
    topic: str = "outcome.submit",
) -> InboxRow:
    """Build an inbox row with dedupe key for outcome submissions.

    Payload mirrors the API request for transparency; use dedupe_key to
    idempotently process envelopes inside the bucket.
    """
    dedupe = inbox_outcome_dedupe_key(request.event_id, request.miner_hotkey, received_at, bucket_seconds)
    return {
        "topic": topic,
        "payload": request.model_dump(mode="json"),
        "dedupe_key": dedupe,
    }


__all__ = [
    "map_submit_odds_to_miner_submission_rows",
    "map_submit_outcome_to_inbox_row",
]


