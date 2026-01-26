"""Retention-aware cleanup helpers for validator persistence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy import DateTime as SADateTime
from sqlalchemy import delete, literal
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import Delete

from schema.events import Event
from schema.miner import MinerSubmission
from schema.provider import ProviderQuote


# Retain quotes/submissions for two weeks after event start
EVENT_RETENTION = timedelta(weeks=2)


def provider_quote_cleanup_statement(now: Optional[datetime] = None) -> Delete:
    """Delete provider quotes older than the retention window."""

    cutoff = _ensure_utc(now) - EVENT_RETENTION
    return delete(ProviderQuote).where(ProviderQuote.ts < cutoff)


def miner_submission_cleanup_statement(now: Optional[datetime] = None) -> Delete:
    """Delete miner submissions older than the retention window."""

    cutoff = _ensure_utc(now) - EVENT_RETENTION
    return delete(MinerSubmission).where(MinerSubmission.submitted_at < cutoff)


def run_cleanup(session: Session, now: Optional[datetime] = None) -> Dict[str, int]:
    """Execute cleanup statements and return affected-row counts by table."""

    current = _ensure_utc(now)
    statements = {
        "provider_quote": provider_quote_cleanup_statement(current),
        "miner_submission": miner_submission_cleanup_statement(current),
    }

    results: Dict[str, int] = {}
    for name, stmt in statements.items():
        result = session.execute(stmt)
        results[name] = result.rowcount or 0

    return results


def _ensure_utc(value: Optional[datetime]) -> datetime:
    """Return a timezone-aware UTC datetime, defaulting to the current time."""

    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None:
        return result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _now_literal(value: datetime):
    return literal(value, type_=SADateTime(timezone=True))


__all__ = [
    "EVENT_RETENTION",
    "provider_quote_cleanup_statement",
    "miner_submission_cleanup_statement",
    "run_cleanup",
]

