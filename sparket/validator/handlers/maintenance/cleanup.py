from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import bittensor as bt
from sqlalchemy import text

from sparket.validator.config.scoring_params import get_scoring_params


_DELETE_PROVIDER_QUOTE = text(
    """
    DELETE FROM provider_quote
    WHERE ts < :cutoff
    """
)

_DELETE_PROVIDER_CLOSING = text(
    """
    DELETE FROM provider_closing
    WHERE ts_close < :cutoff
    """
)

_DELETE_MINER_SUBMISSION = text(
    """
    DELETE FROM miner_submission
    WHERE submitted_at < :cutoff
    """
)

_DELETE_SUBMISSION_VS_CLOSE = text(
    """
    DELETE FROM submission_vs_close
    WHERE computed_at < :cutoff
    """
)

_DELETE_SUBMISSION_OUTCOME_SCORE = text(
    """
    DELETE FROM submission_outcome_score
    WHERE settled_at < :cutoff
    """
)

_DELETE_GT_SNAPSHOT = text(
    """
    DELETE FROM ground_truth_snapshot
    WHERE snapshot_ts < :cutoff
    """
)

_DELETE_GT_CLOSING = text(
    """
    DELETE FROM ground_truth_closing
    WHERE computed_at < :cutoff
    """
)

_DELETE_INBOX = text(
    """
    DELETE FROM inbox
    WHERE processed = true
      AND COALESCE(processed_at, created_at) < :cutoff
    """
)

_DELETE_OUTBOX = text(
    """
    DELETE FROM outbox
    WHERE sent = true
      AND created_at < :cutoff
    """
)

_DELETE_JOB_STATE = text(
    """
    DELETE FROM scoring_job_state
    WHERE status IN ('completed', 'failed')
      AND completed_at < :cutoff
    """
)

_DELETE_WORKER_HEARTBEAT = text(
    """
    DELETE FROM scoring_worker_heartbeat
    WHERE last_heartbeat < :cutoff
    """
)

_DELETE_WORK_QUEUE = text(
    """
    DELETE FROM scoring_work_queue
    WHERE status IN ('completed', 'failed')
      AND completed_at < :cutoff
    """
)


async def run_cleanup_if_due(*, validator: Any, database: Any) -> Dict[str, int]:
    """
    Step-based cleanup scheduler to enforce retention windows.
    Returns dict of deleted row counts.
    """
    core_cfg = getattr(validator, "app_config", None)
    timers = getattr(getattr(core_cfg, "core", None), "timers", None)
    steps_interval = 200
    if timers is not None:
        try:
            steps_interval = int(getattr(timers, "cleanup_steps", steps_interval))
        except Exception:
            pass

    if steps_interval <= 0 or (validator.step % steps_interval != 0):
        return {}

    now = datetime.now(timezone.utc)
    retention = get_scoring_params().retention

    statements = [
        ("provider_quote", _DELETE_PROVIDER_QUOTE, now - timedelta(days=retention.provider_quote_days)),
        ("provider_closing", _DELETE_PROVIDER_CLOSING, now - timedelta(days=retention.provider_closing_days)),
        ("miner_submission", _DELETE_MINER_SUBMISSION, now - timedelta(days=retention.miner_submission_days)),
        ("submission_vs_close", _DELETE_SUBMISSION_VS_CLOSE, now - timedelta(days=retention.submission_vs_close_days)),
        ("submission_outcome_score", _DELETE_SUBMISSION_OUTCOME_SCORE, now - timedelta(days=retention.submission_outcome_score_days)),
        ("ground_truth_snapshot", _DELETE_GT_SNAPSHOT, now - timedelta(days=retention.ground_truth_snapshot_days)),
        ("ground_truth_closing", _DELETE_GT_CLOSING, now - timedelta(days=retention.ground_truth_closing_days)),
        ("inbox", _DELETE_INBOX, now - timedelta(days=retention.inbox_processed_days)),
        ("outbox", _DELETE_OUTBOX, now - timedelta(days=retention.outbox_sent_days)),
        ("scoring_job_state", _DELETE_JOB_STATE, now - timedelta(days=retention.scoring_job_state_days)),
        ("scoring_worker_heartbeat", _DELETE_WORKER_HEARTBEAT, now - timedelta(days=retention.scoring_worker_heartbeat_days)),
        ("scoring_work_queue", _DELETE_WORK_QUEUE, now - timedelta(days=retention.scoring_work_queue_days)),
    ]

    deleted: Dict[str, int] = {}
    for name, stmt, cutoff in statements:
        try:
            deleted[name] = int(await database.write(stmt, params={"cutoff": cutoff}))
        except Exception as exc:
            bt.logging.debug({"cleanup_error": {"table": name, "error": str(exc)}})
            deleted[name] = 0

    if any(count > 0 for count in deleted.values()):
        bt.logging.info({"cleanup_deleted": deleted})

    return deleted


__all__ = ["run_cleanup_if_due"]
