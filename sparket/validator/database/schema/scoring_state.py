"""Scoring job state and worker health tracking for resilience."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ScoringJobState(Base):
    """Job checkpointing for crash recovery and resumable batch jobs."""

    __tablename__ = "scoring_job_state"

    job_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="Unique job identifier (e.g., 'rolling_aggregates_v1')",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        comment="Job status: pending, running, completed, failed",
    )
    last_checkpoint: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="Timestamp of last checkpoint (UTC)",
    )
    checkpoint_data: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Job-specific checkpoint state for resume",
    )
    items_total: Mapped[int | None] = mapped_column(
        Integer,
        comment="Total items to process (if known)",
    )
    items_processed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of items processed so far",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When the job started (UTC)",
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When the job completed (UTC)",
    )
    next_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="Scheduled next execution time (UTC)",
    )
    error_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Consecutive error count for backoff",
    )
    last_error: Mapped[str | None] = mapped_column(
        String,
        comment="Most recent error message",
    )

    __table_args__ = (
        Index("ix_scoring_job_state_status", "status"),
        Index("ix_scoring_job_state_next_run", "next_run_at"),
    )


class ScoringWorkerHeartbeat(Base):
    """Worker process health monitoring."""

    __tablename__ = "scoring_worker_heartbeat"

    worker_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="Unique worker identifier",
    )
    pid: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="OS process ID",
    )
    hostname: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Machine hostname",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="When the worker started (UTC)",
    )
    last_heartbeat: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Most recent heartbeat timestamp (UTC)",
    )
    current_job: Mapped[str | None] = mapped_column(
        String(64),
        comment="Currently executing job ID (if any)",
    )
    jobs_completed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Total jobs completed by this worker",
    )
    jobs_failed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Total jobs failed by this worker",
    )
    memory_mb: Mapped[int | None] = mapped_column(
        Integer,
        comment="Current memory usage in MB",
    )

    __table_args__ = (
        Index("ix_scoring_worker_heartbeat_last", "last_heartbeat"),
    )


class ScoringWorkQueue(Base):
    """Work queue for parallel scoring workers.
    
    Uses row-level locking (FOR UPDATE SKIP LOCKED) for safe
    concurrent work claiming across multiple workers.
    """

    __tablename__ = "scoring_work_queue"

    work_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="Unique work item identifier (UUID)",
    )
    work_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Type of work: snapshot, outcome, rolling, skill",
    )
    chunk_key: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Chunk identifier (e.g., miner range, market batch)",
    )
    params: Mapped[str | None] = mapped_column(
        String,
        comment="JSON-encoded parameters for this work item",
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Higher priority = processed first",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default="pending",
        nullable=False,
        comment="Status: pending, claimed, completed, failed",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now,
        comment="When this work item was created (UTC)",
    )
    claimed_by: Mapped[str | None] = mapped_column(
        String(64),
        comment="Worker ID that claimed this work",
    )
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When this work was claimed (UTC)",
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When this work was completed (UTC)",
    )
    result: Mapped[str | None] = mapped_column(
        String,
        comment="JSON-encoded result data",
    )
    error: Mapped[str | None] = mapped_column(
        String(1000),
        comment="Error message if failed",
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of retry attempts",
    )

    __table_args__ = (
        Index("ix_scoring_work_queue_status", "status"),
        Index("ix_scoring_work_queue_type_status", "work_type", "status"),
        Index("ix_scoring_work_queue_priority", "priority", "created_at"),
        Index("ix_scoring_work_queue_claimed", "claimed_at"),
        Index("uq_scoring_work_queue_type_chunk", "work_type", "chunk_key", unique=True),
    )


__all__ = [
    "ScoringJobState",
    "ScoringWorkerHeartbeat",
    "ScoringWorkQueue",
]

