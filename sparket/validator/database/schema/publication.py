"""Outbox/inbox publication tables."""

from __future__ import annotations

from datetime import datetime

from typing import Any

from sqlalchemy import Boolean, DateTime, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Outbox(Base):
    __tablename__ = "outbox"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Internal surrogate primary key",
    )
    topic: Mapped[str] = mapped_column(
        String,
        comment="Message topic consumed by downstream delivery",
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        comment="Exactly-once payload awaiting transport",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now,
        comment="UTC enqueue timestamp",
    )
    sent: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="Delivery flag set by transport",
    )

    __table_args__ = (
        Index("ix_outbox_topic", "topic"),
        Index("ix_outbox_created_at", "created_at"),
        Index("ix_outbox_sent", "sent"),
        {
            "comment": "Outbox pattern table ensuring exactly-once publication",
        },
    )


class Inbox(Base):
    __tablename__ = "inbox"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Internal surrogate primary key",
    )
    topic: Mapped[str] = mapped_column(
        String,
        comment="Inbound topic or acknowledgment type",
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        comment="Inbound message payload awaiting processing",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now,
        comment="UTC timestamp when the message was received",
    )
    processed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="Indicates whether the message has been handled",
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="Timestamp when the message was processed (UTC)",
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of processing attempts",
    )
    last_error: Mapped[str | None] = mapped_column(
        String,
        comment="Most recent processing error (if any)",
    )
    dedupe_key: Mapped[str | None] = mapped_column(
        String,
        default=None,
        comment="Idempotency key for inbound messages",
    )

    __table_args__ = (
        Index("ix_inbox_topic", "topic"),
        Index("ix_inbox_created_at", "created_at"),
        Index("ix_inbox_processed", "processed"),
        Index("ix_inbox_processed_at", "processed_at"),
        {
            "comment": "Inbox table for idempotent processing of acknowledgements and callbacks",
        },
    )


__all__ = ["Outbox", "Inbox"]

