"""Score breakdown tables."""

from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, score_component_enum, task_type_enum
from .events import Event


class Score(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Internal surrogate primary key",
    )
    event_id: Mapped[str] = mapped_column(
        ForeignKey("event.event_id", ondelete="CASCADE"),
        comment="FK to event.event_id for the scored event",
    )
    miner_hotkey: Mapped[str] = mapped_column(
        String,
        comment="Miner being scored",
    )
    task_type: Mapped[str] = mapped_column(
        task_type_enum,
        comment="Task classification (odds or outcome)",
    )
    pq: Mapped[float | None] = mapped_column(
        Float,
        comment="Price quality component",
    )
    clv: Mapped[float | None] = mapped_column(
        Float,
        comment="Closing line value component",
    )
    calib: Mapped[float | None] = mapped_column(
        Float,
        comment="Calibration component",
    )
    score_event: Mapped[float | None] = mapped_column(
        Float,
        comment="Aggregated per-event score",
    )
    component: Mapped[str] = mapped_column(
        score_component_enum,
        comment="Lifecycle component (interim/close/final)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now,
        comment="UTC timestamp when this score row was created",
    )

    event: Mapped["Event"] = relationship(back_populates="scores")

    __table_args__ = (
        Index("ix_scores_event_id", "event_id"),
        Index("ix_scores_miner_hotkey", "miner_hotkey"),
        Index("ix_scores_task_type", "task_type"),
        Index("ix_scores_component", "component"),
        Index("ix_scores_score_event", "score_event"),
        Index("ix_scores_created_at", "created_at"),
        {
            "comment": "Per-event scoring breakdowns for miners",
        },
    )


__all__ = ["Score"]

