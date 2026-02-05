"""Ledger state tracking for the primary validator."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class LedgerState(Base):
    """Singleton table tracking ledger export state on the primary.

    Always contains exactly one row (id=1).
    """

    __tablename__ = "ledger_state"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        default=1,
        comment="Singleton row (always id=1)",
    )
    checkpoint_epoch: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Current checkpoint epoch (incremented on recompute)",
    )
    last_checkpoint_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When the last checkpoint was exported",
    )
    last_delta_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When the last delta was exported",
    )
    last_delta_id: Mapped[str | None] = mapped_column(
        String,
        comment="ID of the last delta exported",
    )


__all__ = ["LedgerState"]
