"""Market outcomes."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Numeric, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, market_result_enum


class Outcome(Base):
    __tablename__ = "outcome"

    outcome_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Primary key for the market outcome record",
    )
    market_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("market.market_id", ondelete="CASCADE"),
        nullable=False,
        comment="Market this outcome resolves",
    )
    settled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When the market was settled (if known, UTC)",
    )
    result: Mapped[str | None] = mapped_column(
        market_result_enum,
        comment="Final result (home/away/draw/over/under/etc.)",
    )
    score_home: Mapped[int | None] = mapped_column(
        Numeric,
        comment="Home team final score snapshot",
    )
    score_away: Mapped[int | None] = mapped_column(
        Numeric,
        comment="Away team final score snapshot",
    )
    details: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        comment="Additional settlement data or evidence",
    )

    __table_args__ = (UniqueConstraint("market_id", name="uq_outcome_market"),)


__all__ = ["Outcome"]


