"""Event and market structures."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, market_kind_enum


class Event(Base):
    __tablename__ = "event"
    __api_expose__ = {
        "v1": {
            "read": "EventRead",
            "write": "EventWrite",
            "include": [
                "event_id",
                "league_id",
                "home_team_id",
                "away_team_id",
                "start_time_utc",
                "status",
            ],
            "exclude": [
                "ext_ref",
            ],
        }
    }

    event_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Identifier for a scheduled sporting event",
    )
    league_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("league.league_id"),
        nullable=False,
        comment="League the event belongs to",
    )
    home_team_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("team.team_id"),
        comment="Home team reference",
    )
    away_team_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("team.team_id"),
        comment="Away team reference",
    )
    venue: Mapped[Optional[str]] = mapped_column(String, comment="Venue or location")
    start_time_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Scheduled start time in UTC",
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="scheduled",
        comment="Lifecycle state (scheduled/in_play/finished/void)",
    )
    ext_ref: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        comment="Provider event identifiers or metadata",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now,
        nullable=False,
        comment="Row creation timestamp (UTC)",
    )

    __table_args__ = (
        Index("ix_event_league_start", "league_id", "start_time_utc"),
        Index("ix_event_status_start", "status", "start_time_utc"),
    )


class Market(Base):
    __tablename__ = "market"

    market_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Identifier for a bettable market on an event",
    )
    event_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("event.event_id", ondelete="CASCADE"),
        nullable=False,
        comment="Parent event",
    )
    kind: Mapped[str] = mapped_column(
        market_kind_enum,
        nullable=False,
        comment="Market family (moneyline/spread/total/etc.)",
    )
    line: Mapped[Optional[float]] = mapped_column(
        Numeric,
        comment="Line or total if applicable (NULL for moneyline)",
    )
    points_team_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("team.team_id"),
        comment="Team the handicap applies to (for spreads)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now,
        nullable=False,
        comment="When the market row was created (UTC)",
    )

    __table_args__ = (
        Index("ix_market_event_kind", "event_id", "kind"),
    )


Index(
    "uq_market_event_kind_line_team",
    Market.event_id,
    Market.kind,
    func.coalesce(Market.line, 0),
    func.coalesce(Market.points_team_id, 0),
    unique=True,
)


__all__ = ["Event", "Market"]

