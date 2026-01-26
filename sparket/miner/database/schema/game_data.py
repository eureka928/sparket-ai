"""Local storage for synced game data (events and markets)."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint

from .base import Base


class Event(Base):
    """Locally cached event data pulled from validators."""
    __tablename__ = "event"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_event_event_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, nullable=False, index=True)
    external_id = Column(Integer, nullable=True, index=True)  # SportsData.io GameID
    home_team = Column(String(128), nullable=True)
    away_team = Column(String(128), nullable=True)
    venue = Column(String(255), nullable=True)
    start_time_utc = Column(DateTime, nullable=False, index=True)
    status = Column(String(32), nullable=False, default="scheduled")
    league = Column(String(32), nullable=True)
    sport = Column(String(32), nullable=True)
    synced_at = Column(DateTime, nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc))


class Market(Base):
    """Locally cached market data pulled from validators."""
    __tablename__ = "market"
    __table_args__ = (
        UniqueConstraint("market_id", name="uq_market_market_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, nullable=False, index=True)
    event_id = Column(Integer, nullable=False, index=True)
    kind = Column(String(32), nullable=False)
    line = Column(Float, nullable=True)
    points_team_id = Column(Integer, nullable=True)
    synced_at = Column(DateTime, nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc))


