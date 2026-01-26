from __future__ import annotations

import datetime as dt

from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint

from .base import Base


class ValidatorEndpoint(Base):
    __tablename__ = "validator_endpoint"
    __table_args__ = (
        UniqueConstraint("hotkey", name="uq_validator_endpoint_hotkey"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    hotkey = Column(String(64), nullable=False)
    host = Column(String(255), nullable=True)
    port = Column(Integer, nullable=True)
    url = Column(String(512), nullable=True)
    token = Column(String(512), nullable=True)
    last_seen = Column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc))
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: dt.datetime.now(dt.timezone.utc),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc),
    )

