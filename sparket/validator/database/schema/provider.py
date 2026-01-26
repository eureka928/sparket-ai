"""Provider quotes and closing snapshots."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, Numeric
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, price_side_enum


class ProviderQuote(Base):
    __tablename__ = "provider_quote"

    quote_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Unique identifier for each captured provider price tick",
    )
    provider_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("provider.provider_id"),
        nullable=False,
        comment="Source provider for the quote",
    )
    market_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("market.market_id", ondelete="CASCADE"),
        nullable=False,
        comment="Market the quote applies to",
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Timestamp when the quote was observed (UTC)",
    )
    side: Mapped[str] = mapped_column(
        price_side_enum,
        nullable=False,
        comment="Outcome side within the market",
    )
    odds_eu: Mapped[float] = mapped_column(
        Numeric,
        nullable=False,
        comment="Decimal odds from the provider",
    )
    imp_prob: Mapped[float] = mapped_column(
        Numeric,
        nullable=False,
        comment="Implied probability (1/odds) at observation time",
    )
    imp_prob_norm: Mapped[float | None] = mapped_column(
        Numeric,
        comment="Optional normalized probability after removing overround",
    )
    raw: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        comment="Unstructured payload as captured from the provider",
    )

    __table_args__ = (
        Index("uq_provider_quote", "provider_id", "market_id", "ts", "side", unique=True),
        Index("ix_provider_quote_market_ts", "market_id", "ts"),
        Index("ix_provider_quote_provider_ts", "provider_id", "ts"),
    )


class ProviderClosing(Base):
    __tablename__ = "provider_closing"

    provider_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("provider.provider_id"),
        primary_key=True,
        comment="Provider identifier",
    )
    market_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("market.market_id", ondelete="CASCADE"),
        primary_key=True,
        comment="Market identifier",
    )
    side: Mapped[str] = mapped_column(
        price_side_enum,
        primary_key=True,
        comment="Outcome side",
    )
    ts_close: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Timestamp of the closing quote used for scoring (UTC)",
    )
    odds_eu_close: Mapped[float] = mapped_column(
        Numeric,
        nullable=False,
        comment="Closing decimal odds",
    )
    imp_prob_close: Mapped[float] = mapped_column(
        Numeric,
        nullable=False,
        comment="Closing implied probability (raw)",
    )
    imp_prob_norm_close: Mapped[float | None] = mapped_column(
        Numeric,
        comment="Closing implied probability after normalization",
    )

