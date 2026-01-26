"""Ground truth tables: sportsbooks, bias tracking, consensus closing."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, market_kind_enum, price_side_enum


class Sportsbook(Base):
    """Reference table for sportsbooks within a provider (e.g., Pinnacle, DraftKings)."""

    __tablename__ = "sportsbook"

    sportsbook_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Internal identifier for the sportsbook",
    )
    provider_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("provider.provider_id", ondelete="RESTRICT"),
        nullable=False,
        comment="Parent provider (e.g., SportsDataIO)",
    )
    code: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        nullable=False,
        comment="Unique sportsbook code (e.g., 'PINN', 'DKNG')",
    )
    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Human-readable sportsbook name",
    )
    is_sharp: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this is a known sharp book (e.g., Pinnacle)",
    )
    active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether to include in consensus calculations",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now,
        nullable=False,
        comment="Row creation timestamp (UTC)",
    )

    __table_args__ = (
        Index("ix_sportsbook_provider", "provider_id"),
        Index("ix_sportsbook_active", "active"),
    )


class SportsbookBias(Base):
    """Per (sportsbook, sport, market_kind) bias tracking for ground truth consensus."""

    __tablename__ = "sportsbook_bias"

    sportsbook_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("sportsbook.sportsbook_id", ondelete="CASCADE"),
        primary_key=True,
        comment="Sportsbook being calibrated",
    )
    sport_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("sport.sport_id", ondelete="CASCADE"),
        primary_key=True,
        comment="Sport context for bias estimation",
    )
    market_kind: Mapped[str] = mapped_column(
        market_kind_enum,
        primary_key=True,
        comment="Market type (moneyline, spread, total)",
    )
    bias_factor: Mapped[Decimal] = mapped_column(
        Numeric(10, 6),
        nullable=False,
        default=Decimal("1.0"),
        comment="Multiplicative bias factor (1.0 = no bias)",
    )
    variance: Mapped[Decimal] = mapped_column(
        Numeric(10, 6),
        nullable=False,
        default=Decimal("0.01"),
        comment="Estimated variance of book's errors for inverse-variance weighting",
    )
    sample_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of settled outcomes used in estimation",
    )
    mean_squared_error: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 8),
        comment="MSE of book vs realized outcomes",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Version number for consensus (increments on update)",
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now,
        comment="When this bias estimate was computed (UTC)",
    )
    input_hash: Mapped[str | None] = mapped_column(
        String(64),
        comment="SHA256 hash of inputs for determinism verification",
    )

    __table_args__ = (
        Index("ix_sportsbook_bias_version", "version"),
        Index("ix_sportsbook_bias_computed", "computed_at"),
    )


class GroundTruthSnapshot(Base):
    """Time-series consensus snapshots for fair miner comparison.

    Captures ground truth at regular intervals (e.g., every 6 hours) to enable
    comparing miner predictions against book consensus AT THE TIME OF SUBMISSION,
    not just against closing line.
    """

    __tablename__ = "ground_truth_snapshot"

    market_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("market.market_id", ondelete="CASCADE"),
        primary_key=True,
        comment="Market this snapshot applies to",
    )
    side: Mapped[str] = mapped_column(
        price_side_enum,
        primary_key=True,
        comment="Outcome side (home, away, over, under, etc.)",
    )
    snapshot_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        comment="When this snapshot was captured (UTC)",
    )
    prob_consensus: Mapped[Decimal] = mapped_column(
        Numeric(10, 8),
        nullable=False,
        comment="Consensus probability from bias-weighted average",
    )
    odds_consensus: Mapped[Decimal] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        comment="Consensus decimal odds (1 / prob_consensus)",
    )
    contributing_books: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of sportsbooks contributing to this consensus",
    )
    std_dev: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 8),
        comment="Standard deviation of book probabilities",
    )
    bias_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Version of sportsbook_bias used for this consensus",
    )
    is_closing: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True if this is the final closing line snapshot",
    )

    __table_args__ = (
        Index("ix_gt_snapshot_market_ts", "market_id", "snapshot_ts"),
        Index("ix_gt_snapshot_closing", "market_id", "side", "is_closing"),
    )


class GroundTruthClosing(Base):
    """Consensus closing line per market/side derived from bias-adjusted sportsbooks.

    This is the FINAL closing line, used for CLV/CLE calculations.
    For time-matched comparisons (PSS), use GroundTruthSnapshot.
    """

    __tablename__ = "ground_truth_closing"

    market_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("market.market_id", ondelete="CASCADE"),
        primary_key=True,
        comment="Market this consensus applies to",
    )
    side: Mapped[str] = mapped_column(
        price_side_enum,
        primary_key=True,
        comment="Outcome side (home, away, over, under, etc.)",
    )
    prob_consensus: Mapped[Decimal] = mapped_column(
        Numeric(10, 8),
        nullable=False,
        comment="Consensus probability from bias-weighted average",
    )
    odds_consensus: Mapped[Decimal] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        comment="Consensus decimal odds (1 / prob_consensus)",
    )
    contributing_books: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of sportsbooks contributing to this consensus",
    )
    min_prob: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 8),
        comment="Minimum probability across contributing books",
    )
    max_prob: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 8),
        comment="Maximum probability across contributing books",
    )
    std_dev: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 8),
        comment="Standard deviation of book probabilities",
    )
    bias_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Version of sportsbook_bias used for this consensus",
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now,
        comment="When this consensus was computed (UTC)",
    )

    __table_args__ = (
        Index("ix_ground_truth_closing_computed", "computed_at"),
        Index("ix_ground_truth_closing_bias_version", "bias_version"),
    )


__all__ = [
    "Sportsbook",
    "SportsbookBias",
    "GroundTruthSnapshot",
    "GroundTruthClosing",
]

