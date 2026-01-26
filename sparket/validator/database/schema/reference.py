"""Reference tables: sports, leagues, teams, providers, miners."""

from __future__ import annotations

from typing import Any

from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, Numeric, String, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Sport(Base):
    __tablename__ = "sport"

    sport_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Internal identifier for a sport grouping",
    )
    code: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
        comment="Stable code (e.g. 'nba', 'mlb') used throughout the system",
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="Human-friendly sport name",
    )


class League(Base):
    __tablename__ = "league"

    league_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Internal identifier for a league or competition",
    )
    sport_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("sport.sport_id", ondelete="RESTRICT"),
        nullable=False,
        comment="Parent sport (fk to sport)",
    )
    code: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="League code (e.g. 'EPL', 'NBA')",
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="League display name",
    )
    ext_ref: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="External provider identifiers keyed by origin",
    )

    __table_args__ = (UniqueConstraint("sport_id", "code", name="uq_league_sport_code"),)


class Team(Base):
    __tablename__ = "team"

    team_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Global identifier for a team or competitor",
    )
    league_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("league.league_id", ondelete="RESTRICT"),
        nullable=False,
        comment="League that the team belongs to",
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="Team name",
    )
    abbrev: Mapped[str | None] = mapped_column(
        String,
        comment="Short code or abbreviation, if applicable",
    )
    ext_ref: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        comment="External provider identifiers keyed by origin",
    )


class Provider(Base):
    __tablename__ = "provider"

    provider_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Identifier for an odds provider or book",
    )
    code: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
        comment="Short provider code (e.g. 'PINN')",
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="Human-friendly provider name",
    )


class Miner(Base):
    __tablename__ = "miner"
    __api_expose__ = {
        "v1": {
            "read": "MinerRead",
            "write": "MinerWrite",
            "include": [
                "miner_id",
                "hotkey",
                "uid",
                "netuid",
                "active",
                "total_stake",
                "rank",
                "emission",
                "incentive",
                "consensus",
                "trust",
                "validator_trust",
                "dividends",
                "last_update",
                "validator_permit",
            ],
            "exclude": [
                "stake_dict",
                "prometheus_info",
                "axon_info",
            ],
        }
    }

    miner_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Internal surrogate key for miner records",
    )
    hotkey: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
        comment="Hotkey identifier (matches Bittensor neuron hotkey)",
    )
    coldkey: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="Coldkey address associated with the miner",
    )
    uid: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Neuron UID on the Bittensor subnet",
    )
    netuid: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Subnet network identifier",
    )
    active: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Active flag from metagraph (1 if participating)",
    )
    stake: Mapped[float] = mapped_column(
        Numeric,
        nullable=False,
        comment="Current stake balance for the hotkey",
    )
    stake_dict: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Breakdown of stake by delegator coldkey",
    )
    total_stake: Mapped[float] = mapped_column(
        Numeric,
        nullable=False,
        comment="Synced copy of total stake (for quick filtering)",
    )
    rank: Mapped[float] = mapped_column(Numeric, nullable=False, default=0, comment="Current rank metric")
    emission: Mapped[float] = mapped_column(Numeric, nullable=False, default=0, comment="Emission rate")
    incentive: Mapped[float] = mapped_column(Numeric, nullable=False, default=0, comment="Incentive score")
    consensus: Mapped[float] = mapped_column(Numeric, nullable=False, default=0, comment="Consensus score")
    trust: Mapped[float] = mapped_column(Numeric, nullable=False, default=0, comment="Trust value from metagraph")
    validator_trust: Mapped[float] = mapped_column(Numeric, nullable=False, default=0, comment="Validator trust component")
    dividends: Mapped[float] = mapped_column(Numeric, nullable=False, default=0, comment="Dividends metric")
    last_update: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Block height of last metagraph update",
    )
    validator_permit: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether the miner currently has validator permit",
    )
    prometheus_info: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        default=None,
        nullable=True,
        comment="Optional Prometheus endpoint metadata",
    )
    axon_info: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        default=None,
        nullable=True,
        comment="Axon configuration snapshot (NeuronInfoLite.axon_info)",
    )
    pruning_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Score used for pruning decisions",
    )
    is_null: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Marker for placeholder/null neuron entries",
    )

    __table_args__ = (
        Index("ix_miner_netuid_uid", "netuid", "uid", unique=True),
        UniqueConstraint("miner_id", "hotkey", name="uq_miner_id_hotkey"),
    )


__all__ = ["Sport", "League", "Team", "Provider", "Miner"]

