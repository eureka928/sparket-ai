"""Shared SQLAlchemy base definitions and enums."""

from __future__ import annotations

from enum import Enum

from sqlalchemy import Enum as SAEnum, MetaData
from sqlalchemy.orm import DeclarativeBase


# Shared metadata constant so Alembic sees every table
naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=naming_convention)


class Base(DeclarativeBase):
    metadata = metadata


class MarketKind(str, Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    DRAW_NO_BET = "draw_no_bet"


class PriceSide(str, Enum):
    HOME = "home"
    AWAY = "away"
    DRAW = "draw"
    OVER = "over"
    UNDER = "under"


class MarketResult(str, Enum):
    HOME = "home"
    AWAY = "away"
    DRAW = "draw"
    OVER = "over"
    UNDER = "under"
    VOID = "void"
    PUSH = "push"


class TaskType(str, Enum):
    ODDS = "odds"
    OUTCOME = "outcome"


class ScoreComponent(str, Enum):
    INTERIM = "interim"
    CLOSE = "close"
    FINAL = "final"


# SQLAlchemy Enum instances bound to shared metadata
market_kind_enum = SAEnum(MarketKind, name="market_kind", metadata=metadata)
price_side_enum = SAEnum(PriceSide, name="price_side", metadata=metadata)
market_result_enum = SAEnum(MarketResult, name="market_result", metadata=metadata)
task_type_enum = SAEnum(TaskType, name="task_type", metadata=metadata)
score_component_enum = SAEnum(ScoreComponent, name="score_component", metadata=metadata)


__all__ = [
    "Base",
    "metadata",
    "MarketKind",
    "PriceSide",
    "MarketResult",
    "market_kind_enum",
    "price_side_enum",
    "market_result_enum",
    "TaskType",
    "ScoreComponent",
    "task_type_enum",
    "score_component_enum",
]

