from __future__ import annotations

from enum import Enum


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


__all__ = ["MarketKind", "PriceSide"]


