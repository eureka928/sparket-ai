from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, TypedDict

from .enums import MarketKind, PriceSide


class EventRow(TypedDict, total=False):
    league_id: int
    home_team_id: Optional[int]
    away_team_id: Optional[int]
    venue: Optional[str]
    start_time_utc: datetime
    status: str
    ext_ref: dict
    created_at: datetime


class MarketRow(TypedDict, total=False):
    event_id: int
    kind: MarketKind
    line: Optional[float]
    points_team_id: Optional[int]
    created_at: datetime


class ProviderQuoteRow(TypedDict, total=False):
    provider_id: int
    market_id: int
    ts: datetime
    side: PriceSide | str
    odds_eu: float
    imp_prob: float
    imp_prob_norm: Optional[float]
    raw: dict


class ProviderClosingRow(TypedDict, total=False):
    provider_id: int
    market_id: int
    side: PriceSide | str
    ts_close: datetime
    odds_eu_close: float
    imp_prob_close: float
    imp_prob_norm_close: Optional[float]


class MinerSubmissionRow(TypedDict, total=False):
    miner_id: int
    miner_hotkey: str
    market_id: int
    side: PriceSide | str
    submitted_at: datetime
    priced_at: Optional[datetime]
    odds_eu: float
    imp_prob: float
    payload: dict


class InboxRow(TypedDict, total=False):
    topic: str
    payload: dict
    dedupe_key: Optional[str]


class OutcomeRow(TypedDict, total=False):
    market_id: int
    settled_at: Optional[datetime]
    result: Optional[Literal["home", "away", "draw", "over", "under", "void", "push"]]
    score_home: Optional[int]
    score_away: Optional[int]
    details: dict


__all__ = [
    "EventRow",
    "MarketRow",
    "ProviderQuoteRow",
    "ProviderClosingRow",
    "MinerSubmissionRow",
    "InboxRow",
    "OutcomeRow",
]


