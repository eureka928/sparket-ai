"""SportsDataIO → storage mapping layer.

This module adapts provider payloads to storage row contracts declared in
`sparket.shared.rows`. It deliberately contains no API contract shapes.

Key policies:
- Spread points team = team that receives points (line > 0 → home; line < 0 → away).
- Overround normalization groups by (market_id, ts).
- Closing line defaults to last pre-start quote (see `closing.py`).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from .types import Game, GameOdds, GameOddsSet, Location, MoneylinePrice, SpreadPrice, TotalPrice, Outcome
from .types import american_to_decimal
from sparket.shared.probability import eu_to_implied_prob
from .enums import GameStatus
from sparket.shared.enums import MarketKind
from sparket.shared.rows import (
    EventRow,
    MarketRow,
    ProviderQuoteRow,
    OutcomeRow,
)


def _status_to_event_status(status: Optional[GameStatus]) -> str:
    if status is None:
        return "scheduled"
    mapping = {
        GameStatus.SCHEDULED: "scheduled",
        GameStatus.IN_PROGRESS: "in_play",
        GameStatus.FINAL: "finished",
        GameStatus.POSTPONED: "postponed",
        GameStatus.CANCELED: "canceled",
    }
    return mapping.get(status, "scheduled")


def _normalize_start_time(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize datetime to UTC with timezone info for DB storage."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume naive datetimes are UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def map_game_to_event(game: Game, league_id: int, home_team_id: int | None = None, away_team_id: int | None = None) -> EventRow:
    return {
        "league_id": league_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "venue": (game.location.name if game.location else None),
        "start_time_utc": _normalize_start_time(game.date_time),
        "status": _status_to_event_status(game.status),
        "created_at": datetime.now(timezone.utc),
        "ext_ref": {
            "sportsdataio": {
                "GameID": game.game_id,
                "HomeTeam": game.home_team,
                "AwayTeam": game.away_team,
                "Season": game.season,
                "SeasonType": game.season_type.value if game.season_type else None,
                "Week": game.week,
            }
        },
    }


def map_moneyline_quotes(odds: GameOdds, provider_id: int, market_id: int, ts: Optional[datetime]) -> Iterable[ProviderQuoteRow]:
    ts_val = ts or odds.updated
    prices = odds.moneyline
    dec = prices.as_decimal()
    for side in ("home", "away", "draw"):
        d = dec.get(side)
        if d is None:
            continue
        imp = eu_to_implied_prob(d) if d is not None else None
        yield {
            "provider_id": provider_id,
            "market_id": market_id,
            "ts": ts_val,
            "side": side,
            "odds_eu": d,
            "imp_prob": imp,
            "imp_prob_norm": None,  # filled after per-market normalization
            "raw": {
                "sportsbook": odds.sportsbook,
                "updated": odds.updated,
                "source": {"moneyline": {side: getattr(prices, side)}},
            },
        }


def map_spread_quotes(odds: GameOdds, provider_id: int, market_id: int, ts: Optional[datetime]) -> Iterable[ProviderQuoteRow]:
    ts_val = ts or odds.updated
    prices = odds.spread
    dec = prices.as_decimal()
    for side in ("home", "away"):
        d = dec.get(side)
        if d is None:
            continue
        imp = eu_to_implied_prob(d) if d is not None else None
        yield {
            "provider_id": provider_id,
            "market_id": market_id,
            "ts": ts_val,
            "side": side,
            "odds_eu": d,
            "imp_prob": imp,
            "imp_prob_norm": None,
            "raw": {
                "sportsbook": odds.sportsbook,
                "updated": odds.updated,
                "point_spread": prices.point_spread,
                "source": {"spread": {side: getattr(prices, side)}},
            },
        }


def map_total_quotes(odds: GameOdds, provider_id: int, market_id: int, ts: Optional[datetime]) -> Iterable[ProviderQuoteRow]:
    ts_val = ts or odds.updated
    prices = odds.total
    dec = prices.as_decimal()
    for side in ("over", "under"):
        d = dec.get(side)
        if d is None:
            continue
        imp = eu_to_implied_prob(d) if d is not None else None
        yield {
            "provider_id": provider_id,
            "market_id": market_id,
            "ts": ts_val,
            "side": side,
            "odds_eu": d,
            "imp_prob": imp,
            "imp_prob_norm": None,
            "raw": {
                "sportsbook": odds.sportsbook,
                "updated": odds.updated,
                "total_points": prices.total_points,
                "source": {"total": {side: getattr(prices, side)}},
            },
        }


def map_outcome_to_row(outcome: Outcome, market_id: int) -> OutcomeRow:
    result = None
    if outcome.winner:
        # Map canonical winner labels to our enums
        wl = outcome.winner.lower()
        if wl in ("home", "away", "draw", "over", "under"):
            result = wl
    return {
        "market_id": market_id,
        "settled_at": outcome.finalized_at,
        "result": result,
        "score_home": outcome.home_score,
        "score_away": outcome.away_score,
        "details": {},
    }


def resolve_moneyline_result(event_home_key: str, event_away_key: str, winner_label: str) -> Optional[str]:
    wl = winner_label.lower()
    if wl in ("home", "away", "draw"):
        return wl
    # If provider returns team keys/names, infer side
    if winner_label == event_home_key:
        return "home"
    if winner_label == event_away_key:
        return "away"
    return None


def resolve_total_result(total_points: float, home_score: Optional[int], away_score: Optional[int]) -> Optional[str]:
    if home_score is None or away_score is None:
        return None
    s = float(home_score + away_score)
    if s > total_points:
        return "over"
    if s < total_points:
        return "under"
    return "push"  # exact match


def resolve_spread_result(line: float, points_team_is_home: bool, home_score: Optional[int], away_score: Optional[int]) -> Optional[str]:
    if home_score is None or away_score is None:
        return None
    # Apply handicap to points_team side
    adj_home = float(home_score)
    adj_away = float(away_score)
    if points_team_is_home:
        adj_home += line
    else:
        adj_away += line
    if adj_home > adj_away:
        return "home"
    if adj_home < adj_away:
        return "away"
    return "push"


def ensure_markets_for_event(
    event_id: int,
    moneyline: bool = True,
    spread_lines: Optional[List[float]] = None,
    total_points_list: Optional[List[float]] = None,
    points_team_policy: str = "receives_points",  # policy: points_team is team receiving points
    home_team_id: Optional[int] = None,
    away_team_id: Optional[int] = None,
) -> List[MarketRow]:
    """Return market rows to upsert for an event based on available lines.

    - Moneyline (no line)
    - Spread (line required)
    - Total (total_points required)

    Upsert uniqueness should be enforced externally on (event, kind, line, points_team)
    as per docs/validator_schema/events.py.

    Spread points_team policy (documented and enforced):
    - policy='receives_points' (default): if line > 0, home receives the points → points_team_id=home;
      if line < 0, away receives the points → points_team_id=away. Zero line yields None.
    - This assumes provider PointSpread is quoted relative to the home team, which is the
      convention for SportsDataIO.
    """
    rows: List[dict] = []
    if moneyline:
        rows.append({
            "event_id": event_id,
            "kind": MarketKind.MONEYLINE,
            "line": None,
            "points_team_id": None,
        })
    if spread_lines:
        for line in spread_lines:
            points_team_id = None
            if points_team_policy == "receives_points" and home_team_id and away_team_id:
                if line > 0:
                    points_team_id = home_team_id
                elif line < 0:
                    points_team_id = away_team_id
            rows.append({
                "event_id": event_id,
                "kind": MarketKind.SPREAD,
                "line": line,
                "points_team_id": points_team_id,
            })
    if total_points_list:
        for total in total_points_list:
            rows.append({
                "event_id": event_id,
                "kind": MarketKind.TOTAL,
                "line": total,
                "points_team_id": None,
            })
    return rows


def normalize_quotes_by_market_ts(quotes: Iterable[ProviderQuoteRow]) -> List[ProviderQuoteRow]:
    """Compute normalized implied probabilities per (market_id, ts), removing overround.

    Assumes each quote has 'market_id', 'ts', and raw 'imp_prob'. Writes 'imp_prob_norm'.
    """
    grouped: Dict[Tuple[int, datetime], List[dict]] = {}
    for q in quotes:
        key = (q["market_id"], q["ts"]) if q.get("ts") is not None else (q["market_id"], None)  # type: ignore
        grouped.setdefault(key, []).append(q)
    out: List[dict] = []
    for _, block in grouped.items():
        total = sum((p.get("imp_prob") or 0.0) for p in block)
        if total and total > 0:
            for p in block:
                imp = p.get("imp_prob") or 0.0
                p["imp_prob_norm"] = float(imp) / float(total)
                out.append(p)
        else:
            out.extend(block)
    return out


__all__ = [
    "map_game_to_event",
    "map_moneyline_quotes",
    "map_spread_quotes",
    "map_total_quotes",
    "map_outcome_to_row",
    "ensure_markets_for_event",
]


