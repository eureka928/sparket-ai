"""Closing line selection utilities.

Default strategy: select the last quote strictly before the event start
timestamp. Consumers must normalize all timestamps to the same timezone
(UTC recommended) prior to calling.

If your subscription exposes an explicit "closing" field or endpoint,
prefer that and bypass this selector.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Optional

from .mapping import map_moneyline_quotes, map_spread_quotes, map_total_quotes


def select_closing_quotes(
    pregame_quotes: Iterable[dict],
    start_time_utc: datetime,
) -> List[dict]:
    """Select closing quotes as the last quote strictly before event start.

    Input rows are provider_quote-shaped dicts with keys: provider_id, market_id,
    ts, side, odds_eu, imp_prob, imp_prob_norm, raw.
    """
    t0 = start_time_utc
    best_by_side: dict[str, dict] = {}
    for q in pregame_quotes:
        ts = q.get("ts")
        if ts is None or ts >= t0:
            continue
        side = q["side"]
        prev = best_by_side.get(side)
        if prev is None or ts > prev["ts"]:
            best_by_side[side] = q
    return list(best_by_side.values())


def closing_rows_from_odds(
    odds_rows: Iterable[dict],
    start_time_utc: datetime,
) -> List[dict]:
    """Transform quote rows into provider_closing rows using last pre-start quotes."""
    closers = select_closing_quotes(odds_rows, start_time_utc)
    rows: List[dict] = []
    for q in closers:
        rows.append(
            {
                "provider_id": q["provider_id"],
                "market_id": q["market_id"],
                "side": q["side"],
                "ts_close": q["ts"],
                "odds_eu_close": q["odds_eu"],
                "imp_prob_close": q["imp_prob"],
                "imp_prob_norm_close": q.get("imp_prob_norm"),
            }
        )
    return rows


__all__ = ["select_closing_quotes", "closing_rows_from_odds"]


