from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import bittensor as bt
from sqlalchemy import text

from sparket.shared.rows import ProviderQuoteRow
from sparket.providers.sportsdataio.mapping import normalize_quotes_by_market_ts


_INSERT_PROVIDER_QUOTE = text(
    """
    INSERT INTO provider_quote (
        provider_id, market_id, ts, side, odds_eu, imp_prob, imp_prob_norm, raw
    ) VALUES (
        :provider_id, :market_id, :ts, :side, :odds_eu, :imp_prob, :imp_prob_norm, :raw
    ) ON CONFLICT (provider_id, market_id, ts, side) DO NOTHING
    """
)

_SELECT_LATEST_QUOTES_BEFORE = text(
    """
    SELECT q.provider_id, q.market_id, q.side, MAX(q.ts) AS ts
    FROM provider_quote q
    WHERE q.provider_id = :provider_id AND q.ts <= :close_ts
    GROUP BY q.provider_id, q.market_id, q.side
    """
)

_SELECT_LATEST_QUOTES_BEFORE_FOR_EVENT = text(
    """
    SELECT q.provider_id, q.market_id, q.side, MAX(q.ts) AS ts
    FROM provider_quote q
    JOIN market m ON m.market_id = q.market_id
    WHERE q.provider_id = :provider_id AND m.event_id = :event_id AND q.ts <= :close_ts
    GROUP BY q.provider_id, q.market_id, q.side
    """
)

_SELECT_QUOTE_AT_TS = text(
    """
    SELECT odds_eu, imp_prob, imp_prob_norm
    FROM provider_quote
    WHERE provider_id = :provider_id AND market_id = :market_id AND side = :side AND ts = :ts
    LIMIT 1
    """
)

_UPSERT_PROVIDER_CLOSING = text(
    """
    INSERT INTO provider_closing (
        provider_id, market_id, side, ts_close, odds_eu_close, imp_prob_close, imp_prob_norm_close
    ) VALUES (
        :provider_id, :market_id, :side, :ts_close, :odds_eu_close, :imp_prob_close, :imp_prob_norm_close
    ) ON CONFLICT (provider_id, market_id, side)
    DO UPDATE SET ts_close = EXCLUDED.ts_close,
                  odds_eu_close = EXCLUDED.odds_eu_close,
                  imp_prob_close = EXCLUDED.imp_prob_close,
                  imp_prob_norm_close = EXCLUDED.imp_prob_norm_close
    """
)


async def insert_provider_quotes(*, database: Any, quotes: Iterable[ProviderQuoteRow]) -> int:
    """
    Normalize implied probabilities per (market_id, ts) and insert quotes idempotently.
    Returns number of attempted inserts.
    Uses batch inserts for performance.
    """
    import json
    
    normalized = normalize_quotes_by_market_ts(list(quotes))
    if not normalized:
        return 0
    
    # Build batch of parameters
    batch_params = []
    for row in normalized:
        raw_data = row.get("raw", {})
        raw_json = json.dumps(raw_data, default=str) if isinstance(raw_data, dict) else raw_data
        side = row["side"]
        if isinstance(side, str):
            side = side.upper()
        batch_params.append({
            "provider_id": row["provider_id"],
            "market_id": row["market_id"],
            "ts": row["ts"],
            "side": side,
            "odds_eu": row["odds_eu"],
            "imp_prob": row.get("imp_prob"),
            "imp_prob_norm": row.get("imp_prob_norm"),
            "raw": raw_json,
        })
    
    # Insert in batches of 500 for efficiency
    BATCH_SIZE = 500
    inserted = 0
    for i in range(0, len(batch_params), BATCH_SIZE):
        batch = batch_params[i : i + BATCH_SIZE]
        try:
            await database.write_many(_INSERT_PROVIDER_QUOTE, params_list=batch)
            inserted += len(batch)
        except Exception as e:
            # Fallback to individual inserts if batch fails
            bt.logging.warning({"provider_quote_batch_error": str(e), "batch_size": len(batch), "falling_back": True})
            for params in batch:
                try:
                    await database.write(_INSERT_PROVIDER_QUOTE, params=params)
                    inserted += 1
                except Exception as e2:
                    bt.logging.warning({"provider_quote_insert_error": str(e2), "market_id": params["market_id"]})
    return inserted


async def upsert_provider_closing(*, database: Any, provider_id: int, close_ts: datetime) -> int:
    """
    For each (market_id, side), find the latest quote at or before close_ts and upsert provider_closing.
    Returns count of upserts performed.
    """
    try:
        latest_rows = await database.read(_SELECT_LATEST_QUOTES_BEFORE, params={"provider_id": provider_id, "close_ts": close_ts}, mappings=True)
    except Exception as e:
        bt.logging.warning({"select_latest_quotes_error": str(e)})
        return 0

    upserts = 0
    for r in latest_rows:
        market_id = r["market_id"]
        side = r["side"]
        ts = r["ts"]
        try:
            quote_rows = await database.write(_SELECT_QUOTE_AT_TS, params={
                "provider_id": provider_id,
                "market_id": market_id,
                "side": side,
                "ts": ts,
            }, return_rows=True, mappings=True)
        except Exception as e:
            bt.logging.warning({"select_quote_at_ts_error": str(e)})
            continue
        if not quote_rows:
            continue
        q = quote_rows[0]
        params = {
            "provider_id": provider_id,
            "market_id": market_id,
            "side": side,
            "ts_close": ts,
            "odds_eu_close": q["odds_eu"],
            "imp_prob_close": q["imp_prob"],
            "imp_prob_norm_close": q.get("imp_prob_norm"),
        }
        try:
            await database.write(_UPSERT_PROVIDER_CLOSING, params=params)
            upserts += 1
        except Exception as e:
            bt.logging.warning({"upsert_provider_closing_error": str(e), "market_id": market_id, "side": side})
    return upserts


async def upsert_provider_closing_for_event(*, database: Any, provider_id: int, event_id: int, close_ts: datetime) -> int:
    """
    Upsert provider_closing rows for a single event using last quotes at or before close_ts.
    """
    try:
        latest_rows = await database.read(
            _SELECT_LATEST_QUOTES_BEFORE_FOR_EVENT,
            params={"provider_id": provider_id, "event_id": event_id, "close_ts": close_ts},
            mappings=True,
        )
    except Exception as e:
        bt.logging.warning({"select_latest_quotes_event_error": str(e), "event_id": event_id})
        return 0

    upserts = 0
    for r in latest_rows:
        market_id = r["market_id"]
        side = r["side"]
        ts = r["ts"]
        try:
            quote_rows = await database.write(
                _SELECT_QUOTE_AT_TS,
                params={
                    "provider_id": provider_id,
                    "market_id": market_id,
                    "side": side,
                    "ts": ts,
                },
                return_rows=True,
                mappings=True,
            )
        except Exception as e:
            bt.logging.warning({"select_quote_at_ts_error": str(e)})
            continue
        if not quote_rows:
            continue
        q = quote_rows[0]
        params = {
            "provider_id": provider_id,
            "market_id": market_id,
            "side": side,
            "ts_close": ts,
            "odds_eu_close": q["odds_eu"],
            "imp_prob_close": q["imp_prob"],
            "imp_prob_norm_close": q.get("imp_prob_norm"),
        }
        try:
            await database.write(_UPSERT_PROVIDER_CLOSING, params=params)
            upserts += 1
        except Exception as e:
            bt.logging.warning({"upsert_provider_closing_error": str(e), "market_id": market_id, "side": side})
    return upserts
