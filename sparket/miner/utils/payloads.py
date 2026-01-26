from __future__ import annotations

from datetime import datetime
from typing import Dict


def _deterministic_probs(miner_hotkey: str, market_id: int) -> Dict[str, float]:
    h = (hash(miner_hotkey) ^ int(market_id)) & 0xFFFF
    step = (h % 6) * 0.01
    p_home = 0.45 + step
    p_away = 0.55 - step
    return {"home": p_home, "away": p_away}


def _probs_to_eu_odds(probs: Dict[str, float]) -> Dict[str, float]:
    return {k: round(1.0 / max(1e-9, v), 2) for k, v in probs.items()}


def build_submit_odds_payload(
    *,
    miner_id: int,
    miner_hotkey: str,
    market_id: int,
    kind: str,
    token: str | None,
    now: datetime,
) -> Dict[str, object]:
    probs = _deterministic_probs(miner_hotkey, market_id)
    odds = _probs_to_eu_odds(probs)
    payload: Dict[str, object] = {
        "miner_id": int(miner_id),
        "miner_hotkey": miner_hotkey,
        "submissions": [
            {
                "market_id": int(market_id),
                "kind": kind,
                "priced_at": now,
                "prices": [
                    {"side": "home", "odds_eu": odds["home"], "imp_prob": probs["home"]},
                    {"side": "away", "odds_eu": odds["away"], "imp_prob": probs["away"]},
                ],
            }
        ],
    }
    if token:
        payload["token"] = token
    return payload


def build_submit_outcome_payload(
    *,
    event_id: str,
    miner_hotkey: str,
    token: str | None,
    now: datetime,
) -> Dict[str, object]:
    winner = ["home", "away"][((hash(event_id) & 0xFFFFFFFF) % 2)]
    score = "1-0" if winner == "home" else "0-1"
    payload: Dict[str, object] = {
        "event_id": event_id,
        "miner_hotkey": miner_hotkey,
        "winner_label": winner,
        "final_score": score,
        "ts_submit": now,
        "sources": [
            {
                "url": "https://example.com/official",
                "source_type": "official_site",
                "captured_at": now,
            }
        ],
    }
    if token:
        payload["token"] = token
    return payload


