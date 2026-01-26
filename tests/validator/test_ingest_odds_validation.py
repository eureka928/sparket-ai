from datetime import datetime, timezone

from sparket.validator.handlers.ingest.ingest_odds import _coerce_submit_odds_to_rows


def _make_payload(odds_eu: float, imp_prob=None):
    return {
        "submissions": [
            {
                "market_id": 101,
                "kind": "moneyline",
                "priced_at": datetime.now(timezone.utc).isoformat(),
                "prices": [
                    {
                        "side": "home",
                        "odds_eu": odds_eu,
                        "imp_prob": imp_prob,
                    }
                ],
            }
        ]
    }


def test_imp_prob_is_derived_from_odds():
    payload = _make_payload(odds_eu=2.0, imp_prob=0.9)
    rows = _coerce_submit_odds_to_rows(
        payload=payload,
        miner_id=1,
        miner_hotkey="hk",
        received_at=datetime.now(timezone.utc),
        valid_market_ids={101},
        bucket_seconds=60,
        priced_at_tolerance_sec=300,
    )
    assert len(rows) == 1
    assert abs(rows[0]["imp_prob"] - 0.5) < 1e-9


def test_imp_prob_missing_uses_odds():
    payload = _make_payload(odds_eu=4.0, imp_prob=None)
    rows = _coerce_submit_odds_to_rows(
        payload=payload,
        miner_id=1,
        miner_hotkey="hk",
        received_at=datetime.now(timezone.utc),
        valid_market_ids={101},
        bucket_seconds=60,
        priced_at_tolerance_sec=300,
    )
    assert len(rows) == 1
    assert abs(rows[0]["imp_prob"] - 0.25) < 1e-9


def test_invalid_imp_prob_does_not_block_computed():
    payload = _make_payload(odds_eu=3.0, imp_prob="nan")
    rows = _coerce_submit_odds_to_rows(
        payload=payload,
        miner_id=1,
        miner_hotkey="hk",
        received_at=datetime.now(timezone.utc),
        valid_market_ids={101},
        bucket_seconds=60,
        priced_at_tolerance_sec=300,
    )
    assert len(rows) == 1
    assert abs(rows[0]["imp_prob"] - (1.0 / 3.0)) < 1e-9
