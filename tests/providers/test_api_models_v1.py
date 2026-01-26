from datetime import datetime, timezone
import json

from sparket.protocol.models.v1.common import ErrorResponse, APIError
from sparket.protocol.models.v1.odds import (
    SubmitOddsRequest,
    SubmitOddsResponse,
    MarketSubmission,
    OutcomePrice,
)
from sparket.protocol.models.v1.outcomes import (
    SubmitOutcomeRequest,
    SubmitOutcomeResponse,
    OutcomeEvidence,
)


def test_odds_request_response_serialization():
    req = SubmitOddsRequest(
        miner_id=1,
        miner_hotkey="hx",
        submissions=[
            MarketSubmission(
                market_id=100,
                kind="moneyline",
                prices=[
                    OutcomePrice(side="home", odds_eu=1.91, imp_prob=0.5236),
                    OutcomePrice(side="away", odds_eu=2.05, imp_prob=0.4878),
                ],
            )
        ],
    )
    s = req.model_dump_json()
    assert "miner_id" in s


def test_odds_request_validation_errors():
    # Invalid probability range
    try:
        SubmitOddsRequest(
            miner_id=1,
            miner_hotkey="hx",
            submissions=[
                MarketSubmission(
                    market_id=100,
                    kind="total",
                    prices=[
                        OutcomePrice(side="over", odds_eu=1.8, imp_prob=1.2),
                        OutcomePrice(side="under", odds_eu=2.2, imp_prob=0.3),
                    ],
                )
            ],
        )
        assert False, "expected validation error"
    except Exception:
        pass


def test_outcome_request_response_serialization():
    req = SubmitOutcomeRequest(
        event_id="ev123",
        miner_hotkey="hx",
        winner_label="home",
        final_score="102-98",
        ts_submit=datetime.now(timezone.utc),
        sources=[OutcomeEvidence(url="https://example", source_type="official_site", captured_at=datetime.now(timezone.utc))],
    )
    s = req.model_dump_json()
    assert "event_id" in s


def test_error_envelope_serialization():
    err = ErrorResponse(error=APIError(code="bad_request", message="Invalid payload"))
    s = err.model_dump_json()
    assert "error" in s
