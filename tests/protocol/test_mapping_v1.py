"""Tests for protocol/mapping/v1.py - API to storage mapping."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sparket.protocol.mapping.v1 import (
    _ensure_imp_prob,
    map_submit_odds_to_miner_submission_rows,
    map_submit_outcome_to_inbox_row,
)
from sparket.protocol.models.v1.odds import (
    SubmitOddsRequest,
    MarketSubmission,
    OutcomePrice,
)
from sparket.protocol.models.v1.outcomes import SubmitOutcomeRequest


class TestEnsureImpProb:
    """Tests for _ensure_imp_prob helper function."""
    
    def test_returns_provided_imp_prob(self):
        """Returns provided imp_prob if not None."""
        result = _ensure_imp_prob(odds_eu=2.0, imp_prob=0.48)
        assert result == 0.48
    
    def test_calculates_from_odds_if_none(self):
        """Calculates 1/odds when imp_prob is None."""
        result = _ensure_imp_prob(odds_eu=2.0, imp_prob=None)
        assert result == 0.5
    
    def test_calculation_accuracy(self):
        """Calculation is accurate."""
        result = _ensure_imp_prob(odds_eu=1.5, imp_prob=None)
        assert abs(result - 0.6667) < 0.001


class TestMapSubmitOddsToMinerSubmissionRows:
    """Tests for map_submit_odds_to_miner_submission_rows function."""
    
    def test_maps_single_submission(self):
        """Maps single market submission to rows."""
        request = SubmitOddsRequest(
            miner_id=42,
            miner_hotkey="hotkey123",
            submissions=[
                MarketSubmission(
                    market_id=100,
                    kind="moneyline",
                    prices=[
                        OutcomePrice(side="home", odds_eu=1.91, imp_prob=0.52),
                        OutcomePrice(side="away", odds_eu=2.05, imp_prob=0.48),
                    ],
                )
            ],
        )
        
        received_at = datetime(2025, 12, 8, 14, 33, 45, tzinfo=timezone.utc)
        
        rows = map_submit_odds_to_miner_submission_rows(request, received_at, bucket_seconds=60)
        
        assert len(rows) == 2
        home_row = next(r for r in rows if r["side"] == "home")
        assert home_row["miner_id"] == 42
        assert home_row["miner_hotkey"] == "hotkey123"
        assert home_row["market_id"] == 100
        assert home_row["odds_eu"] == 1.91
        assert home_row["imp_prob"] == 0.52
        assert home_row["payload"] == {"kind": "moneyline"}
    
    def test_maps_multiple_prices(self):
        """Maps multiple prices per market."""
        request = SubmitOddsRequest(
            miner_id=1,
            miner_hotkey="hk",
            submissions=[
                MarketSubmission(
                    market_id=100,
                    kind="moneyline",
                    prices=[
                        OutcomePrice(side="home", odds_eu=1.91, imp_prob=0.52),
                        OutcomePrice(side="away", odds_eu=2.05, imp_prob=0.48),
                    ],
                )
            ],
        )
        
        received_at = datetime(2025, 12, 8, 14, 33, 0, tzinfo=timezone.utc)
        rows = map_submit_odds_to_miner_submission_rows(request, received_at, 60)
        
        assert len(rows) == 2
        sides = {r["side"] for r in rows}
        assert sides == {"home", "away"}
    
    def test_maps_multiple_markets(self):
        """Maps multiple market submissions."""
        request = SubmitOddsRequest(
            miner_id=1,
            miner_hotkey="hk",
            submissions=[
                MarketSubmission(
                    market_id=100,
                    kind="moneyline",
                    prices=[
                        OutcomePrice(side="home", odds_eu=1.91, imp_prob=0.52),
                        OutcomePrice(side="away", odds_eu=2.05, imp_prob=0.48),
                    ],
                ),
                MarketSubmission(
                    market_id=101,
                    kind="spread",
                    prices=[
                        OutcomePrice(side="home", odds_eu=1.95, imp_prob=0.51),
                        OutcomePrice(side="away", odds_eu=1.95, imp_prob=0.49),
                    ],
                ),
            ],
        )
        
        received_at = datetime(2025, 12, 8, 14, 33, 0, tzinfo=timezone.utc)
        rows = map_submit_odds_to_miner_submission_rows(request, received_at, 60)
        
        assert len(rows) == 4
        market_ids = {r["market_id"] for r in rows}
        assert market_ids == {100, 101}
    
    def test_buckets_submitted_at(self):
        """submitted_at is floored to bucket boundary."""
        request = SubmitOddsRequest(
            miner_id=1,
            miner_hotkey="hk",
            submissions=[
                MarketSubmission(
                    market_id=100,
                    kind="moneyline",
                    prices=[
                        OutcomePrice(side="home", odds_eu=1.91, imp_prob=0.52),
                        OutcomePrice(side="away", odds_eu=2.05, imp_prob=0.48),
                    ],
                )
            ],
        )
        
        received_at = datetime(2025, 12, 8, 14, 33, 45, tzinfo=timezone.utc)
        rows = map_submit_odds_to_miner_submission_rows(request, received_at, 60)
        
        # Should be floored to 14:33:00
        assert rows[0]["submitted_at"].second == 0
        assert rows[0]["submitted_at"].minute == 33
    
    def test_preserves_priced_at(self):
        """priced_at from market submission is preserved."""
        priced_time = datetime(2025, 12, 8, 14, 30, 0, tzinfo=timezone.utc)
        
        request = SubmitOddsRequest(
            miner_id=1,
            miner_hotkey="hk",
            submissions=[
                MarketSubmission(
                    market_id=100,
                    kind="moneyline",
                    priced_at=priced_time,
                    prices=[
                        OutcomePrice(side="home", odds_eu=1.91, imp_prob=0.52),
                        OutcomePrice(side="away", odds_eu=2.05, imp_prob=0.48),
                    ],
                )
            ],
        )
        
        received_at = datetime(2025, 12, 8, 14, 33, 0, tzinfo=timezone.utc)
        rows = map_submit_odds_to_miner_submission_rows(request, received_at, 60)
        
        assert rows[0]["priced_at"] == priced_time
    
    def test_uses_imp_prob_from_model(self):
        """Uses imp_prob provided in the model."""
        request = SubmitOddsRequest(
            miner_id=1,
            miner_hotkey="hk",
            submissions=[
                MarketSubmission(
                    market_id=100,
                    kind="moneyline",
                    prices=[
                        OutcomePrice(side="home", odds_eu=2.0, imp_prob=0.48),  # Specific imp_prob
                        OutcomePrice(side="away", odds_eu=2.0, imp_prob=0.52),
                    ],
                )
            ],
        )
        
        received_at = datetime(2025, 12, 8, 14, 33, 0, tzinfo=timezone.utc)
        rows = map_submit_odds_to_miner_submission_rows(request, received_at, 60)
        
        home_row = next(r for r in rows if r["side"] == "home")
        assert home_row["imp_prob"] == 0.48  # Uses provided value, not calculated


class TestMapSubmitOutcomeToInboxRow:
    """Tests for map_submit_outcome_to_inbox_row function."""
    
    def _make_outcome_request(self, event_id: str = "123", miner_hotkey: str = "hotkey_abc"):
        """Helper to create valid SubmitOutcomeRequest."""
        return SubmitOutcomeRequest(
            event_id=event_id,
            miner_hotkey=miner_hotkey,
            winner_label="home",
            final_score="102-98",
            ts_submit=datetime(2025, 12, 8, 20, 0, 0, tzinfo=timezone.utc),
            sources=[],
        )
    
    def test_maps_outcome_to_inbox(self):
        """Maps outcome request to inbox row."""
        request = self._make_outcome_request("event_123", "hotkey_abc")
        
        received_at = datetime(2025, 12, 8, 20, 30, 0, tzinfo=timezone.utc)
        
        row = map_submit_outcome_to_inbox_row(request, received_at, bucket_seconds=300)
        
        assert row["topic"] == "outcome.submit"
        assert "dedupe_key" in row
        assert "outcome:event_123:hotkey_abc:" in row["dedupe_key"]
        assert "payload" in row
    
    def test_payload_contains_request_data(self):
        """Payload contains serialized request."""
        request = self._make_outcome_request("event_456", "hk")
        
        received_at = datetime(2025, 12, 8, 20, 30, 0, tzinfo=timezone.utc)
        row = map_submit_outcome_to_inbox_row(request, received_at, 300)
        
        payload = row["payload"]
        assert payload["event_id"] == "event_456"
        assert payload["miner_hotkey"] == "hk"
        assert payload["winner_label"] == "home"
    
    def test_custom_topic(self):
        """Accepts custom topic."""
        request = self._make_outcome_request()
        
        received_at = datetime(2025, 12, 8, 20, 30, 0, tzinfo=timezone.utc)
        row = map_submit_outcome_to_inbox_row(
            request, received_at, 300, topic="outcome.custom"
        )
        
        assert row["topic"] == "outcome.custom"
    
    def test_dedupe_key_bucketed(self):
        """Dedupe key uses bucketed timestamp."""
        request = self._make_outcome_request()
        
        # Two times in same 5-minute bucket
        time1 = datetime(2025, 12, 8, 20, 31, 0, tzinfo=timezone.utc)
        time2 = datetime(2025, 12, 8, 20, 33, 0, tzinfo=timezone.utc)
        
        row1 = map_submit_outcome_to_inbox_row(request, time1, 300)
        row2 = map_submit_outcome_to_inbox_row(request, time2, 300)
        
        assert row1["dedupe_key"] == row2["dedupe_key"]
    
    def test_different_events_different_keys(self):
        """Different events produce different dedupe keys."""
        request1 = self._make_outcome_request("event_123")
        request2 = self._make_outcome_request("event_456")
        
        received_at = datetime(2025, 12, 8, 20, 30, 0, tzinfo=timezone.utc)
        
        row1 = map_submit_outcome_to_inbox_row(request1, received_at, 300)
        row2 = map_submit_outcome_to_inbox_row(request2, received_at, 300)
        
        assert row1["dedupe_key"] != row2["dedupe_key"]
