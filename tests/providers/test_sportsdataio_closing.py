"""Tests for providers/sportsdataio/closing.py - Closing line selection."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from sparket.providers.sportsdataio.closing import (
    select_closing_quotes,
    closing_rows_from_odds,
)


class TestSelectClosingQuotes:
    """Tests for select_closing_quotes function."""
    
    def test_selects_last_quote_before_start(self):
        """Selects the last quote strictly before event start."""
        start_time = datetime(2025, 12, 8, 19, 0, 0, tzinfo=timezone.utc)
        
        quotes = [
            {"side": "home", "ts": start_time - timedelta(hours=24), "odds_eu": 1.90},
            {"side": "home", "ts": start_time - timedelta(hours=12), "odds_eu": 1.85},
            {"side": "home", "ts": start_time - timedelta(hours=1), "odds_eu": 1.91},  # Last before start
            {"side": "home", "ts": start_time + timedelta(hours=1), "odds_eu": 1.95},  # After start
        ]
        
        result = select_closing_quotes(quotes, start_time)
        
        assert len(result) == 1
        assert result[0]["odds_eu"] == 1.91
    
    def test_excludes_quotes_at_exactly_start_time(self):
        """Quotes at exactly start time are excluded."""
        start_time = datetime(2025, 12, 8, 19, 0, 0, tzinfo=timezone.utc)
        
        quotes = [
            {"side": "home", "ts": start_time - timedelta(hours=1), "odds_eu": 1.85},
            {"side": "home", "ts": start_time, "odds_eu": 1.91},  # Exactly at start
        ]
        
        result = select_closing_quotes(quotes, start_time)
        
        assert len(result) == 1
        assert result[0]["odds_eu"] == 1.85
    
    def test_handles_multiple_sides(self):
        """Selects last quote for each side independently."""
        start_time = datetime(2025, 12, 8, 19, 0, 0, tzinfo=timezone.utc)
        
        quotes = [
            {"side": "home", "ts": start_time - timedelta(hours=2), "odds_eu": 1.90},
            {"side": "home", "ts": start_time - timedelta(hours=1), "odds_eu": 1.85},
            {"side": "away", "ts": start_time - timedelta(hours=3), "odds_eu": 2.10},
            {"side": "away", "ts": start_time - timedelta(hours=1), "odds_eu": 2.05},
        ]
        
        result = select_closing_quotes(quotes, start_time)
        
        assert len(result) == 2
        sides = {q["side"]: q["odds_eu"] for q in result}
        assert sides["home"] == 1.85
        assert sides["away"] == 2.05
    
    def test_returns_empty_for_no_quotes_before_start(self):
        """Returns empty list when no quotes before start."""
        start_time = datetime(2025, 12, 8, 19, 0, 0, tzinfo=timezone.utc)
        
        quotes = [
            {"side": "home", "ts": start_time + timedelta(hours=1), "odds_eu": 1.90},
            {"side": "away", "ts": start_time + timedelta(hours=2), "odds_eu": 2.10},
        ]
        
        result = select_closing_quotes(quotes, start_time)
        
        assert result == []
    
    def test_handles_empty_input(self):
        """Handles empty quote list."""
        start_time = datetime(2025, 12, 8, 19, 0, 0, tzinfo=timezone.utc)
        result = select_closing_quotes([], start_time)
        assert result == []
    
    def test_skips_quotes_with_none_timestamp(self):
        """Skips quotes with None timestamp."""
        start_time = datetime(2025, 12, 8, 19, 0, 0, tzinfo=timezone.utc)
        
        quotes = [
            {"side": "home", "ts": None, "odds_eu": 1.90},
            {"side": "home", "ts": start_time - timedelta(hours=1), "odds_eu": 1.85},
        ]
        
        result = select_closing_quotes(quotes, start_time)
        
        assert len(result) == 1
        assert result[0]["odds_eu"] == 1.85
    
    def test_handles_three_way_market(self):
        """Handles three-way market (home, away, draw)."""
        start_time = datetime(2025, 12, 8, 19, 0, 0, tzinfo=timezone.utc)
        
        quotes = [
            {"side": "home", "ts": start_time - timedelta(hours=1), "odds_eu": 2.10},
            {"side": "away", "ts": start_time - timedelta(hours=1), "odds_eu": 3.50},
            {"side": "draw", "ts": start_time - timedelta(hours=1), "odds_eu": 3.20},
        ]
        
        result = select_closing_quotes(quotes, start_time)
        
        assert len(result) == 3
        sides = {q["side"] for q in result}
        assert sides == {"home", "away", "draw"}


class TestClosingRowsFromOdds:
    """Tests for closing_rows_from_odds function."""
    
    def test_transforms_quotes_to_closing_rows(self):
        """Transforms quote rows into closing rows."""
        start_time = datetime(2025, 12, 8, 19, 0, 0, tzinfo=timezone.utc)
        
        quotes = [
            {
                "provider_id": 1,
                "market_id": 100,
                "side": "home",
                "ts": start_time - timedelta(hours=1),
                "odds_eu": 1.91,
                "imp_prob": 0.523,
                "imp_prob_norm": 0.51,
            },
            {
                "provider_id": 1,
                "market_id": 100,
                "side": "away",
                "ts": start_time - timedelta(hours=1),
                "odds_eu": 2.05,
                "imp_prob": 0.487,
                "imp_prob_norm": 0.49,
            },
        ]
        
        result = closing_rows_from_odds(quotes, start_time)
        
        assert len(result) == 2
        
        home_row = next(r for r in result if r["side"] == "home")
        assert home_row["provider_id"] == 1
        assert home_row["market_id"] == 100
        assert home_row["ts_close"] == start_time - timedelta(hours=1)
        assert home_row["odds_eu_close"] == 1.91
        assert home_row["imp_prob_close"] == 0.523
        assert home_row["imp_prob_norm_close"] == 0.51
    
    def test_handles_missing_imp_prob_norm(self):
        """Handles missing imp_prob_norm field."""
        start_time = datetime(2025, 12, 8, 19, 0, 0, tzinfo=timezone.utc)
        
        quotes = [
            {
                "provider_id": 1,
                "market_id": 100,
                "side": "home",
                "ts": start_time - timedelta(hours=1),
                "odds_eu": 1.91,
                "imp_prob": 0.523,
                # No imp_prob_norm
            },
        ]
        
        result = closing_rows_from_odds(quotes, start_time)
        
        assert len(result) == 1
        assert result[0]["imp_prob_norm_close"] is None
    
    def test_returns_empty_for_no_closing_quotes(self):
        """Returns empty list when no closing quotes found."""
        start_time = datetime(2025, 12, 8, 19, 0, 0, tzinfo=timezone.utc)
        
        quotes = [
            {
                "provider_id": 1,
                "market_id": 100,
                "side": "home",
                "ts": start_time + timedelta(hours=1),  # After start
                "odds_eu": 1.91,
                "imp_prob": 0.523,
            },
        ]
        
        result = closing_rows_from_odds(quotes, start_time)
        
        assert result == []

