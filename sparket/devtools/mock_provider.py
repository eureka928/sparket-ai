"""Mock provider for test mode.

Replaces SportsDataIO in test mode. Test scripts can populate this
with mock data, and the validator will "pull" from it as if it were
a real provider.

Enhanced with:
- Multiple sportsbook support (sharp/soft books)
- Time-series odds movement
- Vig/juice modeling
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List
from uuid import uuid4


@dataclass
class MockEvent:
    """Mock event data."""
    event_id: str
    home_team: str
    away_team: str
    start_time: datetime
    status: str = "scheduled"
    home_score: int | None = None
    away_score: int | None = None
    league_code: str = "TEST"
    sport_code: str = "soccer"
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "start_time": self.start_time.isoformat(),
            "status": self.status,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "league_code": self.league_code,
            "sport_code": self.sport_code,
        }


@dataclass
class MockMarket:
    """Mock market data."""
    market_id: str
    event_id: str
    kind: str  # moneyline, spread, total
    line: float | None = None
    true_prob_home: float | None = None  # Ground truth for testing
    
    def to_dict(self) -> dict:
        d = {
            "market_id": self.market_id,
            "event_id": self.event_id,
            "kind": self.kind,
            "line": self.line,
        }
        if self.true_prob_home is not None:
            d["true_prob_home"] = self.true_prob_home
        return d


@dataclass
class MockSportsbook:
    """Mock sportsbook configuration."""
    code: str           # e.g., "PINN", "DKNG"
    name: str           # e.g., "Pinnacle", "DraftKings"
    is_sharp: bool      # Sharp books have lower vig, more accurate
    vig: float = 0.04   # ~4% overround for soft books
    noise: float = 0.02 # Price noise standard deviation
    
    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "is_sharp": self.is_sharp,
            "vig": self.vig,
            "noise": self.noise,
        }


# Default sportsbook configurations
DEFAULT_SPORTSBOOKS = {
    "PINN": MockSportsbook(
        code="PINN",
        name="Pinnacle",
        is_sharp=True,
        vig=0.02,
        noise=0.01,
    ),
    "DKNG": MockSportsbook(
        code="DKNG",
        name="DraftKings",
        is_sharp=False,
        vig=0.045,
        noise=0.025,
    ),
    "FDUEL": MockSportsbook(
        code="FDUEL",
        name="FanDuel",
        is_sharp=False,
        vig=0.045,
        noise=0.025,
    ),
    "MGMBET": MockSportsbook(
        code="MGMBET",
        name="BetMGM",
        is_sharp=False,
        vig=0.05,
        noise=0.03,
    ),
    "TEST": MockSportsbook(
        code="TEST",
        name="TestBook",
        is_sharp=False,
        vig=0.0,
        noise=0.0,
    ),
}


@dataclass
class MockOdds:
    """Mock odds snapshot from provider."""
    market_id: str
    side: str  # home, away, over, under
    odds_eu: float
    imp_prob: float | None = None
    sportsbook_code: str = "TEST"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "side": self.side,
            "odds_eu": self.odds_eu,
            "imp_prob": self.imp_prob or (1.0 / self.odds_eu if self.odds_eu > 0 else None),
            "sportsbook_code": self.sportsbook_code,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OddsSnapshot:
    """A complete odds snapshot for one market at one time."""
    market_id: str
    timestamp: datetime
    sportsbook_code: str
    home_odds: float
    away_odds: float
    home_prob: float
    away_prob: float
    
    def to_odds_list(self) -> List[MockOdds]:
        """Convert to list of MockOdds for storage."""
        return [
            MockOdds(
                market_id=self.market_id,
                side="HOME",
                odds_eu=self.home_odds,
                imp_prob=self.home_prob,
                sportsbook_code=self.sportsbook_code,
                timestamp=self.timestamp,
            ),
            MockOdds(
                market_id=self.market_id,
                side="AWAY",
                odds_eu=self.away_odds,
                imp_prob=self.away_prob,
                sportsbook_code=self.sportsbook_code,
                timestamp=self.timestamp,
            ),
        ]


class OddsGenerator:
    """Generates realistic time-series odds movement."""
    
    def __init__(
        self,
        true_prob_home: float,
        opening_deviation: float = 0.05,
        volatility: float = 0.02,
        seed: int | None = None,
    ):
        """
        Args:
            true_prob_home: Ground truth home win probability (0.1 to 0.9)
            opening_deviation: How far opening odds deviate from true prob
            volatility: Random walk volatility per snapshot
            seed: Random seed for reproducibility
        """
        self.true_prob_home = max(0.1, min(0.9, true_prob_home))
        self.opening_deviation = opening_deviation
        self.volatility = volatility
        self.rng = random.Random(seed)
    
    def generate_series(
        self,
        market_id: str,
        sportsbook: MockSportsbook,
        open_time: datetime,
        close_time: datetime,
        interval_hours: int = 6,
    ) -> List[OddsSnapshot]:
        """Generate time-series odds that drift toward true probability.
        
        Args:
            market_id: Market identifier
            sportsbook: Sportsbook configuration
            open_time: When market opens
            close_time: When market closes (event start)
            interval_hours: Hours between snapshots
        
        Returns:
            List of OddsSnapshot from open to close
        """
        snapshots = []
        
        # Calculate number of intervals
        total_hours = (close_time - open_time).total_seconds() / 3600
        num_intervals = max(1, int(total_hours / interval_hours))
        
        # Opening odds with deviation
        deviation = self.rng.uniform(-self.opening_deviation, self.opening_deviation)
        current_prob = self.true_prob_home + deviation
        current_prob = max(0.1, min(0.9, current_prob))
        
        for i in range(num_intervals + 1):
            # Time for this snapshot
            progress = i / num_intervals if num_intervals > 0 else 1.0
            ts = open_time + timedelta(hours=i * interval_hours)
            if ts > close_time:
                ts = close_time
            
            # Drift toward true probability as time approaches close
            # Sharp books converge faster
            drift_rate = 0.3 if sportsbook.is_sharp else 0.2
            target_diff = self.true_prob_home - current_prob
            drift = target_diff * drift_rate * (1 + progress)
            
            # Add noise (less for sharp books)
            noise = self.rng.gauss(0, sportsbook.noise * (1 - progress * 0.5))
            
            current_prob += drift + noise
            current_prob = max(0.1, min(0.9, current_prob))
            
            # Apply vig to create odds
            home_prob, away_prob = self._apply_vig(current_prob, sportsbook.vig)
            home_odds = round(1.0 / home_prob, 3)
            away_odds = round(1.0 / away_prob, 3)
            
            snapshots.append(OddsSnapshot(
                market_id=market_id,
                timestamp=ts,
                sportsbook_code=sportsbook.code,
                home_odds=home_odds,
                away_odds=away_odds,
                home_prob=round(home_prob, 6),
                away_prob=round(away_prob, 6),
            ))
        
        return snapshots
    
    def _apply_vig(self, true_prob: float, vig: float) -> tuple[float, float]:
        """Apply vig/juice to true probabilities.
        
        Args:
            true_prob: True home win probability
            vig: Total overround (e.g., 0.04 = 4%)
        
        Returns:
            (home_implied_prob, away_implied_prob) that sum to 1+vig
        """
        away_prob = 1.0 - true_prob
        
        # Distribute vig proportionally
        total = 1.0 + vig
        home_implied = true_prob * total / (true_prob + away_prob)
        away_implied = away_prob * total / (true_prob + away_prob)
        
        return home_implied, away_implied


class MockProvider:
    """Mock provider that replaces SportsDataIO in test mode.
    
    Supports:
    - Creating mock events
    - Creating mock markets
    - Adding mock odds and outcomes
    - Multiple sportsbooks with different characteristics
    - Time-series odds generation
    """
    
    def __init__(self):
        self.events: dict[str, MockEvent] = {}
        self.markets: dict[str, MockMarket] = {}
        self.odds: dict[str, list[MockOdds]] = {}
        self.outcomes: dict[str, dict] = {}
        self.sportsbooks: dict[str, MockSportsbook] = dict(DEFAULT_SPORTSBOOKS)
        self.time_series: dict[str, List[OddsSnapshot]] = {}
    
    def reset(self):
        """Clear all mock data."""
        self.events.clear()
        self.markets.clear()
        self.odds.clear()
        self.outcomes.clear()
        self.time_series.clear()
        # Reset sportsbooks to defaults
        self.sportsbooks = dict(DEFAULT_SPORTSBOOKS)
    
    def add_sportsbook(
        self,
        code: str,
        name: str,
        is_sharp: bool = False,
        vig: float = 0.04,
        noise: float = 0.02,
    ) -> MockSportsbook:
        """Add or update a sportsbook configuration."""
        book = MockSportsbook(
            code=code,
            name=name,
            is_sharp=is_sharp,
            vig=vig,
            noise=noise,
        )
        self.sportsbooks[code] = book
        return book
    
    def get_sportsbook(self, code: str) -> MockSportsbook | None:
        """Get sportsbook by code."""
        return self.sportsbooks.get(code)
    
    def add_event(
        self,
        home_team: str,
        away_team: str,
        start_time: datetime,
        status: str = "scheduled",
        league_code: str = "TEST",
        sport_code: str = "soccer",
        event_id: str | None = None,
    ) -> MockEvent:
        """Create a mock event."""
        event_id = event_id or str(uuid4())
        event = MockEvent(
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            start_time=start_time,
            status=status,
            league_code=league_code,
            sport_code=sport_code,
        )
        self.events[event_id] = event
        return event
    
    def create_event(
        self,
        home_team: str,
        away_team: str,
        start_time: datetime,
        status: str = "scheduled",
        league_code: str = "TEST",
        sport_code: str = "soccer",
    ) -> MockEvent:
        return self.add_event(
            home_team=home_team,
            away_team=away_team,
            start_time=start_time,
            status=status,
            league_code=league_code,
            sport_code=sport_code,
        )
    
    def add_market(
        self,
        event_id: str,
        kind: str = "moneyline",
        line: float | None = None,
        market_id: str | None = None,
        true_prob_home: float | None = None,
    ) -> MockMarket:
        """Create a mock market."""
        market_id = market_id or str(uuid4())
        market = MockMarket(
            market_id=market_id,
            event_id=event_id,
            kind=kind,
            line=line,
            true_prob_home=true_prob_home,
        )
        self.markets[market_id] = market
        return market
    
    def create_market(
        self,
        event_id: str,
        kind: str = "moneyline",
        line: float | None = None,
        true_prob_home: float | None = None,
    ) -> MockMarket:
        return self.add_market(
            event_id=event_id,
            kind=kind,
            line=line,
            true_prob_home=true_prob_home,
        )
    
    def add_odds(
        self,
        market_id: str,
        side: str,
        odds_eu: float,
        timestamp: datetime | None = None,
        sportsbook_code: str = "TEST",
    ) -> MockOdds:
        """Add odds for a market."""
        odds = MockOdds(
            market_id=market_id,
            side=side,
            odds_eu=odds_eu,
            imp_prob=1.0 / odds_eu if odds_eu > 0 else None,
            sportsbook_code=sportsbook_code,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        self.odds.setdefault(market_id, []).append(odds)
        return odds
    
    def add_sportsbook_odds(
        self,
        market_id: str,
        sportsbook_code: str,
        home_odds: float,
        away_odds: float,
        timestamp: datetime | None = None,
    ) -> List[MockOdds]:
        """Add odds from a specific sportsbook (both home and away)."""
        ts = timestamp or datetime.now(timezone.utc)
        home = self.add_odds(market_id, "HOME", home_odds, ts, sportsbook_code)
        away = self.add_odds(market_id, "AWAY", away_odds, ts, sportsbook_code)
        return [home, away]
    
    def generate_odds_series(
        self,
        market_id: str,
        true_prob_home: float,
        open_time: datetime,
        close_time: datetime,
        interval_hours: int = 6,
        sportsbook_codes: List[str] | None = None,
        seed: int | None = None,
    ) -> List[OddsSnapshot]:
        """Generate time-series odds for a market across multiple sportsbooks.
        
        Args:
            market_id: Market identifier
            true_prob_home: True home win probability
            open_time: Market open time
            close_time: Market close time (event start)
            interval_hours: Hours between snapshots
            sportsbook_codes: Which books to include (default: all)
            seed: Random seed for reproducibility
        
        Returns:
            List of all generated snapshots
        """
        if sportsbook_codes is None:
            sportsbook_codes = list(self.sportsbooks.keys())
        
        all_snapshots = []
        
        for i, code in enumerate(sportsbook_codes):
            book = self.sportsbooks.get(code)
            if not book:
                continue
            
            # Use different seed per book for variety
            book_seed = (seed + i) if seed is not None else None
            
            generator = OddsGenerator(
                true_prob_home=true_prob_home,
                opening_deviation=0.05 if book.is_sharp else 0.08,
                volatility=book.noise,
                seed=book_seed,
            )
            
            snapshots = generator.generate_series(
                market_id=market_id,
                sportsbook=book,
                open_time=open_time,
                close_time=close_time,
                interval_hours=interval_hours,
            )
            
            # Store in time_series for later retrieval
            key = f"{market_id}:{code}"
            self.time_series[key] = snapshots
            
            # Also add to odds dict
            for snap in snapshots:
                for odds in snap.to_odds_list():
                    self.odds.setdefault(market_id, []).append(odds)
            
            all_snapshots.extend(snapshots)
        
        return all_snapshots
    
    def get_closing_odds(
        self,
        market_id: str,
        sportsbook_code: str | None = None,
    ) -> List[MockOdds]:
        """Get the closing (latest) odds for a market.
        
        Args:
            market_id: Market identifier
            sportsbook_code: Filter by book (None = all books)
        
        Returns:
            Latest odds for each side
        """
        market_odds = self.odds.get(market_id, [])
        if not market_odds:
            return []
        
        if sportsbook_code:
            market_odds = [o for o in market_odds if o.sportsbook_code == sportsbook_code]
        
        # Group by side, get latest
        latest_by_side: dict[str, MockOdds] = {}
        for odds in sorted(market_odds, key=lambda o: o.timestamp):
            latest_by_side[odds.side] = odds
        
        return list(latest_by_side.values())
    
    def get_consensus_closing(
        self,
        market_id: str,
        side: str = "HOME",
    ) -> dict | None:
        """Compute consensus closing odds across all sportsbooks.
        
        Returns simple average (production uses bias-weighted average).
        """
        closing_odds = self.get_closing_odds(market_id)
        side_odds = [o for o in closing_odds if o.side == side]
        
        if not side_odds:
            return None
        
        probs = [o.imp_prob or (1.0 / o.odds_eu) for o in side_odds]
        avg_prob = sum(probs) / len(probs)
        
        return {
            "market_id": market_id,
            "side": side,
            "prob_consensus": round(avg_prob, 6),
            "odds_consensus": round(1.0 / avg_prob, 4) if avg_prob > 0 else None,
            "contributing_books": len(side_odds),
            "std_dev": round(
                math.sqrt(sum((p - avg_prob) ** 2 for p in probs) / len(probs)), 6
            ) if len(probs) > 1 else 0.0,
        }
    
    def set_outcome(
        self,
        event_id: str,
        result: str,
        score_home: int | None = None,
        score_away: int | None = None,
    ) -> dict:
        """Set outcome for an event."""
        outcome = {
            "event_id": event_id,
            "result": result,
            "score_home": score_home,
            "score_away": score_away,
            "settled_at": datetime.now(timezone.utc).isoformat(),
        }
        self.outcomes[event_id] = outcome
        return outcome
    
    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of mock data."""
        return {
            "events": [event.to_dict() for event in self.events.values()],
            "markets": [market.to_dict() for market in self.markets.values()],
            "odds": {
                market_id: [odds.to_dict() for odds in odds_list]
                for market_id, odds_list in self.odds.items()
            },
            "outcomes": self.outcomes,
            "sportsbooks": [book.to_dict() for book in self.sportsbooks.values()],
            "time_series_count": len(self.time_series),
        }


_provider: MockProvider | None = None


def get_mock_provider() -> MockProvider:
    """Get or create global mock provider instance."""
    global _provider
    if _provider is None:
        _provider = MockProvider()
    return _provider
