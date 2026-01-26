"""Naive odds engine using statistical baselines.

Generates odds based on historical league averages without external data.
Used as a fallback when no market data or team stats are available.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sparket.miner.base.engines.interface import OddsEngine, OddsPrices


# Historical home-field advantage by league (approximate)
HOME_ADVANTAGE: Dict[str, float] = {
    "NFL": 0.03,   # ~53% home win rate
    "NBA": 0.06,   # ~56% home win rate
    "MLB": 0.04,   # ~54% home win rate
    "NHL": 0.05,   # ~55% home win rate
    "NCAAF": 0.04,
    "NCAAB": 0.06,
}

# Standard vig (juice) to apply
STANDARD_VIG = 0.045  # ~4.5% total vig (-110 on both sides)


class NaiveEngine(OddsEngine):
    """Generates odds using statistical baselines.
    
    This is the simplest possible odds engine. It assumes:
    - Home team has a slight advantage (varies by sport)
    - All games are ~50/50 with home adjustment
    - Standard vig applies
    
    This will score poorly but provides a working baseline.
    
    Example:
        engine = NaiveEngine()
        odds = engine.get_odds_sync({"kind": "MONEYLINE", "sport": "NFL"})
        print(f"Home: {odds.home_prob:.1%}, Away: {odds.away_prob:.1%}")
    """
    
    def __init__(self, vig: float = STANDARD_VIG) -> None:
        """Initialize naive engine.
        
        Args:
            vig: Total vigorish to apply (default: 4.5%)
        """
        self.vig = vig
    
    async def get_odds(self, market: Dict[str, Any]) -> Optional[OddsPrices]:
        """Generate naive odds for a market."""
        return self.get_odds_sync(market)
    
    def get_odds_sync(self, market: Dict[str, Any]) -> Optional[OddsPrices]:
        """Synchronous odds generation.
        
        Args:
            market: Market info with 'kind' and 'sport' keys
        
        Returns:
            OddsPrices with naive probabilities
        """
        kind = market.get("kind", "MONEYLINE").upper()
        sport = market.get("sport", "NFL").upper()
        
        if kind == "MONEYLINE":
            return self._moneyline_odds(sport)
        elif kind == "SPREAD":
            return self._spread_odds(sport, market.get("line", 0))
        elif kind == "TOTAL":
            return self._total_odds(sport)
        else:
            # Default to moneyline logic
            return self._moneyline_odds(sport)
    
    def _moneyline_odds(self, sport: str) -> OddsPrices:
        """Generate moneyline odds with home advantage."""
        home_advantage = HOME_ADVANTAGE.get(sport, 0.03)
        
        # Base probability: 50/50 with home advantage
        true_home_prob = 0.5 + home_advantage
        true_away_prob = 1.0 - true_home_prob
        
        # Apply vig proportionally
        return self._apply_vig(true_home_prob, true_away_prob)
    
    def _spread_odds(self, sport: str, line: float) -> OddsPrices:
        """Generate spread odds.
        
        For spreads, the line theoretically equalizes the teams,
        so we expect ~50/50 on each side with vig.
        """
        # Spreads are designed to be 50/50
        true_home_prob = 0.5
        true_away_prob = 0.5
        
        # Small adjustment based on line direction
        # Larger negative spread for home = slightly less likely to cover
        if line != 0:
            adjustment = min(0.02, abs(line) * 0.001)
            if line < 0:  # Home is favored (giving points)
                true_home_prob -= adjustment
            else:  # Away is favored
                true_home_prob += adjustment
            true_away_prob = 1.0 - true_home_prob
        
        return self._apply_vig(true_home_prob, true_away_prob)
    
    def _total_odds(self, sport: str) -> OddsPrices:
        """Generate totals (over/under) odds.
        
        Totals are also designed to be ~50/50.
        """
        true_over_prob = 0.5
        true_under_prob = 0.5
        
        vigged = self._apply_vig(true_over_prob, true_under_prob)
        
        # Map to over/under fields
        return OddsPrices(
            home_prob=vigged.home_prob,  # Use home as over
            away_prob=vigged.away_prob,  # Use away as under
            home_odds_eu=vigged.home_odds_eu,
            away_odds_eu=vigged.away_odds_eu,
            over_prob=vigged.home_prob,
            under_prob=vigged.away_prob,
            over_odds_eu=vigged.home_odds_eu,
            under_odds_eu=vigged.away_odds_eu,
        )
    
    def _apply_vig(self, prob1: float, prob2: float) -> OddsPrices:
        """Apply vigorish to true probabilities.
        
        The vig is added proportionally so both sides are slightly
        worse than fair odds.
        
        Args:
            prob1: True probability for side 1 (home/over)
            prob2: True probability for side 2 (away/under)
        
        Returns:
            OddsPrices with vigged probabilities and EU odds
        """
        # Add vig proportionally
        total_vig = 1.0 + self.vig
        vigged_prob1 = prob1 * total_vig / (prob1 * total_vig + prob2 * total_vig)
        vigged_prob2 = 1.0 - vigged_prob1
        
        # But we need implied probabilities that sum to > 1
        # Standard approach: each side's implied prob = fair prob + (vig/2)
        implied_prob1 = prob1 + (self.vig / 2)
        implied_prob2 = prob2 + (self.vig / 2)
        
        # EU odds = 1 / implied_probability
        odds1 = 1.0 / implied_prob1
        odds2 = 1.0 / implied_prob2
        
        return OddsPrices(
            home_prob=prob1,  # Return true probabilities
            away_prob=prob2,
            home_odds_eu=round(odds1, 2),
            away_odds_eu=round(odds2, 2),
        )








