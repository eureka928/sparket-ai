"""Abstract base class for odds engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OddsPrices:
    """Odds output from an engine.
    
    Probabilities should sum to ~1.0 (allowing for vig).
    EU odds are decimal odds (1/probability before vig).
    """
    home_prob: float
    away_prob: float
    home_odds_eu: float
    away_odds_eu: float
    
    # For totals markets
    over_prob: Optional[float] = None
    under_prob: Optional[float] = None
    over_odds_eu: Optional[float] = None
    under_odds_eu: Optional[float] = None
    
    # For spreads - same as home/away but with line context
    # (home_prob represents probability home covers the spread)
    
    def __post_init__(self) -> None:
        """Validate odds are reasonable."""
        assert 0 < self.home_prob < 1, f"home_prob must be in (0,1), got {self.home_prob}"
        assert 0 < self.away_prob < 1, f"away_prob must be in (0,1), got {self.away_prob}"
        assert self.home_odds_eu > 1, f"home_odds_eu must be > 1, got {self.home_odds_eu}"
        assert self.away_odds_eu > 1, f"away_odds_eu must be > 1, got {self.away_odds_eu}"


class OddsEngine(ABC):
    """Abstract base class for odds generation/fetching.
    
    Implementations can:
    - Generate odds from statistical models (NaiveEngine)
    - Fetch from external APIs (TheOddsEngine)
    - Use custom data sources
    
    Example:
        class MyCustomEngine(OddsEngine):
            async def get_odds(self, market: dict) -> Optional[OddsPrices]:
                # Your logic here
                return OddsPrices(...)
    """
    
    @abstractmethod
    async def get_odds(self, market: Dict[str, Any]) -> Optional[OddsPrices]:
        """Generate or fetch odds for a market.
        
        Args:
            market: Market info dict with keys:
                - market_id: int
                - kind: str (MONEYLINE, SPREAD, TOTAL)
                - home_team: str (team code like "KC")
                - away_team: str (team code like "DAL")
                - sport: str (league code like "NFL")
                - line: Optional[float] (for spreads/totals)
        
        Returns:
            OddsPrices if odds available, None otherwise.
        """
        pass
    
    def get_odds_sync(self, market: Dict[str, Any]) -> Optional[OddsPrices]:
        """Synchronous version for simple engines that don't need async.
        
        Default implementation raises NotImplementedError.
        Subclasses can override for sync-only engines.
        """
        raise NotImplementedError("Use get_odds() for async engines")








