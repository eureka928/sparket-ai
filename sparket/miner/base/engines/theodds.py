"""The-Odds-API engine for fetching real market odds.

Uses The-Odds-API (https://the-odds-api.com/) to fetch odds from multiple sportsbooks.
Free tier: 500 requests/month (~16/day).

Requires API key via:
- SPARKET_BASE_MINER__ODDS_API_KEY environment variable
- Or passed directly to constructor
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from sparket.miner.base.engines.interface import OddsEngine, OddsPrices
from sparket.miner.base.data.teams import get_team_info
from sparket.miner.base.utils.cache import TTLCache


# The-Odds-API base URL
THEODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Sport key mapping (our codes -> The-Odds-API sport keys)
SPORT_KEYS: Dict[str, str] = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "NHL": "icehockey_nhl",
    "NCAAF": "americanfootball_ncaaf",
    "NCAAB": "basketball_ncaab",
}

# Default regions and markets
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h,spreads,totals"


class TheOddsEngine(OddsEngine):
    """Fetches odds from The-Odds-API.
    
    Features:
    - Fetches real odds from multiple sportsbooks
    - Caches responses aggressively (1 hour default)
    - Falls back to None on errors (caller should fallback to naive)
    
    Example:
        engine = TheOddsEngine(api_key="your_key")
        odds = await engine.get_odds({
            "sport": "NFL",
            "home_team": "KC",
            "away_team": "DAL",
            "kind": "MONEYLINE",
        })
    
    Rate Limits:
    - Free tier: 500 requests/month
    - Each sport fetch counts as 1 request
    - Cache results to minimize API calls
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_seconds: float = 3600,  # 1 hour
        timeout: float = 10.0,
    ) -> None:
        """Initialize The-Odds-API engine.
        
        Args:
            api_key: API key (or uses SPARKET_BASE_MINER__ODDS_API_KEY env var)
            cache_ttl_seconds: How long to cache responses
            timeout: HTTP request timeout
        """
        self.api_key = api_key or os.getenv("SPARKET_BASE_MINER__ODDS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TheOddsEngine requires an API key. "
                "Set SPARKET_BASE_MINER__ODDS_API_KEY or pass api_key parameter."
            )
        
        self._cache = TTLCache[Dict[str, Any]](ttl_seconds=cache_ttl_seconds)
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        
        # Track API usage
        self._requests_remaining: Optional[int] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client
    
    async def get_odds(self, market: Dict[str, Any]) -> Optional[OddsPrices]:
        """Fetch odds for a market from The-Odds-API.
        
        Args:
            market: Market info dict with sport, home_team, away_team, kind
        
        Returns:
            OddsPrices if found, None otherwise
        """
        sport = market.get("sport", "NFL").upper()
        home_team = market.get("home_team", "")
        away_team = market.get("away_team", "")
        kind = market.get("kind", "MONEYLINE").upper()
        
        # Get sport key for API
        sport_key = SPORT_KEYS.get(sport)
        if not sport_key:
            return None
        
        # Fetch odds for the sport (cached)
        all_odds = await self._fetch_sport_odds(sport_key)
        if not all_odds:
            return None
        
        # Find the specific game
        game_odds = self._find_game(all_odds, home_team, away_team, sport)
        if not game_odds:
            return None
        
        # Extract the right market type
        return self._extract_odds(game_odds, kind, market.get("line"))
    
    async def _fetch_sport_odds(self, sport_key: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch all odds for a sport (cached)."""
        cache_key = f"theodds:{sport_key}"
        
        async def fetch():
            url = f"{THEODDS_API_BASE}/sports/{sport_key}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": DEFAULT_REGIONS,
                "markets": DEFAULT_MARKETS,
                "oddsFormat": "decimal",
            }
            
            try:
                client = await self._get_client()
                response = await client.get(url, params=params)
                
                # Track remaining requests
                remaining = response.headers.get("x-requests-remaining")
                if remaining:
                    self._requests_remaining = int(remaining)
                
                response.raise_for_status()
                return response.json()
            except Exception:
                return None
        
        return await self._cache.get_or_set(cache_key, fetch)
    
    def _find_game(
        self,
        odds_list: List[Dict[str, Any]],
        home_team: str,
        away_team: str,
        sport: str,
    ) -> Optional[Dict[str, Any]]:
        """Find a specific game in the odds list."""
        # Get full team names for matching
        home_info = get_team_info(sport, home_team)
        away_info = get_team_info(sport, away_team)
        
        home_name = home_info["name"] if home_info else home_team
        away_name = away_info["name"] if away_info else away_team
        
        for game in odds_list:
            api_home = game.get("home_team", "")
            api_away = game.get("away_team", "")
            
            # Match by name (case-insensitive, partial match)
            if self._teams_match(home_name, api_home) and self._teams_match(away_name, api_away):
                return game
        
        return None
    
    def _teams_match(self, our_name: str, api_name: str) -> bool:
        """Check if team names match (fuzzy)."""
        our_lower = our_name.lower()
        api_lower = api_name.lower()
        
        # Exact match
        if our_lower == api_lower:
            return True
        
        # One contains the other
        if our_lower in api_lower or api_lower in our_lower:
            return True
        
        # Last word match (e.g., "Chiefs" matches "Kansas City Chiefs")
        our_last = our_lower.split()[-1] if our_lower else ""
        api_last = api_lower.split()[-1] if api_lower else ""
        if our_last and api_last and our_last == api_last:
            return True
        
        return False
    
    def _extract_odds(
        self,
        game: Dict[str, Any],
        kind: str,
        line: Optional[float] = None,
    ) -> Optional[OddsPrices]:
        """Extract odds for a specific market type."""
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            return None
        
        # Use first bookmaker (usually consensus-ish)
        book = bookmakers[0]
        markets = book.get("markets", [])
        
        # Map our market kind to API market key
        market_key = {
            "MONEYLINE": "h2h",
            "SPREAD": "spreads",
            "TOTAL": "totals",
        }.get(kind, "h2h")
        
        # Find the market
        market_data = None
        for m in markets:
            if m.get("key") == market_key:
                market_data = m
                break
        
        if not market_data:
            return None
        
        outcomes = market_data.get("outcomes", [])
        if len(outcomes) < 2:
            return None
        
        home_team = game.get("home_team", "")
        
        if kind == "TOTAL":
            return self._parse_total_odds(outcomes)
        else:
            return self._parse_side_odds(outcomes, home_team)
    
    def _parse_side_odds(
        self,
        outcomes: List[Dict[str, Any]],
        home_team: str,
    ) -> Optional[OddsPrices]:
        """Parse moneyline or spread odds."""
        home_odds = None
        away_odds = None
        
        for outcome in outcomes:
            name = outcome.get("name", "")
            price = outcome.get("price", 0)
            
            if self._teams_match(name, home_team):
                home_odds = price
            else:
                away_odds = price
        
        if not home_odds or not away_odds:
            return None
        
        # Convert EU odds to implied probabilities
        home_prob = 1.0 / home_odds
        away_prob = 1.0 / away_odds
        
        # Normalize to remove vig for "true" probabilities
        total = home_prob + away_prob
        home_prob_true = home_prob / total
        away_prob_true = away_prob / total
        
        return OddsPrices(
            home_prob=home_prob_true,
            away_prob=away_prob_true,
            home_odds_eu=round(home_odds, 2),
            away_odds_eu=round(away_odds, 2),
        )
    
    def _parse_total_odds(
        self,
        outcomes: List[Dict[str, Any]],
    ) -> Optional[OddsPrices]:
        """Parse totals (over/under) odds."""
        over_odds = None
        under_odds = None
        
        for outcome in outcomes:
            name = outcome.get("name", "").lower()
            price = outcome.get("price", 0)
            
            if "over" in name:
                over_odds = price
            elif "under" in name:
                under_odds = price
        
        if not over_odds or not under_odds:
            return None
        
        # Convert to probabilities
        over_prob = 1.0 / over_odds
        under_prob = 1.0 / under_odds
        total = over_prob + under_prob
        over_prob_true = over_prob / total
        under_prob_true = under_prob / total
        
        return OddsPrices(
            home_prob=over_prob_true,  # Map over to home
            away_prob=under_prob_true,  # Map under to away
            home_odds_eu=round(over_odds, 2),
            away_odds_eu=round(under_odds, 2),
            over_prob=over_prob_true,
            under_prob=under_prob_true,
            over_odds_eu=round(over_odds, 2),
            under_odds_eu=round(under_odds, 2),
        )
    
    @property
    def requests_remaining(self) -> Optional[int]:
        """Number of API requests remaining this month."""
        return self._requests_remaining
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()








