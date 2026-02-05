"""The-Odds-API integration for market consensus odds.

This fetches real-time odds from multiple sportsbooks and calculates
a sharp-weighted consensus to blend with our model predictions.

Key insight: Pinnacle is the sharpest book - weight it highest.

API Docs: https://the-odds-api.com/liveapi/guides/v4/
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from sparket.miner.base.engines.interface import OddsPrices


# Sport key mapping (Sparket -> The-Odds-API)
SPORT_MAPPING = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "NHL": "icehockey_nhl",
    "NCAAF": "americanfootball_ncaaf",
    "NCAAB": "basketball_ncaab",
}

# Sportsbook sharpness weights (higher = sharper/more accurate)
BOOK_WEIGHTS = {
    "pinnacle": 3.0,      # Sharpest book - highest weight
    "betfair_ex_eu": 2.5, # Exchange - sharp
    "betfair": 2.5,
    "matchbook": 2.0,
    "betcris": 2.0,
    "bookmaker": 1.5,
    "bovada": 1.2,
    "betonlineag": 1.2,
    "draftkings": 1.0,    # Retail books - lower weight
    "fanduel": 1.0,
    "betmgm": 1.0,
    "pointsbetus": 1.0,
    "williamhill_us": 1.0,
    "caesars": 1.0,
    "unibet_us": 0.8,
    "superbook": 0.8,
    "wynnbet": 0.8,
    "betrivers": 0.8,
    "twinspires": 0.8,
    "betus": 0.7,
}


@dataclass
class MarketOdds:
    """Odds for a single market from multiple books."""

    event_id: str
    home_team: str
    away_team: str
    commence_time: datetime

    # Consensus odds (sharp-weighted average)
    home_prob: float
    away_prob: float
    home_odds: float
    away_odds: float

    # Individual book odds
    book_odds: Dict[str, Dict[str, float]]  # {book: {home: odds, away: odds}}

    # Metadata
    num_books: int
    has_pinnacle: bool
    last_update: datetime


class OddsAPIFetcher:
    """Fetch and process odds from The-Odds-API.

    Usage:
        fetcher = OddsAPIFetcher(api_key="your_key")

        # Get consensus for a game
        odds = await fetcher.get_consensus_odds("NFL", "KC", "BUF")
        print(f"Consensus: KC {odds.home_prob:.1%}, BUF {odds.away_prob:.1%}")

        # Get as OddsPrices for blending
        prices = await fetcher.get_odds_prices("NFL", "KC", "BUF")
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(
        self,
        api_key: str,
        regions: str = "us,us2,eu",
        timeout: float = 10.0,
        cache_ttl_seconds: int = 300,  # 5 minute cache
    ) -> None:
        """Initialize the fetcher.

        Args:
            api_key: The-Odds-API key
            regions: Comma-separated regions (us, us2, uk, eu, au)
            timeout: Request timeout in seconds
            cache_ttl_seconds: How long to cache responses
        """
        self.api_key = api_key
        self.regions = regions
        self.timeout = timeout
        self.cache_ttl = cache_ttl_seconds

        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self._remaining_requests: Optional[int] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def remaining_requests(self) -> Optional[int]:
        """Remaining API requests this month."""
        return self._remaining_requests

    async def get_odds(
        self,
        sport: str,
        markets: str = "h2h",
    ) -> List[Dict[str, Any]]:
        """Fetch odds for all games in a sport.

        Args:
            sport: Sport code (NFL, NBA, etc.)
            markets: Market types (h2h, spreads, totals)

        Returns:
            List of event data with odds from multiple books
        """
        # Map sport code
        api_sport = SPORT_MAPPING.get(sport.upper(), sport.lower())

        # Check cache
        cache_key = f"{api_sport}:{markets}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.cache_ttl:
                return cached_data

        # Fetch from API
        client = await self._get_client()
        url = f"{self.BASE_URL}/sports/{api_sport}/odds"

        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": markets,
            "oddsFormat": "decimal",
        }

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()

            # Track remaining requests
            self._remaining_requests = int(
                response.headers.get("x-requests-remaining", 0)
            )

            data = response.json()

            # Cache response
            self._cache[cache_key] = (datetime.now(timezone.utc), data)

            return data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid API key") from e
            elif e.response.status_code == 429:
                raise ValueError("API rate limit exceeded") from e
            raise

    async def get_consensus_odds(
        self,
        sport: str,
        home_team: str,
        away_team: str,
    ) -> Optional[MarketOdds]:
        """Get sharp-weighted consensus odds for a specific game.

        Args:
            sport: Sport code
            home_team: Home team code (e.g., "KC")
            away_team: Away team code (e.g., "BUF")

        Returns:
            MarketOdds with consensus probabilities, or None if not found
        """
        events = await self.get_odds(sport, markets="h2h")

        # Find matching event
        for event in events:
            # Match by team names (fuzzy matching)
            event_home = event.get("home_team", "").upper()
            event_away = event.get("away_team", "").upper()

            # Check if teams match (could be full name or abbreviation)
            home_match = (
                home_team.upper() in event_home or
                event_home in home_team.upper() or
                self._team_matches(home_team, event_home)
            )
            away_match = (
                away_team.upper() in event_away or
                event_away in away_team.upper() or
                self._team_matches(away_team, event_away)
            )

            if home_match and away_match:
                return self._process_event(event)

        return None

    def _team_matches(self, code: str, full_name: str) -> bool:
        """Check if team code matches full name."""
        # Common mappings for NFL, NBA, MLB, NHL
        mappings = {
            # NFL Teams
            "KC": "KANSAS CITY",
            "BUF": "BUFFALO",
            "SF": "SAN FRANCISCO",
            "PHI": "PHILADELPHIA",
            "DAL": "DALLAS",
            "MIA": "MIAMI",
            "DET": "DETROIT",
            "BAL": "BALTIMORE",
            "CIN": "CINCINNATI",
            "LAR": "LOS ANGELES RAMS",
            "LAC": "LOS ANGELES CHARGERS",
            "TB": "TAMPA BAY",
            "GB": "GREEN BAY",
            "NO": "NEW ORLEANS",
            "NE": "NEW ENGLAND",
            "NYG": "NEW YORK GIANTS",
            "NYJ": "NEW YORK JETS",
            "LV": "LAS VEGAS",
            "MIN": "MINNESOTA",
            "SEA": "SEATTLE",
            "ATL": "ATLANTA",
            "CAR": "CAROLINA",
            "CHI": "CHICAGO",
            "ARI": "ARIZONA",
            "WAS": "WASHINGTON",
            "DEN": "DENVER",
            "HOU": "HOUSTON",
            "IND": "INDIANAPOLIS",
            "JAX": "JACKSONVILLE",
            "TEN": "TENNESSEE",
            "PIT": "PITTSBURGH",
            "CLE": "CLEVELAND",
            # NBA Teams
            "BOS": "BOSTON",
            "LAL": "LOS ANGELES LAKERS",
            "GSW": "GOLDEN STATE",
            "MIL": "MILWAUKEE",
            "PHX": "PHOENIX",
            "SAC": "SACRAMENTO",
            "NYK": "NEW YORK KNICKS",
            "BKN": "BROOKLYN",
            "TOR": "TORONTO",
            "OKC": "OKLAHOMA CITY",
            "ORL": "ORLANDO",
            "UTA": "UTAH",
            "POR": "PORTLAND",
            "SAS": "SAN ANTONIO",
            "CHA": "CHARLOTTE",
            "MEM": "MEMPHIS",
            "NOP": "NEW ORLEANS PELICANS",
            # MLB Teams
            "LAD": "LOS ANGELES DODGERS",
            "NYY": "NEW YORK YANKEES",
            "NYM": "NEW YORK METS",
            "SD": "SAN DIEGO",
            "STL": "ST. LOUIS",
            "TEX": "TEXAS",
            "CHW": "CHICAGO WHITE SOX",
            "CHC": "CHICAGO CUBS",
            "LAA": "LOS ANGELES ANGELS",
            "OAK": "OAKLAND",
            "COL": "COLORADO",
            # NHL Teams
            "VGK": "VEGAS",
            "NJ": "NEW JERSEY",
            "EDM": "EDMONTON",
            "CGY": "CALGARY",
            "NSH": "NASHVILLE",
            "WPG": "WINNIPEG",
            "OTT": "OTTAWA",
            "MTL": "MONTREAL",
            "SJ": "SAN JOSE",
            "CBJ": "COLUMBUS",
            "ANA": "ANAHEIM",
            "FLA": "FLORIDA",
            "NYR": "NEW YORK RANGERS",
            "NYI": "NEW YORK ISLANDERS",
            "VAN": "VANCOUVER",
            "LAK": "LOS ANGELES KINGS",
        }

        expected = mappings.get(code.upper(), code.upper())
        return expected in full_name.upper()

    def _process_event(self, event: Dict[str, Any]) -> MarketOdds:
        """Process event data into MarketOdds."""
        bookmakers = event.get("bookmakers", [])

        book_odds: Dict[str, Dict[str, float]] = {}
        weighted_home = 0.0
        weighted_away = 0.0
        total_weight = 0.0
        has_pinnacle = False

        for book in bookmakers:
            book_key = book.get("key", "").lower()
            weight = BOOK_WEIGHTS.get(book_key, 0.5)

            if book_key == "pinnacle":
                has_pinnacle = True

            # Get h2h market
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                outcomes = {
                    o.get("name"): o.get("price", 2.0)
                    for o in market.get("outcomes", [])
                }

                home_name = event.get("home_team", "")
                away_name = event.get("away_team", "")

                home_odds = outcomes.get(home_name, 2.0)
                away_odds = outcomes.get(away_name, 2.0)

                book_odds[book_key] = {
                    "home": home_odds,
                    "away": away_odds,
                }

                # Convert to implied probability (remove vig proportionally)
                # Guard against division by zero
                if home_odds <= 1.0 or away_odds <= 1.0:
                    continue
                implied_home = 1 / home_odds
                implied_away = 1 / away_odds
                total_implied = implied_home + implied_away

                if total_implied <= 0:
                    continue

                true_home = implied_home / total_implied
                true_away = implied_away / total_implied

                weighted_home += weight * true_home
                weighted_away += weight * true_away
                total_weight += weight

        # Calculate consensus
        if total_weight > 0:
            consensus_home = weighted_home / total_weight
            consensus_away = weighted_away / total_weight
        else:
            consensus_home = 0.5
            consensus_away = 0.5

        # Store raw probabilities - odds will be computed when needed with caller's vig
        # Using 0 vig here since these odds aren't used for submission
        home_odds_final = round(1 / max(0.001, consensus_home), 2)
        away_odds_final = round(1 / max(0.001, consensus_away), 2)

        return MarketOdds(
            event_id=event.get("id", ""),
            home_team=event.get("home_team", ""),
            away_team=event.get("away_team", ""),
            commence_time=datetime.fromisoformat(
                event.get("commence_time", "").replace("Z", "+00:00")
            ),
            home_prob=consensus_home,
            away_prob=consensus_away,
            home_odds=home_odds_final,
            away_odds=away_odds_final,
            book_odds=book_odds,
            num_books=len(book_odds),
            has_pinnacle=has_pinnacle,
            last_update=datetime.now(timezone.utc),
        )

    async def get_odds_prices(
        self,
        sport: str,
        home_team: str,
        away_team: str,
        vig: float = 0.045,
    ) -> Optional[OddsPrices]:
        """Get consensus odds as OddsPrices for blending.

        Args:
            sport: Sport code
            home_team: Home team code
            away_team: Away team code
            vig: Vigorish to apply to the odds

        Returns:
            OddsPrices or None if not found
        """
        market = await self.get_consensus_odds(sport, home_team, away_team)

        if market is None:
            return None

        # Apply vig to compute odds from probabilities
        implied_home = market.home_prob + vig / 2
        implied_away = market.away_prob + vig / 2
        implied_home = max(0.001, min(0.999, implied_home))
        implied_away = max(0.001, min(0.999, implied_away))

        home_odds = max(1.01, min(1000.0, 1 / implied_home))
        away_odds = max(1.01, min(1000.0, 1 / implied_away))

        return OddsPrices(
            home_prob=market.home_prob,
            away_prob=market.away_prob,
            home_odds_eu=round(home_odds, 2),
            away_odds_eu=round(away_odds, 2),
        )

    async def get_all_games(self, sport: str) -> List[MarketOdds]:
        """Get consensus odds for all games in a sport.

        Args:
            sport: Sport code

        Returns:
            List of MarketOdds for each game
        """
        events = await self.get_odds(sport, markets="h2h")
        return [self._process_event(e) for e in events]


def blend_with_market(
    model_prob: float,
    market_prob: float,
    model_weight: float = 0.4,
    vig: float = 0.045,
) -> OddsPrices:
    """Blend model prediction with market consensus.

    Args:
        model_prob: Your model's home win probability
        market_prob: Market consensus home win probability
        model_weight: Weight for model (0-1), rest goes to market
        vig: Vigorish to apply

    Returns:
        OddsPrices with blended probabilities

    Validator bounds: odds_eu in (1.01, 1000], imp_prob in (0.001, 0.999)
    """
    market_weight = 1 - model_weight

    blended_home = model_weight * model_prob + market_weight * market_prob
    blended_away = 1 - blended_home

    # Clamp to validator-accepted range (0.001, 0.999)
    blended_home = max(0.001, min(0.999, blended_home))
    blended_away = max(0.001, min(0.999, blended_away))

    # Convert to odds with vig
    implied_home = blended_home + vig / 2
    implied_away = blended_away + vig / 2
    implied_home = max(0.001, min(0.999, implied_home))
    implied_away = max(0.001, min(0.999, implied_away))

    home_odds = max(1.01, min(1000.0, 1 / implied_home))
    away_odds = max(1.01, min(1000.0, 1 / implied_away))

    return OddsPrices(
        home_prob=blended_home,
        away_prob=blended_away,
        home_odds_eu=round(home_odds, 2),
        away_odds_eu=round(away_odds, 2),
    )
