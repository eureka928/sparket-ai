"""SportsDataIO provider models (NFL focus).

Pydantic v2 models and enums that map SportsDataIO JSON payloads to
strongly-typed structures aligned with our validator schema needs.
"""

from .enums import League, SeasonType, GameStatus, MarketWindow
from .config import LeagueCode, LeagueConfig, SportsDataIOConfig, build_default_config
from .client import SportsDataIOClient
from .types import (
    Team,
    Location,
    Game,
    MoneylinePrice,
    SpreadPrice,
    TotalPrice,
    GameOdds,
    GameOddsSet,
    Outcome,
)
from .mapping import (
    map_game_to_event,
    map_moneyline_quotes,
    map_spread_quotes,
    map_total_quotes,
    map_outcome_to_row,
    ensure_markets_for_event,
    normalize_quotes_by_market_ts,
    resolve_moneyline_result,
    resolve_total_result,
    resolve_spread_result,
)
from .leagues import (
    LeagueEntry,
    SoccerCompetition,
    league_rows_from_catalog,
    league_rows_from_soccer_competitions,
    build_league_index_by_sdio,
)
from .closing import (
    select_closing_quotes,
    closing_rows_from_odds,
)
from .provider_constants import PROVIDER_CODE, PROVIDER_ID
from .catalog import sport_rows

__all__ = [
    "League",
    "LeagueCode",
    "LeagueConfig",
    "SportsDataIOConfig",
    "build_default_config",
    "SportsDataIOClient",
    "SeasonType",
    "GameStatus",
    "MarketWindow",
    "Team",
    "Location",
    "Game",
    "MoneylinePrice",
    "SpreadPrice",
    "TotalPrice",
    "GameOdds",
    "GameOddsSet",
    "Outcome",
    "map_game_to_event",
    "map_moneyline_quotes",
    "map_spread_quotes",
    "map_total_quotes",
    "map_outcome_to_row",
    "ensure_markets_for_event",
    "normalize_quotes_by_market_ts",
    "resolve_moneyline_result",
    "resolve_total_result",
    "resolve_spread_result",
    "LeagueEntry",
    "SoccerCompetition",
    "league_rows_from_catalog",
    "league_rows_from_soccer_competitions",
    "build_league_index_by_sdio",
    "select_closing_quotes",
    "closing_rows_from_odds",
    "PROVIDER_CODE",
    "PROVIDER_ID",
    "sport_rows",
]


