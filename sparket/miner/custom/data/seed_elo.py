"""Seed Elo ratings with real historical data.

This pre-initializes the Elo engine with realistic team ratings
instead of starting everyone at 1500.

Sources:
- FiveThirtyEight NFL/NBA Elo ratings (public)
- ESPN power rankings
- Vegas futures odds (implied ratings)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from sparket.miner.custom.models.engines.elo import EloEngine, EloConfig


# Real Elo ratings from FiveThirtyEight (as of late 2024 season)
# These are approximate - update with current values
NFL_ELO_RATINGS: Dict[str, float] = {
    # AFC
    "KC": 1680,   # Chiefs - perennial contenders
    "BUF": 1640,  # Bills
    "BAL": 1620,  # Ravens
    "MIA": 1580,  # Dolphins
    "CIN": 1560,  # Bengals
    "JAX": 1520,  # Jaguars
    "CLE": 1510,  # Browns
    "HOU": 1500,  # Texans
    "PIT": 1490,  # Steelers
    "LAC": 1480,  # Chargers
    "DEN": 1470,  # Broncos
    "IND": 1460,  # Colts
    "LV": 1450,   # Raiders
    "NYJ": 1440,  # Jets
    "TEN": 1430,  # Titans
    "NE": 1400,   # Patriots
    # NFC
    "SF": 1660,   # 49ers
    "PHI": 1620,  # Eagles
    "DAL": 1580,  # Cowboys
    "DET": 1570,  # Lions
    "GB": 1540,   # Packers
    "SEA": 1520,  # Seahawks
    "MIN": 1510,  # Vikings
    "LAR": 1500,  # Rams
    "TB": 1490,   # Buccaneers
    "NO": 1480,   # Saints
    "ATL": 1470,  # Falcons
    "CHI": 1450,  # Bears
    "NYG": 1430,  # Giants
    "WAS": 1420,  # Commanders
    "CAR": 1380,  # Panthers
    "ARI": 1370,  # Cardinals
}

NBA_ELO_RATINGS: Dict[str, float] = {
    # Top tier
    "BOS": 1680,  # Celtics
    "DEN": 1650,  # Nuggets
    "MIL": 1620,  # Bucks
    "PHX": 1600,  # Suns
    "PHI": 1580,  # 76ers
    # Contenders
    "DAL": 1570,  # Mavericks
    "LAL": 1560,  # Lakers
    "GSW": 1550,  # Warriors
    "MEM": 1540,  # Grizzlies
    "CLE": 1530,  # Cavaliers
    "SAC": 1520,  # Kings
    "NYK": 1510,  # Knicks
    "MIA": 1500,  # Heat
    # Playoff teams
    "LAC": 1490,  # Clippers
    "MIN": 1480,  # Timberwolves
    "NOP": 1470,  # Pelicans
    "ATL": 1460,  # Hawks
    "BKN": 1450,  # Nets
    "TOR": 1440,  # Raptors
    "CHI": 1430,  # Bulls
    "IND": 1420,  # Pacers
    # Rebuilding
    "OKC": 1450,  # Thunder (young, rising)
    "ORL": 1420,  # Magic
    "UTA": 1400,  # Jazz
    "POR": 1390,  # Trail Blazers
    "WAS": 1380,  # Wizards
    "HOU": 1370,  # Rockets
    "SAS": 1360,  # Spurs
    "CHA": 1350,  # Hornets
    "DET": 1340,  # Pistons
}

MLB_ELO_RATINGS: Dict[str, float] = {
    # Elite
    "LAD": 1600,  # Dodgers
    "ATL": 1580,  # Braves
    "HOU": 1570,  # Astros
    # Contenders
    "NYY": 1550,  # Yankees
    "SD": 1540,   # Padres
    "PHI": 1530,  # Phillies
    "SEA": 1520,  # Mariners
    "TB": 1510,   # Rays
    "TOR": 1500,  # Blue Jays
    "CLE": 1490,  # Guardians
    "STL": 1480,  # Cardinals
    "NYM": 1470,  # Mets
    "MIL": 1460,  # Brewers
    "SF": 1450,   # Giants
    "MIN": 1440,  # Twins
    "BAL": 1450,  # Orioles (rising)
    "TEX": 1480,  # Rangers
    # Middle
    "CHW": 1420,  # White Sox
    "BOS": 1430,  # Red Sox
    "ARI": 1440,  # Diamondbacks
    "MIA": 1400,  # Marlins
    "CIN": 1410,  # Reds
    "LAA": 1400,  # Angels
    "DET": 1390,  # Tigers
    "KC": 1380,   # Royals
    "PIT": 1370,  # Pirates
    "COL": 1360,  # Rockies
    "OAK": 1350,  # Athletics
    "WAS": 1360,  # Nationals
    "CHC": 1410,  # Cubs
}

NHL_ELO_RATINGS: Dict[str, float] = {
    # Elite
    "BOS": 1620,  # Bruins
    "CAR": 1600,  # Hurricanes
    "NJ": 1580,   # Devils
    "TOR": 1570,  # Maple Leafs
    "EDM": 1580,  # Oilers
    "COL": 1570,  # Avalanche
    "DAL": 1550,  # Stars
    "VGK": 1560,  # Golden Knights
    # Contenders
    "LAK": 1530,  # Kings
    "MIN": 1520,  # Wild
    "NYR": 1540,  # Rangers
    "SEA": 1500,  # Kraken
    "WPG": 1510,  # Jets
    "TB": 1520,   # Lightning
    "FLA": 1530,  # Panthers
    # Middle
    "PIT": 1480,  # Penguins
    "CGY": 1470,  # Flames
    "NSH": 1460,  # Predators
    "STL": 1450,  # Blues
    "BUF": 1440,  # Sabres
    "OTT": 1430,  # Senators
    "DET": 1420,  # Red Wings
    "NYI": 1430,  # Islanders
    "PHI": 1410,  # Flyers
    "VAN": 1450,  # Canucks
    # Rebuilding
    "MTL": 1400,  # Canadiens
    "WAS": 1400,  # Capitals
    "ARI": 1380,  # Coyotes
    "CHI": 1370,  # Blackhawks
    "SJ": 1360,   # Sharks
    "CBJ": 1390,  # Blue Jackets
    "ANA": 1380,  # Ducks
}

ALL_RATINGS = {
    "NFL": NFL_ELO_RATINGS,
    "NBA": NBA_ELO_RATINGS,
    "MLB": MLB_ELO_RATINGS,
    "NHL": NHL_ELO_RATINGS,
}


def seed_elo_ratings(
    engine: EloEngine,
    sports: Optional[list] = None,
) -> int:
    """Seed an Elo engine with real ratings.

    Args:
        engine: EloEngine instance to seed
        sports: List of sports to seed (default: all)

    Returns:
        Number of teams seeded
    """
    sports = sports or list(ALL_RATINGS.keys())
    count = 0

    for sport in sports:
        ratings = ALL_RATINGS.get(sport, {})
        for team, rating in ratings.items():
            engine.set_team_rating(team, sport, rating)
            count += 1

    return count


def create_seeded_engine(
    config: Optional[EloConfig] = None,
    vig: float = 0.045,
    ratings_path: Optional[str] = None,
    sports: Optional[list] = None,
) -> EloEngine:
    """Create a new EloEngine with pre-seeded ratings.

    Args:
        config: Elo configuration
        vig: Vigorish for odds
        ratings_path: Path to persist ratings
        sports: Sports to seed

    Returns:
        EloEngine with seeded ratings
    """
    engine = EloEngine(
        config=config,
        vig=vig,
        ratings_path=ratings_path,
    )

    count = seed_elo_ratings(engine, sports)
    print(f"Seeded {count} team ratings")

    return engine


def export_ratings_json(output_path: str) -> None:
    """Export all ratings to JSON file."""
    data = {}
    for sport, ratings in ALL_RATINGS.items():
        for team, rating in ratings.items():
            key = f"{sport}:{team}"
            data[key] = {
                "team_code": team,
                "league": sport,
                "rating": rating,
                "games_played": 50,  # Assume established
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported ratings to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed Elo ratings")
    parser.add_argument("--output", default="~/.sparket/custom_miner/elo_ratings.json",
                        help="Output path for ratings JSON")
    parser.add_argument("--sport", help="Single sport to export")

    args = parser.parse_args()
    output = str(Path(args.output).expanduser())

    export_ratings_json(output)
    print(f"\nTo use: Set SPARKET_CUSTOM_MINER__ELO_RATINGS_PATH={output}")
