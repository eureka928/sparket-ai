"""Team strength calculation from statistics.

Calculates a team's "strength rating" from their season record,
home/away splits, recent form, and other available stats.
"""

from __future__ import annotations

from typing import Optional

from sparket.miner.base.data.stats import TeamStats


# Weight parameters for strength calculation
DEFAULT_WEIGHTS = {
    "season": 0.40,      # Season win rate
    "home_away": 0.20,   # Home/away split adjustment
    "recent_form": 0.25, # Last 5 games
    "advanced": 0.15,    # Point differential / other
}

# Bounds for strength rating
MIN_STRENGTH = 0.25
MAX_STRENGTH = 0.75


def calculate_team_strength(
    stats: TeamStats,
    at_home: bool = False,
    weights: Optional[dict] = None,
) -> float:
    """Calculate team strength rating from statistics.
    
    Strength is a value between 0.25 and 0.75 where:
    - 0.50 = league average team
    - 0.75 = elite team (best in league)
    - 0.25 = worst team in league
    
    Args:
        stats: Team statistics
        at_home: Whether team is playing at home
        weights: Optional custom weights for components
    
    Returns:
        Strength rating between MIN_STRENGTH and MAX_STRENGTH
    
    Example:
        stats = TeamStats(team_code="KC", league="NFL", wins=14, losses=2)
        strength = calculate_team_strength(stats, at_home=True)
        # strength ≈ 0.70 (very strong team)
    """
    w = weights or DEFAULT_WEIGHTS
    
    # Component 1: Season win rate (most important)
    season_component = _season_strength(stats)
    
    # Component 2: Home/away adjustment
    home_away_component = _home_away_strength(stats, at_home)
    
    # Component 3: Recent form
    form_component = _recent_form_strength(stats)
    
    # Component 4: Advanced stats (point differential)
    advanced_component = _advanced_strength(stats)
    
    # Weighted combination
    raw_strength = (
        w["season"] * season_component +
        w["home_away"] * home_away_component +
        w["recent_form"] * form_component +
        w["advanced"] * advanced_component
    )
    
    # Clamp to valid range
    return max(MIN_STRENGTH, min(MAX_STRENGTH, raw_strength))


def _season_strength(stats: TeamStats) -> float:
    """Calculate strength from season record.
    
    Maps win rate to strength, with some regression to mean.
    """
    win_rate = stats.win_rate
    
    # Regress to mean based on sample size
    # Fewer games = trust less, pull toward 0.5
    games = stats.games_played
    if games == 0:
        return 0.5
    
    # Regression factor: 1 game -> 90% regression, 16 games -> minimal
    regression = max(0.1, 1.0 - (games / 20.0))
    regressed = win_rate * (1 - regression) + 0.5 * regression
    
    return regressed


def _home_away_strength(stats: TeamStats, at_home: bool) -> float:
    """Adjust strength based on home/away performance.
    
    If playing at home, boost if home record is better than overall.
    If playing away, penalize if away record is worse than overall.
    """
    base = stats.win_rate
    
    if at_home:
        home_rate = stats.home_win_rate
        # Difference from overall
        home_boost = home_rate - base
        return base + (home_boost * 0.5)  # Apply half the difference
    else:
        away_rate = stats.away_win_rate
        away_penalty = away_rate - base
        return base + (away_penalty * 0.5)


def _recent_form_strength(stats: TeamStats) -> float:
    """Calculate strength from recent results.
    
    Hot teams get a boost, cold teams get a penalty.
    """
    recent = stats.recent_form
    overall = stats.win_rate
    
    # How different is recent from overall?
    form_diff = recent - overall
    
    # Apply form as adjustment to overall
    return overall + (form_diff * 0.5)


def _advanced_strength(stats: TeamStats) -> float:
    """Calculate strength from advanced stats.
    
    Currently uses point differential if available.
    Could be extended with ELO, SOS, etc.
    """
    # Check for custom ELO rating
    if stats.elo_rating is not None:
        # Convert ELO to 0-1 scale
        # Assume ELO ranges from ~1200 (bad) to ~1800 (elite)
        # 1500 = average
        elo_normalized = (stats.elo_rating - 1200) / 600
        return max(0, min(1, elo_normalized))
    
    # Use point differential
    point_diff = stats.point_differential
    if point_diff is not None:
        # Map point differential to strength
        # NFL: +10 ppg is elite, -10 ppg is terrible
        # Normalize to ~0.3 to 0.7 range
        diff_component = 0.5 + (point_diff / 40)
        return max(0.3, min(0.7, diff_component))
    
    # Fallback to win rate
    return stats.win_rate








