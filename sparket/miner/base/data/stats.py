"""Team statistics dataclass for strength calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TeamStats:
    """Team statistics for strength calculation.
    
    All fields are optional except team_code and league.
    The model uses whatever data is available.
    
    Miners can extend this by:
    1. Adding data to the 'extra' dict
    2. Subclassing and adding custom fields
    3. Using a custom stats provider that enriches this data
    
    Example:
        # Basic stats from ESPN
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=14,
            losses=2,
            home_wins=7,
            home_losses=1,
        )
        
        # Add custom data
        stats.extra["elo_rating"] = 1650
        stats.extra["strength_of_schedule"] = 0.52
    """
    
    team_code: str
    league: str
    
    # Basic season record
    wins: int = 0
    losses: int = 0
    ties: int = 0
    
    # Derived - set automatically if not provided
    games_played: int = 0
    
    # Home/away splits
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    
    # Recent form (last N games)
    last_5_wins: int = 0
    last_5_losses: int = 0
    
    # Scoring (optional - for point differential)
    points_for: Optional[float] = None
    points_against: Optional[float] = None
    
    # Advanced stats (optional - miners can populate)
    elo_rating: Optional[float] = None
    strength_of_schedule: Optional[float] = None
    
    # Extensible storage for custom data
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Calculate derived fields."""
        if self.games_played == 0:
            self.games_played = self.wins + self.losses + self.ties
    
    @property
    def win_rate(self) -> float:
        """Season win rate (0.0 to 1.0)."""
        if self.games_played == 0:
            return 0.5  # Default to average
        return self.wins / self.games_played
    
    @property
    def home_games(self) -> int:
        """Total home games played."""
        return self.home_wins + self.home_losses
    
    @property
    def away_games(self) -> int:
        """Total away games played."""
        return self.away_wins + self.away_losses
    
    @property
    def home_win_rate(self) -> float:
        """Win rate at home (0.0 to 1.0)."""
        if self.home_games == 0:
            return self.win_rate  # Fall back to overall
        return self.home_wins / self.home_games
    
    @property
    def away_win_rate(self) -> float:
        """Win rate on the road (0.0 to 1.0)."""
        if self.away_games == 0:
            return self.win_rate  # Fall back to overall
        return self.away_wins / self.away_games
    
    @property
    def recent_form(self) -> float:
        """Recent form based on last 5 games (0.0 to 1.0)."""
        recent_total = self.last_5_wins + self.last_5_losses
        if recent_total == 0:
            return self.win_rate  # Fall back to season
        return self.last_5_wins / recent_total
    
    @property
    def point_differential(self) -> Optional[float]:
        """Average point differential per game."""
        if self.points_for is None or self.points_against is None:
            return None
        if self.games_played == 0:
            return 0.0
        return (self.points_for - self.points_against) / self.games_played








