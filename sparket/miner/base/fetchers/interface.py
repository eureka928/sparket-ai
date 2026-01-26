"""Abstract base classes for data fetchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sparket.miner.base.data.stats import TeamStats


@dataclass
class GameResult:
    """Result of a completed game."""
    
    is_final: bool
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    winner: Optional[str] = None  # "HOME", "AWAY", "DRAW"
    status: Optional[str] = None  # "scheduled", "in_progress", "final"
    
    def __post_init__(self) -> None:
        """Derive winner from scores if not provided."""
        if self.is_final and self.winner is None:
            if self.home_score is not None and self.away_score is not None:
                if self.home_score > self.away_score:
                    self.winner = "HOME"
                elif self.away_score > self.home_score:
                    self.winner = "AWAY"
                else:
                    self.winner = "DRAW"


@dataclass
class RecentGame:
    """A recent game result for form calculation."""
    
    date: datetime
    opponent_code: str
    was_home: bool
    score_for: int
    score_against: int
    won: bool


class ScoreFetcher(ABC):
    """Abstract base class for fetching game scores/results.
    
    Used for outcome submission - checking if games are finished
    and what the final scores were.
    """
    
    @abstractmethod
    async def get_result(self, event: Dict[str, Any]) -> Optional[GameResult]:
        """Fetch the result of a game.
        
        Args:
            event: Event info dict with keys:
                - event_id: int
                - home_team: str
                - away_team: str
                - sport: str
                - start_time_utc: datetime or str
        
        Returns:
            GameResult if available, None otherwise.
        """
        pass


class StatsFetcher(ABC):
    """Abstract base class for fetching team statistics.
    
    Used for the team strength model - calculating ratings
    from season records, recent form, etc.
    """
    
    @abstractmethod
    async def get_team_stats(self, team_code: str, league: str) -> Optional[TeamStats]:
        """Fetch statistics for a team.
        
        Args:
            team_code: Short team code (e.g., "KC", "DAL")
            league: League code (e.g., "NFL", "NBA")
        
        Returns:
            TeamStats if available, None otherwise.
        """
        pass
    
    @abstractmethod
    async def get_standings(self, league: str) -> List[TeamStats]:
        """Fetch standings/stats for all teams in a league.
        
        Args:
            league: League code (e.g., "NFL", "NBA")
        
        Returns:
            List of TeamStats for all teams.
        """
        pass
    
    @abstractmethod
    async def get_recent_games(
        self, 
        team_code: str, 
        league: str, 
        limit: int = 5
    ) -> List[RecentGame]:
        """Fetch recent game results for a team.
        
        Args:
            team_code: Short team code
            league: League code
            limit: Maximum games to return
        
        Returns:
            List of recent games, most recent first.
        """
        pass








