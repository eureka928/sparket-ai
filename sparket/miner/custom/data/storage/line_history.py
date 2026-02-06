"""Line movement tracking for detecting sharp money and optimal timing.

Tracks historical odds/lines for markets to:
- Detect steam moves (sudden sharp movements)
- Calculate line movement velocity
- Identify when lines are moving toward or away from our prediction
- Help determine optimal submission timing

Usage:
    history = LineHistory(data_path="/path/to/line_history.json")

    # Record odds as they come in
    history.record(market_id=123, home_prob=0.55, away_prob=0.45)

    # Get movement analysis
    movement = history.get_movement(market_id=123, hours=6)
    # movement.direction = "toward_home" | "toward_away" | "stable"
    # movement.velocity = 0.02  # 2% per hour
    # movement.is_steam = True  # Sharp money detected

    # Get optimal timing
    if movement.is_steam and movement.direction == "toward_home":
        # Sharp money agrees with home, submit now
        pass
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt


@dataclass
class LinePoint:
    """A single point in line history."""
    timestamp: float  # Unix timestamp
    home_prob: float
    away_prob: float
    source: str = "market"  # "market" or "prediction"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "home_prob": self.home_prob,
            "away_prob": self.away_prob,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinePoint":
        return cls(
            timestamp=data["timestamp"],
            home_prob=data["home_prob"],
            away_prob=data["away_prob"],
            source=data.get("source", "market"),
        )


@dataclass
class LineMovement:
    """Analysis of line movement over a time window."""
    market_id: int
    direction: str  # "toward_home", "toward_away", "stable"
    velocity: float  # Change in home_prob per hour
    total_change: float  # Total change in home_prob
    is_steam: bool  # Sudden sharp movement detected
    points_analyzed: int
    window_hours: float
    current_home_prob: float
    start_home_prob: float

    @property
    def is_significant(self) -> bool:
        """Movement is significant if > 2% total change."""
        return abs(self.total_change) > 0.02

    @property
    def moving_toward_home(self) -> bool:
        return self.direction == "toward_home"

    @property
    def moving_toward_away(self) -> bool:
        return self.direction == "toward_away"


@dataclass
class SteamMove:
    """A detected steam move (sharp money signal)."""
    market_id: int
    detected_at: float  # Unix timestamp
    direction: str  # "home" or "away"
    magnitude: float  # Size of the move
    velocity: float  # How fast it happened

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "detected_at": self.detected_at,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "velocity": self.velocity,
        }


class LineHistory:
    """Tracks line/odds history for markets.

    Stores historical odds movements and provides analysis tools for:
    - Detecting steam moves (sharp money)
    - Calculating movement velocity and direction
    - Determining optimal submission timing
    """

    # Steam move thresholds
    STEAM_MIN_VELOCITY = 0.03  # 3% per hour minimum
    STEAM_MIN_MAGNITUDE = 0.02  # 2% minimum move
    STEAM_MAX_WINDOW_MINUTES = 30  # Must happen within 30 min

    # Movement direction thresholds
    STABLE_THRESHOLD = 0.005  # < 0.5% change = stable

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_history_hours: float = 168.0,  # 7 days
    ) -> None:
        """Initialize line history tracker.

        Args:
            data_path: Path to persist history (JSON file)
            max_history_hours: Maximum history to keep per market
        """
        self._data_path = Path(data_path) if data_path else None
        self._max_history_seconds = max_history_hours * 3600

        # market_id -> list of LinePoints (sorted by timestamp)
        self._history: Dict[int, List[LinePoint]] = {}

        # Recent steam moves for reference
        self._steam_moves: List[SteamMove] = []

        if self._data_path and self._data_path.exists():
            self._load()

    def record(
        self,
        market_id: int,
        home_prob: float,
        away_prob: float,
        source: str = "market",
        timestamp: Optional[float] = None,
    ) -> Optional[SteamMove]:
        """Record a line/odds point for a market.

        Args:
            market_id: Market identifier
            home_prob: Home team probability
            away_prob: Away team probability
            source: "market" for external odds, "prediction" for our model
            timestamp: Unix timestamp (defaults to now)

        Returns:
            SteamMove if sharp movement detected, None otherwise
        """
        ts = timestamp or time.time()

        point = LinePoint(
            timestamp=ts,
            home_prob=home_prob,
            away_prob=away_prob,
            source=source,
        )

        if market_id not in self._history:
            self._history[market_id] = []

        # Add point and keep sorted
        self._history[market_id].append(point)
        self._history[market_id].sort(key=lambda p: p.timestamp)

        # Prune old history
        self._prune_market(market_id)

        # Check for steam move
        steam = self._detect_steam(market_id)
        if steam:
            self._steam_moves.append(steam)
            bt.logging.info({
                "line_history": "steam_detected",
                "market_id": market_id,
                "direction": steam.direction,
                "magnitude": round(steam.magnitude, 3),
                "velocity": round(steam.velocity, 3),
            })

        # Periodic save
        if len(self._history[market_id]) % 10 == 0:
            self._save()

        return steam

    def get_movement(
        self,
        market_id: int,
        hours: float = 6.0,
    ) -> Optional[LineMovement]:
        """Analyze line movement over a time window.

        Args:
            market_id: Market identifier
            hours: Hours to look back

        Returns:
            LineMovement analysis or None if insufficient data
        """
        if market_id not in self._history:
            return None

        points = self._history[market_id]
        if len(points) < 2:
            return None

        now = time.time()
        window_start = now - (hours * 3600)

        # Get points in window
        window_points = [p for p in points if p.timestamp >= window_start]
        if len(window_points) < 2:
            # Fall back to last 2 points if window is empty
            window_points = points[-2:]

        first = window_points[0]
        last = window_points[-1]

        total_change = last.home_prob - first.home_prob
        time_diff_hours = (last.timestamp - first.timestamp) / 3600

        if time_diff_hours > 0:
            velocity = total_change / time_diff_hours
        else:
            velocity = 0.0

        # Determine direction
        if total_change > self.STABLE_THRESHOLD:
            direction = "toward_home"
        elif total_change < -self.STABLE_THRESHOLD:
            direction = "toward_away"
        else:
            direction = "stable"

        # Check for steam in this window
        is_steam = self._has_recent_steam(market_id, hours)

        return LineMovement(
            market_id=market_id,
            direction=direction,
            velocity=velocity,
            total_change=total_change,
            is_steam=is_steam,
            points_analyzed=len(window_points),
            window_hours=time_diff_hours,
            current_home_prob=last.home_prob,
            start_home_prob=first.home_prob,
        )

    def get_consensus_prob(
        self,
        market_id: int,
        hours: float = 24.0,
    ) -> Optional[Tuple[float, float]]:
        """Get time-weighted average probability over window.

        Args:
            market_id: Market identifier
            hours: Hours to average over

        Returns:
            (home_prob, away_prob) tuple or None
        """
        if market_id not in self._history:
            return None

        points = self._history[market_id]
        if not points:
            return None

        now = time.time()
        window_start = now - (hours * 3600)

        # Filter to market-only sources in window (exclude our own predictions)
        window_points = [p for p in points if p.timestamp >= window_start and p.source == "market"]
        if not window_points:
            # Fall back to most recent market point
            market_points = [p for p in points if p.source == "market"]
            if market_points:
                return (market_points[-1].home_prob, market_points[-1].away_prob)
            return None  # No market data at all

        # Time-weighted average (more recent = more weight)
        total_weight = 0.0
        weighted_home = 0.0
        weighted_away = 0.0

        for point in window_points:
            # Linear weight: recent points weighted higher
            age_hours = (now - point.timestamp) / 3600
            weight = max(0.1, 1.0 - (age_hours / hours))

            weighted_home += point.home_prob * weight
            weighted_away += point.away_prob * weight
            total_weight += weight

        if total_weight > 0:
            return (weighted_home / total_weight, weighted_away / total_weight)
        return None

    def should_submit_now(
        self,
        market_id: int,
        our_home_prob: float,
        hours_to_game: float,
    ) -> Tuple[bool, str]:
        """Determine if now is a good time to submit.

        Uses line movement to decide timing:
        - If line is moving toward our prediction, wait (may get better)
        - If line is moving away from our prediction, submit now
        - If steam detected agreeing with us, submit immediately
        - If close to game time, submit anyway

        Args:
            market_id: Market identifier
            our_home_prob: Our model's home probability
            hours_to_game: Hours until game starts

        Returns:
            (should_submit, reason) tuple
        """
        # Always submit if very close to game time
        if hours_to_game < 1.0:
            return (True, "game_imminent")

        movement = self.get_movement(market_id, hours=6.0)
        if not movement:
            return (True, "no_movement_data")

        # Our edge: how much we disagree with market
        our_edge = our_home_prob - movement.current_home_prob

        # Check for steam move
        if movement.is_steam:
            if movement.moving_toward_home and our_edge > 0.02:
                # Sharp money agrees with our home lean
                return (True, "steam_agrees_home")
            elif movement.moving_toward_away and our_edge < -0.02:
                # Sharp money agrees with our away lean
                return (True, "steam_agrees_away")
            elif abs(our_edge) > 0.05:
                # We strongly disagree with steam, still submit
                return (True, "strong_conviction")
            else:
                # Steam disagrees with us, maybe wait
                return (False, "steam_disagrees")

        # No steam, check if line is moving toward or away from us
        if movement.is_significant:
            if movement.moving_toward_home and our_edge > 0:
                # Line moving toward our home prediction, wait
                return (False, "line_moving_toward_us")
            elif movement.moving_toward_away and our_edge < 0:
                # Line moving toward our away prediction, wait
                return (False, "line_moving_toward_us")
            elif movement.moving_toward_home and our_edge < 0:
                # Line moving away from our away prediction, submit
                return (True, "line_moving_away")
            elif movement.moving_toward_away and our_edge > 0:
                # Line moving away from our home prediction, submit
                return (True, "line_moving_away")

        # Default: submit if we have meaningful edge
        if abs(our_edge) > 0.03:
            return (True, "has_edge")

        return (True, "default")

    def get_recent_steam_moves(
        self,
        hours: float = 24.0,
    ) -> List[SteamMove]:
        """Get recent steam moves across all markets."""
        cutoff = time.time() - (hours * 3600)
        return [s for s in self._steam_moves if s.detected_at >= cutoff]

    def _detect_steam(self, market_id: int) -> Optional[SteamMove]:
        """Detect if a steam move just happened.

        Steam move = sudden sharp line movement indicating sharp money.
        """
        if market_id not in self._history:
            return None

        points = self._history[market_id]
        if len(points) < 3:
            return None

        now = time.time()
        window_start = now - (self.STEAM_MAX_WINDOW_MINUTES * 60)

        # Get recent points
        recent = [p for p in points if p.timestamp >= window_start]
        if len(recent) < 2:
            return None

        # Check for sharp movement
        first = recent[0]
        last = recent[-1]

        magnitude = last.home_prob - first.home_prob
        time_diff_hours = (last.timestamp - first.timestamp) / 3600

        if time_diff_hours <= 0:
            return None

        velocity = abs(magnitude) / time_diff_hours

        # Check thresholds
        if velocity < self.STEAM_MIN_VELOCITY:
            return None
        if abs(magnitude) < self.STEAM_MIN_MAGNITUDE:
            return None

        # Steam detected!
        direction = "home" if magnitude > 0 else "away"

        return SteamMove(
            market_id=market_id,
            detected_at=now,
            direction=direction,
            magnitude=abs(magnitude),
            velocity=velocity,
        )

    def _has_recent_steam(self, market_id: int, hours: float) -> bool:
        """Check if market has recent steam move."""
        cutoff = time.time() - (hours * 3600)
        return any(
            s.market_id == market_id and s.detected_at >= cutoff
            for s in self._steam_moves
        )

    def _prune_market(self, market_id: int) -> None:
        """Remove old history for a market."""
        if market_id not in self._history:
            return

        cutoff = time.time() - self._max_history_seconds
        self._history[market_id] = [
            p for p in self._history[market_id]
            if p.timestamp >= cutoff
        ]

    def cleanup_old_markets(self, max_age_hours: float = 168.0) -> int:
        """Remove markets with no recent activity.

        Args:
            max_age_hours: Remove markets with no points newer than this

        Returns:
            Number of markets removed
        """
        cutoff = time.time() - (max_age_hours * 3600)

        stale = []
        for market_id, points in self._history.items():
            if not points or points[-1].timestamp < cutoff:
                stale.append(market_id)

        for market_id in stale:
            del self._history[market_id]

        if stale:
            bt.logging.debug({
                "line_history": "cleanup",
                "removed_markets": len(stale),
            })
            self._save()

        return len(stale)

    def _save(self) -> None:
        """Persist history to disk."""
        if not self._data_path:
            return

        try:
            self._data_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "history": {
                    str(market_id): [p.to_dict() for p in points]
                    for market_id, points in self._history.items()
                },
                "steam_moves": [s.to_dict() for s in self._steam_moves[-100:]],
            }

            with open(self._data_path, "w") as f:
                json.dump(data, f)

        except Exception as e:
            bt.logging.warning({
                "line_history": "save_failed",
                "error": str(e),
            })

    def _load(self) -> None:
        """Load history from disk."""
        if not self._data_path or not self._data_path.exists():
            return

        try:
            with open(self._data_path) as f:
                data = json.load(f)

            # Load history
            for market_id_str, points_data in data.get("history", {}).items():
                market_id = int(market_id_str)
                self._history[market_id] = [
                    LinePoint.from_dict(p) for p in points_data
                ]

            # Load steam moves
            for steam_data in data.get("steam_moves", []):
                self._steam_moves.append(SteamMove(
                    market_id=steam_data["market_id"],
                    detected_at=steam_data["detected_at"],
                    direction=steam_data["direction"],
                    magnitude=steam_data["magnitude"],
                    velocity=steam_data["velocity"],
                ))

            bt.logging.info({
                "line_history": "loaded",
                "markets": len(self._history),
                "steam_moves": len(self._steam_moves),
            })

        except Exception as e:
            bt.logging.warning({
                "line_history": "load_failed",
                "error": str(e),
            })

    def stats(self) -> Dict[str, Any]:
        """Get statistics about tracked history."""
        total_points = sum(len(pts) for pts in self._history.values())

        return {
            "markets_tracked": len(self._history),
            "total_points": total_points,
            "recent_steam_moves": len(self.get_recent_steam_moves(24.0)),
            "avg_points_per_market": total_points / max(1, len(self._history)),
        }
