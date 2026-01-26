"""Time controller for simulating time progression in E2E tests.

The scoring system uses time windows (30-day rolling, 60-day calibration).
This controller allows tests to simulate time passing by shifting timestamps
in the database.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


class TimeController:
    """Simulates time progression for scoring windows.
    
    Allows tests to:
    - Advance time by N days (shifts all timestamps)
    - Transition events to finished state
    - Backdate submissions for scoring window inclusion
    """
    
    def __init__(self, database_url: str):
        self._database_url = database_url
        self._engine: AsyncEngine | None = None
        self._time_offset_days: int = 0
    
    async def connect(self) -> None:
        """Establish database connection."""
        if self._engine is None:
            self._engine = create_async_engine(
                self._database_url,
                pool_size=5,
                max_overflow=2,
            )
    
    async def close(self) -> None:
        """Close database connection."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
    
    @property
    def current_offset_days(self) -> int:
        """Total days offset from real time."""
        return self._time_offset_days
    
    async def advance_days(self, days: int) -> None:
        """Shift all relevant timestamps back by N days.
        
        This makes data appear older, placing it within scoring windows.
        Tables affected:
        - miner_submission.submitted_at
        - outcome.settled_at
        - submission_vs_close.close_ts, computed_at
        - submission_outcome_score.settled_at
        - event.start_time_utc
        - ground_truth_closing.computed_at
        - ground_truth_snapshot.snapshot_ts
        """
        if self._engine is None:
            raise RuntimeError("TimeController not connected")
        
        self._time_offset_days += days
        interval = f"{days} days"
        
        async with self._engine.begin() as conn:
            # Submissions
            await conn.execute(text(f"""
                UPDATE miner_submission
                SET submitted_at = submitted_at - INTERVAL '{interval}'
            """))
            
            # Outcomes
            await conn.execute(text(f"""
                UPDATE outcome
                SET settled_at = settled_at - INTERVAL '{interval}'
                WHERE settled_at IS NOT NULL
            """))
            
            # Submission vs close scores
            await conn.execute(text(f"""
                UPDATE submission_vs_close
                SET close_ts = close_ts - INTERVAL '{interval}',
                    computed_at = computed_at - INTERVAL '{interval}'
            """))
            
            # Submission outcome scores
            await conn.execute(text(f"""
                UPDATE submission_outcome_score
                SET settled_at = settled_at - INTERVAL '{interval}'
                WHERE settled_at IS NOT NULL
            """))
            
            # Events
            await conn.execute(text(f"""
                UPDATE event
                SET start_time_utc = start_time_utc - INTERVAL '{interval}'
            """))
            
            # Ground truth
            await conn.execute(text(f"""
                UPDATE ground_truth_closing
                SET computed_at = computed_at - INTERVAL '{interval}'
            """))
            
            await conn.execute(text(f"""
                UPDATE ground_truth_snapshot
                SET snapshot_ts = snapshot_ts - INTERVAL '{interval}'
            """))
    
    async def transition_event_to_finished(
        self,
        event_id: int,
        home_score: int = 1,
        away_score: int = 0,
    ) -> dict:
        """Transition an event to finished state with outcome.
        
        Returns:
            Dict with event_id, market_id, result
        """
        if self._engine is None:
            raise RuntimeError("TimeController not connected")
        
        # Determine result
        if home_score > away_score:
            result = "HOME"
        elif away_score > home_score:
            result = "AWAY"
        else:
            result = "DRAW"
        
        now = datetime.now(timezone.utc)
        
        async with self._engine.begin() as conn:
            # Update event status
            await conn.execute(text("""
                UPDATE event SET status = 'finished'
                WHERE event_id = :event_id
            """), {"event_id": event_id})
            
            # Get market_id for this event
            result_row = await conn.execute(text("""
                SELECT market_id FROM market WHERE event_id = :event_id LIMIT 1
            """), {"event_id": event_id})
            market_row = result_row.fetchone()
            
            if market_row is None:
                raise ValueError(f"No market found for event {event_id}")
            
            market_id = market_row[0]
            
            # Insert outcome
            await conn.execute(text("""
                INSERT INTO outcome (market_id, result, score_home, score_away, settled_at, details)
                VALUES (:market_id, :result, :home_score, :away_score, :settled_at, '{}'::jsonb)
                ON CONFLICT (market_id) DO UPDATE SET
                    result = EXCLUDED.result,
                    score_home = EXCLUDED.score_home,
                    score_away = EXCLUDED.score_away,
                    settled_at = EXCLUDED.settled_at
            """), {
                "market_id": market_id,
                "result": result,
                "home_score": home_score,
                "away_score": away_score,
                "settled_at": now,
            })
        
        return {
            "event_id": event_id,
            "market_id": market_id,
            "result": result,
            "home_score": home_score,
            "away_score": away_score,
        }
    
    async def backdate_recent_data(self, days: int = 1) -> dict:
        """Backdate all data from the last 24 hours.
        
        Useful after seeding test data to place it within scoring windows.
        
        Returns:
            Dict with counts of updated rows per table.
        """
        if self._engine is None:
            raise RuntimeError("TimeController not connected")
        
        interval = f"{days} days"
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        counts = {}
        
        async with self._engine.begin() as conn:
            # Submissions
            result = await conn.execute(text(f"""
                UPDATE miner_submission
                SET submitted_at = submitted_at - INTERVAL '{interval}'
                WHERE submitted_at > :cutoff
            """), {"cutoff": cutoff})
            counts["submissions"] = result.rowcount
            
            # Outcomes
            result = await conn.execute(text(f"""
                UPDATE outcome
                SET settled_at = settled_at - INTERVAL '{interval}'
                WHERE settled_at > :cutoff
            """), {"cutoff": cutoff})
            counts["outcomes"] = result.rowcount
            
            # Events
            result = await conn.execute(text(f"""
                UPDATE event
                SET start_time_utc = start_time_utc - INTERVAL '{interval}'
                WHERE start_time_utc > :cutoff
            """), {"cutoff": cutoff})
            counts["events"] = result.rowcount
        
        return counts
