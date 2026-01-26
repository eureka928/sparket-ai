"""Localnet test harness for orchestrating E2E tests.

The harness manages:
- Validator control API connection
- Multiple miner instances via MinerPool
- Time progression via TimeController
- Database access for verification
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, List, Type

import aiohttp
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .config import LocalnetConfig, DEFAULT_CONFIG
from .miner_pool import MinerPool
from .time_controller import TimeController
from .scenarios.base import BaseScenario, ScenarioResult


class ValidatorClient:
    """HTTP client for validator control API."""
    
    def __init__(self, control_url: str):
        self._url = control_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=120)
    
    async def _get(self, path: str) -> dict:
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.get(f"{self._url}{path}") as resp:
                return await resp.json()
    
    async def _post(self, path: str, data: dict | None = None) -> dict:
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(f"{self._url}{path}", json=data or {}) as resp:
                return await resp.json()
    
    async def health_check(self) -> bool:
        """Check if validator control API is healthy."""
        try:
            result = await self._get("/health")
            return result.get("status") == "ok"
        except Exception:
            return False
    
    async def wipe_db(self) -> dict:
        """Wipe test database for clean state."""
        return await self._post("/admin/wipe-db")
    
    async def sync_miners(self) -> dict:
        """Sync miners from metagraph to database."""
        return await self._post("/admin/sync-miners")
    
    async def create_event(
        self,
        home_team: str,
        away_team: str,
        hours_ahead: int = 48,
        status: str = "scheduled",
    ) -> dict:
        """Create a mock event."""
        start_time = datetime.now(timezone.utc) + timedelta(hours=hours_ahead)
        return await self._post("/mock/event", {
            "home_team": home_team,
            "away_team": away_team,
            "start_time": start_time.isoformat(),
            "status": status,
        })
    
    async def seed_ground_truth(
        self,
        market_id: int,
        home_prob: float,
        away_prob: float,
    ) -> dict:
        """Seed ground truth closing for a market."""
        home_odds = round(1.0 / home_prob, 2) if home_prob > 0 else 100.0
        away_odds = round(1.0 / away_prob, 2) if away_prob > 0 else 100.0
        
        return await self._post("/mock/ground-truth-closing", {
            "market_id": market_id,
            "sides": [
                {"side": "HOME", "prob_consensus": home_prob, "odds_consensus": home_odds},
                {"side": "AWAY", "prob_consensus": away_prob, "odds_consensus": away_odds},
            ]
        })
    
    # --- Enhanced MockProvider methods ---
    
    async def get_sportsbooks(self) -> dict:
        """Get list of configured sportsbooks."""
        return await self._get("/mock/sportsbooks")
    
    async def add_sportsbook(
        self,
        code: str,
        name: str,
        is_sharp: bool = False,
        vig: float = 0.04,
        noise: float = 0.02,
    ) -> dict:
        """Add or update a sportsbook."""
        return await self._post("/mock/sportsbook", {
            "code": code,
            "name": name,
            "is_sharp": is_sharp,
            "vig": vig,
            "noise": noise,
        })
    
    async def add_sportsbook_odds(
        self,
        market_id: str,
        sportsbook_code: str,
        home_odds: float,
        away_odds: float,
        timestamp: str | None = None,
    ) -> dict:
        """Add odds from a specific sportsbook."""
        data = {
            "market_id": market_id,
            "sportsbook_code": sportsbook_code,
            "home_odds": home_odds,
            "away_odds": away_odds,
        }
        if timestamp:
            data["timestamp"] = timestamp
        return await self._post("/mock/sportsbook-odds", data)
    
    async def generate_timeseries(
        self,
        market_id: str,
        true_prob_home: float,
        open_time: datetime,
        close_time: datetime,
        interval_hours: int = 6,
        sportsbook_codes: list[str] | None = None,
        seed: int | None = None,
    ) -> dict:
        """Generate time-series odds for a market."""
        data = {
            "market_id": market_id,
            "true_prob_home": true_prob_home,
            "open_time": open_time.isoformat(),
            "close_time": close_time.isoformat(),
            "interval_hours": interval_hours,
        }
        if sportsbook_codes:
            data["sportsbook_codes"] = sportsbook_codes
        if seed is not None:
            data["seed"] = seed
        return await self._post("/mock/timeseries", data)
    
    async def seed_ground_truth_from_timeseries(
        self,
        mock_market_id: str,
        db_market_id: int,
    ) -> dict:
        """Seed ground truth tables from mock provider time-series."""
        return await self._post("/mock/seed-ground-truth-from-timeseries", {
            "market_id": mock_market_id,
            "db_market_id": db_market_id,
        })
    
    async def get_closing_odds(
        self,
        market_id: str,
        sportsbook_code: str | None = None,
    ) -> dict:
        """Get closing odds from mock provider."""
        params = f"?market_id={market_id}"
        if sportsbook_code:
            params += f"&sportsbook_code={sportsbook_code}"
        return await self._get(f"/mock/closing-odds{params}")
    
    async def get_consensus(
        self,
        market_id: str,
        side: str = "HOME",
    ) -> dict:
        """Get consensus closing odds from mock provider."""
        return await self._get(f"/mock/consensus?market_id={market_id}&side={side}")
    
    async def seed_outcome(
        self,
        market_id: int,
        result: str,
        home_score: int = 1,
        away_score: int = 0,
    ) -> dict:
        """Seed outcome for a market."""
        return await self._post("/mock/settled-outcome", {
            "market_id": market_id,
            "result": result,
            "score_home": home_score,
            "score_away": away_score,
        })
    
    async def trigger_odds_scoring(self) -> dict:
        """Trigger CLV/CLE scoring."""
        return await self._post("/trigger/odds-scoring")
    
    async def trigger_outcome_scoring(self) -> dict:
        """Trigger Brier/PSS scoring."""
        return await self._post("/trigger/outcome-scoring")
    
    async def trigger_scoring(self) -> dict:
        """Trigger full scoring pipeline."""
        return await self._post("/trigger/scoring")
    
    async def backdate_submissions(self, days: int = 1) -> dict:
        """Backdate recent submissions for scoring window."""
        return await self._post("/mock/backdate-submissions", {"days_back": days})
    
    async def get_submissions(self) -> dict:
        """Get submissions from database."""
        return await self._get("/db/submissions")
    
    async def get_scores(self) -> dict:
        """Get miner scores."""
        return await self._get("/db/scores")
    
    async def get_rolling_scores(self) -> dict:
        """Get rolling aggregate scores."""
        return await self._get("/db/rolling-scores")
    
    async def get_skill_scores(self) -> dict:
        """Get final skill scores."""
        return await self._get("/db/skill-scores")
    
    async def get_weights(self) -> dict:
        """Get computed weights."""
        return await self._get("/db/weights")
    
    async def get_memory(self) -> dict:
        """Get validator memory usage."""
        return await self._get("/health/memory")


class LocalnetHarness:
    """Orchestrates E2E tests with validator + N miners.
    
    Example usage:
        async with LocalnetHarness() as harness:
            await harness.setup_clean_state()
            
            # Create events
            events = await harness.validator.create_event(...)
            
            # Miners submit
            await harness.miners[0].submit_odds(...)
            
            # Verify scores
            scores = await harness.validator.get_skill_scores()
    """
    
    def __init__(self, config: LocalnetConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        
        # Components (initialized in setup)
        self.validator: ValidatorClient
        self.miners: MinerPool
        self.time_controller: TimeController
        self.db: AsyncEngine | None = None
        
        self._initialized = False
    
    async def __aenter__(self) -> LocalnetHarness:
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.teardown()
    
    async def setup(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        # Initialize validator client
        self.validator = ValidatorClient(self.config.validator_control_url)
        
        # Initialize miner pool
        self.miners = MinerPool(self.config.miners)
        
        # Initialize time controller
        self.time_controller = TimeController(self.config.database_url)
        await self.time_controller.connect()
        
        # Initialize database connection
        self.db = create_async_engine(
            self.config.database_url,
            pool_size=5,
            max_overflow=2,
        )
        
        self._initialized = True
    
    async def teardown(self) -> None:
        """Clean up resources."""
        if self.time_controller:
            await self.time_controller.close()
        
        if self.db:
            await self.db.dispose()
            self.db = None
        
        self._initialized = False
    
    async def health_check(self) -> dict[str, bool]:
        """Check health of all components."""
        results = {}
        
        # Validator
        results["validator"] = await self.validator.health_check()
        
        # Miners
        miner_health = await self.miners.health_check_all()
        results.update(miner_health)
        
        # Database
        try:
            async with self.db.connect() as conn:
                await conn.execute(text("SELECT 1"))
            results["database"] = True
        except Exception:
            results["database"] = False
        
        return results
    
    async def setup_clean_state(self) -> dict:
        """Reset database to clean state and sync miners.
        
        Call at the start of each test for isolation.
        """
        results = {}
        
        # Wipe database
        wipe_result = await self.validator.wipe_db()
        results["wipe"] = wipe_result
        
        # Sync miners from metagraph
        sync_result = await self.validator.sync_miners()
        results["sync"] = sync_result
        
        return results
    
    async def run_scenario(
        self,
        scenario_class: Type[BaseScenario],
    ) -> ScenarioResult:
        """Execute a test scenario.
        
        Args:
            scenario_class: The scenario class to run
            
        Returns:
            ScenarioResult with pass/fail status and metrics
        """
        scenario = scenario_class(self)
        return await scenario.run()
    
    async def run_scoring_cycle(self) -> dict:
        """Run a complete scoring cycle.
        
        1. Trigger odds scoring (CLV/CLE)
        2. Trigger outcome scoring (Brier/PSS)
        3. Trigger main scoring pipeline (aggregation)
        
        Returns summary of all results.
        """
        results = {}
        
        # Odds scoring
        results["odds"] = await self.validator.trigger_odds_scoring()
        
        # Outcome scoring
        results["outcome"] = await self.validator.trigger_outcome_scoring()
        
        # Main pipeline
        results["main"] = await self.validator.trigger_scoring()
        
        return results
    
    async def create_test_events(
        self,
        n: int = 5,
        hours_ahead: int = 48,
    ) -> List[dict]:
        """Create N test events with markets.
        
        Returns list of created events with db_market_id.
        """
        events = []
        for i in range(n):
            result = await self.validator.create_event(
                home_team=f"Team{i*2}",
                away_team=f"Team{i*2+1}",
                hours_ahead=hours_ahead,
            )
            if result.get("status") == "ok":
                events.append(result.get("event", {}))
        return events
    
    async def seed_ground_truth_for_events(
        self,
        events: List[dict],
    ) -> dict[int, dict]:
        """Seed ground truth closing for all events.
        
        Returns dict mapping market_id to ground truth.
        """
        ground_truth = {}
        
        for i, event in enumerate(events):
            market_id = event.get("db_market_id")
            if market_id is None:
                continue
            
            # Varied probabilities
            home_prob = 0.3 + (i % 5) * 0.1
            away_prob = round(1.0 - home_prob, 4)
            
            await self.validator.seed_ground_truth(
                market_id=market_id,
                home_prob=home_prob,
                away_prob=away_prob,
            )
            
            ground_truth[market_id] = {
                "home_prob": home_prob,
                "away_prob": away_prob,
                "home_odds": round(1.0 / home_prob, 2),
                "away_odds": round(1.0 / away_prob, 2),
            }
        
        return ground_truth
    
    async def seed_outcomes_for_events(
        self,
        events: List[dict],
        ground_truth: dict[int, dict],
    ) -> dict[int, str]:
        """Seed outcomes based on ground truth probabilities.
        
        Higher probability side wins more often (realistic).
        """
        import random
        
        outcomes = {}
        
        for event in events:
            market_id = event.get("db_market_id")
            if market_id is None:
                continue
            
            gt = ground_truth.get(market_id, {})
            home_prob = gt.get("home_prob", 0.5)
            
            # Probabilistic outcome
            if random.random() < home_prob:
                result = "HOME"
                home_score, away_score = 2, 1
            else:
                result = "AWAY"
                home_score, away_score = 0, 1
            
            await self.validator.seed_outcome(
                market_id=market_id,
                result=result,
                home_score=home_score,
                away_score=away_score,
            )
            
            outcomes[market_id] = result
        
        return outcomes
