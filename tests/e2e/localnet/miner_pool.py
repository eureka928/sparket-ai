"""Miner pool for managing multiple miner instances in E2E tests.

Provides a high-level interface for controlling miners via their HTTP control APIs.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List

import aiohttp

from .config import MinerConfig


@dataclass
class OddsSubmission:
    """Represents an odds submission from a miner."""
    
    market_id: int
    home_odds: float
    away_odds: float
    home_prob: float
    away_prob: float


class MinerClient:
    """HTTP client for controlling a single miner via its control API."""
    
    def __init__(self, config: MinerConfig):
        self.config = config
        self._timeout = aiohttp.ClientTimeout(total=60)
    
    @property
    def wallet_name(self) -> str:
        return self.config.wallet_name
    
    @property
    def uid(self) -> int | None:
        return self.config.uid
    
    @property
    def control_url(self) -> str:
        return self.config.control_url
    
    async def health_check(self) -> bool:
        """Check if miner control API is healthy."""
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(f"{self.control_url}/health") as resp:
                    data = await resp.json()
                    return data.get("status") == "ok"
        except Exception:
            return False
    
    async def fetch_games(self) -> dict:
        """Trigger miner to fetch game data from validator."""
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(f"{self.control_url}/action/fetch-games") as resp:
                return await resp.json()
    
    async def submit_odds(
        self,
        market_id: int,
        home_odds: float,
        away_odds: float,
    ) -> dict:
        """Submit odds for a market via miner control API.
        
        The miner forwards submissions to the validator via dendrite.
        """
        home_prob = round(1.0 / home_odds, 4) if home_odds > 0 else 0.5
        away_prob = round(1.0 / away_odds, 4) if away_odds > 0 else 0.5
        
        payload = {
            "submissions": [{
                "market_id": market_id,
                "kind": "MONEYLINE",
                "prices": [
                    {"side": "HOME", "odds_eu": home_odds, "imp_prob": home_prob},
                    {"side": "AWAY", "odds_eu": away_odds, "imp_prob": away_prob},
                ]
            }]
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.post(
                    f"{self.control_url}/action/submit-odds",
                    json=payload
                ) as resp:
                    return await resp.json()
        except aiohttp.ClientError as e:
            return {"status": "error", "message": str(e)}
    
    async def submit_outcome(
        self,
        event_id: int,
        result: str,
        home_score: int,
        away_score: int,
    ) -> dict:
        """Submit an outcome for an event."""
        payload = {
            "event_id": event_id,
            "result": result,
            "score_home": home_score,
            "score_away": away_score,
        }
        
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                f"{self.control_url}/action/submit-outcome",
                json=payload
            ) as resp:
                return await resp.json()
    
    async def submit_early_accurate(
        self,
        markets: List[dict],
        ground_truth: dict[int, dict],
    ) -> List[dict]:
        """Submit early, accurate odds (close to ground truth).
        
        Strategy: Submit odds very close to ground truth for high CLV.
        """
        results = []
        for market in markets:
            market_id = market["market_id"]
            gt = ground_truth.get(market_id, {})
            
            # Accurate: very close to ground truth
            home_prob = gt.get("home_prob", 0.5)
            away_prob = gt.get("away_prob", 0.5)
            
            # Small random deviation (±2%)
            home_prob = max(0.05, min(0.95, home_prob + random.uniform(-0.02, 0.02)))
            away_prob = 1.0 - home_prob
            
            home_odds = round(1.0 / home_prob, 2)
            away_odds = round(1.0 / away_prob, 2)
            
            result = await self.submit_odds(market_id, home_odds, away_odds)
            results.append(result)
        
        return results
    
    async def submit_late_accurate(
        self,
        markets: List[dict],
        ground_truth: dict[int, dict],
        delay_seconds: float = 0,
    ) -> List[dict]:
        """Submit late but accurate odds.
        
        Strategy: Accurate odds but submitted late (lower time bonus).
        """
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        
        return await self.submit_early_accurate(markets, ground_truth)
    
    async def submit_inaccurate(
        self,
        markets: List[dict],
        ground_truth: dict[int, dict],
    ) -> List[dict]:
        """Submit inaccurate odds (far from ground truth).
        
        Strategy: Submit odds with large deviation from ground truth.
        """
        results = []
        for market in markets:
            market_id = market["market_id"]
            gt = ground_truth.get(market_id, {})
            
            # Inaccurate: large deviation from ground truth
            home_prob = gt.get("home_prob", 0.5)
            
            # Flip probability or add large deviation
            if random.random() > 0.5:
                home_prob = 1.0 - home_prob  # Flip
            else:
                home_prob = max(0.1, min(0.9, home_prob + random.uniform(-0.3, 0.3)))
            
            away_prob = 1.0 - home_prob
            
            home_odds = round(1.0 / home_prob, 2)
            away_odds = round(1.0 / away_prob, 2)
            
            result = await self.submit_odds(market_id, home_odds, away_odds)
            results.append(result)
        
        return results
    
    async def submit_market_copy(
        self,
        markets: List[dict],
        ground_truth: dict[int, dict],
    ) -> List[dict]:
        """Submit odds identical to market (copy-trading).
        
        Strategy: Exact copy of ground truth (low originality score).
        """
        results = []
        for market in markets:
            market_id = market["market_id"]
            gt = ground_truth.get(market_id, {})
            
            # Exact copy
            home_prob = gt.get("home_prob", 0.5)
            away_prob = gt.get("away_prob", 0.5)
            
            home_odds = round(1.0 / home_prob, 2) if home_prob > 0 else 100.0
            away_odds = round(1.0 / away_prob, 2) if away_prob > 0 else 100.0
            
            result = await self.submit_odds(market_id, home_odds, away_odds)
            results.append(result)
        
        return results


class MinerPool:
    """Manages multiple miner instances for E2E testing.
    
    Provides coordinated control over a pool of miners.
    """
    
    def __init__(self, configs: List[MinerConfig]):
        self.miners = [MinerClient(config) for config in configs]
    
    def __len__(self) -> int:
        return len(self.miners)
    
    def __getitem__(self, index: int) -> MinerClient:
        return self.miners[index]
    
    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all miners."""
        results = {}
        for miner in self.miners:
            results[miner.wallet_name] = await miner.health_check()
        return results
    
    async def fetch_games_all(self) -> dict[str, dict]:
        """Trigger all miners to fetch games."""
        results = {}
        for miner in self.miners:
            results[miner.wallet_name] = await miner.fetch_games()
        return results
    
    async def submit_varied_odds(
        self,
        markets: List[dict],
        ground_truth: dict[int, dict],
    ) -> dict[str, List[dict]]:
        """Each miner submits with a different strategy.
        
        - Miner 0: Early accurate
        - Miner 1: Late accurate  
        - Miner 2: Inaccurate
        
        Returns results keyed by wallet name.
        """
        results = {}
        
        if len(self.miners) >= 1:
            results[self.miners[0].wallet_name] = await self.miners[0].submit_early_accurate(
                markets, ground_truth
            )
        
        if len(self.miners) >= 2:
            results[self.miners[1].wallet_name] = await self.miners[1].submit_late_accurate(
                markets, ground_truth, delay_seconds=1.0
            )
        
        if len(self.miners) >= 3:
            results[self.miners[2].wallet_name] = await self.miners[2].submit_inaccurate(
                markets, ground_truth
            )
        
        return results
