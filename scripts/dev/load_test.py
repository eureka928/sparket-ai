#!/usr/bin/env python3
"""Load test for validator submission handling.

Tests the validator's ability to handle concurrent miner submissions
at expected peak load (hundreds of requests per minute).

Usage:
    python scripts/dev/load_test.py [--miners N] [--submissions N] [--duration N]

Requires:
    - E2E validator running on port 8199
    - E2E miners running on ports 8198, 8200, 8201
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp


@dataclass
class LoadTestConfig:
    """Load test configuration."""
    validator_url: str = "http://127.0.0.1:8199"
    miner_ports: List[int] = field(default_factory=lambda: [8198, 8197, 8196])
    num_virtual_miners: int = 10  # Simulated miners per real miner
    submissions_per_miner: int = 50
    concurrent_limit: int = 20  # Max concurrent requests
    ramp_up_seconds: float = 2.0  # Spread initial requests
    timeout_seconds: float = 30.0


@dataclass 
class RequestResult:
    """Result of a single request."""
    success: bool
    latency_ms: float
    status_code: Optional[int] = None
    error: Optional[str] = None
    rate_limited: bool = False


@dataclass
class LoadTestResults:
    """Aggregated load test results."""
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    rate_limited: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def throughput_rps(self) -> float:
        if self.duration_seconds > 0:
            return self.total_requests / self.duration_seconds
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests > 0:
            return self.successful / self.total_requests
        return 0.0
    
    def latency_percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * p / 100)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)
        print(f"Duration:        {self.duration_seconds:.2f}s")
        print(f"Total Requests:  {self.total_requests}")
        print(f"Successful:      {self.successful} ({self.success_rate * 100:.1f}%)")
        print(f"Failed:          {self.failed}")
        print(f"Rate Limited:    {self.rate_limited}")
        print(f"Throughput:      {self.throughput_rps:.1f} req/s")
        print()
        if self.latencies_ms:
            print("Latency (successful requests):")
            print(f"  Min:    {min(self.latencies_ms):.1f}ms")
            print(f"  p50:    {self.latency_percentile(50):.1f}ms")
            print(f"  p95:    {self.latency_percentile(95):.1f}ms")
            print(f"  p99:    {self.latency_percentile(99):.1f}ms")
            print(f"  Max:    {max(self.latencies_ms):.1f}ms")
            print(f"  Avg:    {statistics.mean(self.latencies_ms):.1f}ms")
        if self.errors:
            print(f"\nUnique Errors ({len(set(self.errors))}):")
            for err in list(set(self.errors))[:5]:
                print(f"  - {err[:80]}")
        print("=" * 60)


class LoadTester:
    """Executes load tests against the validator."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = LoadTestResults()
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    async def run(self) -> LoadTestResults:
        """Execute the load test."""
        print(f"Starting load test...")
        print(f"  Virtual miners: {self.config.num_virtual_miners * len(self.config.miner_ports)}")
        print(f"  Submissions per miner: {self.config.submissions_per_miner}")
        print(f"  Concurrent limit: {self.config.concurrent_limit}")
        
        # Check health first
        if not await self._check_health():
            print("ERROR: Validator or miners not healthy. Start E2E nodes first.")
            return self.results
        
        # Create test markets
        market_ids = await self._setup_markets()
        if not market_ids:
            print("ERROR: Failed to create test markets")
            return self.results
        
        print(f"  Test markets: {market_ids}")
        
        # Sync miners with validator
        await self._sync_miners()
        
        self._semaphore = asyncio.Semaphore(self.config.concurrent_limit)
        self.results.start_time = time.time()
        
        # Create tasks for all virtual miners
        tasks = []
        for port in self.config.miner_ports:
            for vm_id in range(self.config.num_virtual_miners):
                tasks.append(
                    self._run_virtual_miner(port, vm_id, market_ids)
                )
        
        # Run all miners concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.results.end_time = time.time()
        return self.results
    
    async def _check_health(self) -> bool:
        """Check that validator and miners are running."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.config.validator_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        return False
                    data = await resp.json()
                    if data.get("status") != "ok":
                        return False
            except Exception as e:
                print(f"Validator health check failed: {e}")
                return False
            
            # Check at least one miner
            for port in self.config.miner_ports[:1]:
                try:
                    async with session.get(
                        f"http://127.0.0.1:{port}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status != 200:
                            return False
                except Exception as e:
                    print(f"Miner health check failed (port {port}): {e}")
                    return False
        
        return True
    
    async def _setup_markets(self) -> List[int]:
        """Create test markets for load testing."""
        from datetime import datetime, timezone, timedelta
        
        market_ids = []
        async with aiohttp.ClientSession() as session:
            for i in range(5):  # Create 5 test markets
                try:
                    # Calculate future start time
                    start_time = datetime.now(timezone.utc) + timedelta(hours=24 + i)
                    
                    async with session.post(
                        f"{self.config.validator_url}/mock/event",
                        json={
                            "home_team": f"LoadTest Home {i}",
                            "away_team": f"LoadTest Away {i}",
                            "start_time": start_time.isoformat(),
                        },
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            market_id = data.get("event", {}).get("db_market_id")
                            if market_id:
                                market_ids.append(market_id)
                        else:
                            text = await resp.text()
                            print(f"Failed to create market {i}: {resp.status} - {text[:100]}")
                except Exception as e:
                    print(f"Failed to create market: {e}")
        return market_ids
    
    async def _sync_miners(self):
        """Sync miners with metagraph."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.config.validator_url}/sync-miners",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        print("  Miners synced with validator")
            except Exception:
                pass
    
    async def _run_virtual_miner(
        self, 
        miner_port: int, 
        vm_id: int,
        market_ids: List[int]
    ):
        """Simulate a single miner making submissions."""
        # Stagger start times
        await asyncio.sleep(random.uniform(0, self.config.ramp_up_seconds))
        
        async with aiohttp.ClientSession() as session:
            for i in range(self.config.submissions_per_miner):
                # Pick a random market
                market_id = random.choice(market_ids)
                
                # Generate random odds
                home_prob = random.uniform(0.3, 0.7)
                home_odds = round(1.0 / home_prob, 2)
                away_odds = round(1.0 / (1 - home_prob), 2)
                
                result = await self._submit_odds(
                    session, miner_port, market_id, home_odds, away_odds
                )
                
                self.results.total_requests += 1
                if result.success:
                    self.results.successful += 1
                    self.results.latencies_ms.append(result.latency_ms)
                else:
                    self.results.failed += 1
                    if result.error:
                        self.results.errors.append(result.error)
                
                if result.rate_limited:
                    self.results.rate_limited += 1
                
                # Small delay between submissions
                await asyncio.sleep(random.uniform(0.01, 0.05))
    
    async def _submit_odds(
        self,
        session: aiohttp.ClientSession,
        miner_port: int,
        market_id: int,
        home_odds: float,
        away_odds: float,
    ) -> RequestResult:
        """Submit odds and measure latency."""
        async with self._semaphore:
            start = time.time()
            try:
                async with session.post(
                    f"http://127.0.0.1:{miner_port}/action/submit-odds",
                    json={
                        "submissions": [{
                            "market_id": market_id,
                            "kind": "MONEYLINE",
                            "prices": [
                                {"side": "HOME", "odds_eu": home_odds},
                                {"side": "AWAY", "odds_eu": away_odds},
                            ]
                        }]
                    },
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as resp:
                    latency_ms = (time.time() - start) * 1000
                    
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "ok":
                            return RequestResult(
                                success=True,
                                latency_ms=latency_ms,
                                status_code=resp.status
                            )
                        else:
                            return RequestResult(
                                success=False,
                                latency_ms=latency_ms,
                                status_code=resp.status,
                                error=data.get("message", "Unknown error")
                            )
                    elif resp.status == 429:
                        return RequestResult(
                            success=False,
                            latency_ms=latency_ms,
                            status_code=resp.status,
                            rate_limited=True,
                            error="Rate limited"
                        )
                    else:
                        text = await resp.text()
                        return RequestResult(
                            success=False,
                            latency_ms=latency_ms,
                            status_code=resp.status,
                            error=f"HTTP {resp.status}: {text[:100]}"
                        )
            except asyncio.TimeoutError:
                return RequestResult(
                    success=False,
                    latency_ms=(time.time() - start) * 1000,
                    error="Timeout"
                )
            except Exception as e:
                return RequestResult(
                    success=False,
                    latency_ms=(time.time() - start) * 1000,
                    error=str(e)[:100]
                )


async def run_load_test(config: LoadTestConfig) -> LoadTestResults:
    """Run the load test with given configuration."""
    tester = LoadTester(config)
    return await tester.run()


def main():
    parser = argparse.ArgumentParser(description="Load test validator submission handling")
    parser.add_argument("--miners", type=int, default=10, help="Virtual miners per real miner")
    parser.add_argument("--submissions", type=int, default=50, help="Submissions per virtual miner")
    parser.add_argument("--concurrent", type=int, default=20, help="Max concurrent requests")
    parser.add_argument("--ramp-up", type=float, default=2.0, help="Ramp up period (seconds)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout (seconds)")
    args = parser.parse_args()
    
    config = LoadTestConfig(
        num_virtual_miners=args.miners,
        submissions_per_miner=args.submissions,
        concurrent_limit=args.concurrent,
        ramp_up_seconds=args.ramp_up,
        timeout_seconds=args.timeout,
    )
    
    results = asyncio.run(run_load_test(config))
    results.print_summary()
    
    # Exit with error if too many failures
    if results.success_rate < 0.95:
        print("\nWARNING: Success rate below 95%!")
        exit(1)


if __name__ == "__main__":
    main()
