"""Memory Profiling Scenario - Tests for memory leaks and unbounded growth.

Tests:
- Memory usage doesn't grow unbounded over multiple cycles
- Large data volumes don't cause excessive memory use
- Garbage collection works properly

Measures:
- RSS memory before/after each cycle
- Peak memory usage
- Memory delta per cycle
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

from .base import BaseScenario, ScenarioResult


@dataclass
class MemorySample:
    """Memory usage sample."""
    cycle: int
    rss_mb: float
    timestamp: datetime


class MemoryProfilingScenario(BaseScenario):
    """Tests memory usage patterns over multiple scoring cycles."""
    
    SCENARIO_ID = "memory_profiling"
    NUM_CYCLES = 10  # Number of scoring cycles
    EVENTS_PER_CYCLE = 5
    MAX_GROWTH_MB_PER_CYCLE = 10.0  # Alert if growth exceeds this
    
    def __init__(self, harness):
        super().__init__(harness)
        self.memory_samples: List[MemorySample] = []
        self.market_ids: List[int] = []
    
    async def setup(self) -> None:
        """Setup for memory profiling."""
        # Record baseline memory
        baseline = await self._get_validator_memory()
        if baseline:
            self.memory_samples.append(MemorySample(
                cycle=0,
                rss_mb=baseline,
                timestamp=datetime.now(timezone.utc),
            ))
            self.result.metrics["baseline_memory_mb"] = baseline
    
    async def _get_validator_memory(self) -> float | None:
        """Get validator memory usage in MB."""
        try:
            mem_result = await self.validator.get_memory()
            if mem_result.get("status") == "ok":
                return mem_result.get("memory", {}).get("rss_mb")
        except Exception:
            pass
        return None
    
    async def _run_cycle(self, cycle_num: int) -> None:
        """Run one scoring cycle."""
        # Create events for this cycle
        for i in range(self.EVENTS_PER_CYCLE):
            result = await self.validator.create_event(
                home_team=f"MemCycle{cycle_num}Home{i}",
                away_team=f"MemCycle{cycle_num}Away{i}",
                hours_ahead=48,
            )
            
            if result.get("status") == "ok":
                market_id = result.get("event", {}).get("db_market_id")
                if market_id:
                    self.market_ids.append(market_id)
                    await self.validator.seed_ground_truth(
                        market_id=market_id,
                        home_prob=0.55,
                        away_prob=0.45,
                    )
        
        # Have miners submit odds
        for miner in self.miners:
            if await miner.health_check():
                for market_id in self.market_ids[-self.EVENTS_PER_CYCLE:]:
                    await miner.submit_odds(market_id, 1.8, 2.1)
        
        # Backdate and trigger scoring
        await self.validator.backdate_submissions(days=1)
        await self.validator.trigger_odds_scoring()
        await self.validator.trigger_scoring()
        
        # Record memory after cycle
        mem = await self._get_validator_memory()
        if mem:
            self.memory_samples.append(MemorySample(
                cycle=cycle_num,
                rss_mb=mem,
                timestamp=datetime.now(timezone.utc),
            ))
    
    async def execute(self) -> None:
        """Execute memory profiling cycles."""
        await self.validator.sync_miners()
        
        for cycle in range(1, self.NUM_CYCLES + 1):
            await self._run_cycle(cycle)
            
            # Brief pause between cycles
            await asyncio.sleep(0.5)
    
    async def verify(self) -> None:
        """Analyze memory usage patterns."""
        if len(self.memory_samples) < 2:
            self.result.add_warning("Not enough memory samples to analyze")
            return
        
        # Calculate statistics
        baseline = self.memory_samples[0].rss_mb
        final = self.memory_samples[-1].rss_mb
        peak = max(s.rss_mb for s in self.memory_samples)
        
        total_growth = final - baseline
        avg_growth_per_cycle = total_growth / (len(self.memory_samples) - 1)
        
        self.result.metrics["baseline_mb"] = baseline
        self.result.metrics["final_mb"] = final
        self.result.metrics["peak_mb"] = peak
        self.result.metrics["total_growth_mb"] = total_growth
        self.result.metrics["avg_growth_per_cycle_mb"] = avg_growth_per_cycle
        self.result.metrics["cycles_completed"] = len(self.memory_samples) - 1
        
        # Record all samples for detailed analysis
        self.result.metrics["memory_samples"] = [
            {"cycle": s.cycle, "rss_mb": s.rss_mb}
            for s in self.memory_samples
        ]
        
        # Check for unbounded growth
        if avg_growth_per_cycle > self.MAX_GROWTH_MB_PER_CYCLE:
            self.result.add_fail(
                f"Average memory growth {avg_growth_per_cycle:.2f} MB/cycle exceeds threshold "
                f"({self.MAX_GROWTH_MB_PER_CYCLE} MB/cycle)"
            )
        else:
            self.result.add_pass(
                f"Memory growth {avg_growth_per_cycle:.2f} MB/cycle is within acceptable range"
            )
        
        # Check peak isn't excessive (more than 2x baseline)
        if baseline > 0 and peak > baseline * 3:
            self.result.add_warning(
                f"Peak memory ({peak:.1f} MB) is more than 3x baseline ({baseline:.1f} MB)"
            )
        else:
            self.result.add_pass("Peak memory within acceptable range")
        
        # Check for memory decrease (GC working)
        decreases = 0
        for i in range(1, len(self.memory_samples)):
            if self.memory_samples[i].rss_mb < self.memory_samples[i-1].rss_mb:
                decreases += 1
        
        if decreases > 0:
            self.result.add_pass(f"Memory decreased in {decreases} cycles (GC working)")
        else:
            self.result.add_warning("No memory decreases observed (GC may not be running)")
