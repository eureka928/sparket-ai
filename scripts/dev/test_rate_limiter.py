#!/usr/bin/env python3
"""Test the rate limiter by hammering a single miner.

Rate limits (from ratelimit.py):
- Per-hotkey: 5/sec, 60/min
- Global: 100/sec, 2000/min

This test verifies the rate limiter kicks in appropriately.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List

import aiohttp


@dataclass
class RateLimitTestResult:
    total_requests: int = 0
    successful: int = 0
    rate_limited: int = 0
    errors: int = 0
    duration_sec: float = 0.0


async def submit_odds_direct(
    session: aiohttp.ClientSession,
    miner_port: int,
    market_id: int,
) -> str:
    """Submit odds and return result status."""
    try:
        async with session.post(
            f"http://127.0.0.1:{miner_port}/action/submit-odds",
            json={
                "submissions": [{
                    "market_id": market_id,
                    "kind": "MONEYLINE",
                    "prices": [
                        {"side": "HOME", "odds_eu": 1.9},
                        {"side": "AWAY", "odds_eu": 2.0},
                    ]
                }]
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return "ok" if data.get("status") == "ok" else "error"
            elif resp.status == 429:
                return "rate_limited"
            else:
                return "error"
    except Exception:
        return "error"


async def test_per_second_limit():
    """Test per-hotkey per-second limit (5/sec)."""
    print("\n" + "=" * 60)
    print("TEST 1: Per-Second Rate Limit (5 req/sec per hotkey)")
    print("=" * 60)
    print("Sending 20 requests as fast as possible from one miner...")
    
    result = RateLimitTestResult()
    
    async with aiohttp.ClientSession() as session:
        start = time.time()
        
        # Send 20 requests rapidly
        tasks = []
        for _ in range(20):
            tasks.append(submit_odds_direct(session, 8198, 907))
        
        responses = await asyncio.gather(*tasks)
        result.duration_sec = time.time() - start
        
        for resp in responses:
            result.total_requests += 1
            if resp == "ok":
                result.successful += 1
            elif resp == "rate_limited":
                result.rate_limited += 1
            else:
                result.errors += 1
    
    print(f"Duration: {result.duration_sec:.2f}s")
    print(f"Total: {result.total_requests}")
    print(f"Successful: {result.successful}")
    print(f"Rate Limited: {result.rate_limited}")
    print(f"Errors: {result.errors}")
    
    # With 5/sec limit and ~20 requests in <1 sec, we expect ~15 to be rate limited
    if result.rate_limited > 0:
        print("✓ Rate limiter is working!")
    else:
        print("⚠ No rate limiting detected - check if limiter is enabled")
    
    return result


async def test_per_minute_limit():
    """Test per-hotkey per-minute limit (60/min)."""
    print("\n" + "=" * 60)
    print("TEST 2: Per-Minute Rate Limit (60 req/min per hotkey)")
    print("=" * 60)
    print("Sending 80 requests over ~15 seconds from one miner...")
    
    result = RateLimitTestResult()
    
    async with aiohttp.ClientSession() as session:
        start = time.time()
        
        # Send 80 requests spread over 15 seconds (to not hit per-second limit)
        for i in range(80):
            resp = await submit_odds_direct(session, 8198, 907)
            result.total_requests += 1
            if resp == "ok":
                result.successful += 1
            elif resp == "rate_limited":
                result.rate_limited += 1
            else:
                result.errors += 1
            
            # Small delay to avoid per-second limit
            await asyncio.sleep(0.15)
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Sent {i + 1}/80 requests...")
        
        result.duration_sec = time.time() - start
    
    print(f"Duration: {result.duration_sec:.2f}s")
    print(f"Total: {result.total_requests}")
    print(f"Successful: {result.successful}")
    print(f"Rate Limited: {result.rate_limited}")
    print(f"Errors: {result.errors}")
    
    # With 60/min limit and 80 requests, we expect ~20 to be rate limited
    if result.rate_limited > 0:
        print("✓ Per-minute rate limiter is working!")
    else:
        print("⚠ No per-minute rate limiting detected")
    
    return result


async def test_different_miners_separate_limits():
    """Test that different miners have separate limits."""
    print("\n" + "=" * 60)
    print("TEST 3: Separate Limits Per Miner")
    print("=" * 60)
    print("Sending 10 rapid requests from each of 3 miners...")
    
    results_by_miner = {}
    
    async with aiohttp.ClientSession() as session:
        for port in [8198, 8197, 8196]:
            result = RateLimitTestResult()
            
            tasks = []
            for _ in range(10):
                tasks.append(submit_odds_direct(session, port, 907))
            
            responses = await asyncio.gather(*tasks)
            
            for resp in responses:
                result.total_requests += 1
                if resp == "ok":
                    result.successful += 1
                elif resp == "rate_limited":
                    result.rate_limited += 1
                else:
                    result.errors += 1
            
            results_by_miner[port] = result
            print(f"  Miner {port}: {result.successful} ok, {result.rate_limited} limited, {result.errors} errors")
    
    # Each miner should have ~5 successful (per-second limit)
    all_have_some_success = all(r.successful > 0 for r in results_by_miner.values())
    if all_have_some_success:
        print("✓ Each miner has separate rate limits!")
    else:
        print("⚠ Some miners had no successful requests")
    
    return results_by_miner


async def check_rate_limiter_stats():
    """Check the rate limiter stats endpoint."""
    print("\n" + "=" * 60)
    print("RATE LIMITER STATS")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                "http://127.0.0.1:8199/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    print("Validator is healthy")
        except Exception as e:
            print(f"Could not reach validator: {e}")
            return


async def main():
    print("=" * 60)
    print("RATE LIMITER TEST SUITE")
    print("=" * 60)
    print("\nRate Limits:")
    print("  Per-hotkey: 5/sec, 60/min")
    print("  Global: 100/sec, 2000/min")
    
    # Check health first
    await check_rate_limiter_stats()
    
    # Wait for any previous rate limits to expire
    print("\nWaiting 5 seconds for rate limits to reset...")
    await asyncio.sleep(5)
    
    # Run tests
    await test_per_second_limit()
    
    print("\nWaiting 5 seconds between tests...")
    await asyncio.sleep(5)
    
    await test_different_miners_separate_limits()
    
    # Per-minute test takes longer and may not show rate limiting
    # if the dendrite is slow enough to naturally space requests
    print("\nWaiting 60 seconds for per-minute limit to reset...")
    await asyncio.sleep(60)
    
    await test_per_minute_limit()
    
    print("\n" + "=" * 60)
    print("RATE LIMITER TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
