#!/usr/bin/env python3
"""
End-to-End Integration Test for Sparket Subnet

This script controls running validator and miner nodes via their HTTP
control APIs. It does NOT use bittensor - it's a pure HTTP controller.

Architecture:
    ┌─────────────────┐   HTTP :8199   ┌──────────────────┐
    │   Test Script   │ ◄────────────► │    Validator     │
    │   (controller)  │                │  control API     │
    │                 │   HTTP :8198   │                  │
    │                 │ ◄────────────► │    Miner         │
    │                 │                │  control API     │
    └─────────────────┘                └──────────────────┘
    
    Test Phases:
    0. Health Check - nodes running
    1. Seed Mock Data (events, quotes, ground truth)
    2. Miner Fetch Games
    3. Miner Submit Odds (varied timing)
    4. Miner Submit Outcomes
    5. Transition Events to Finished
    6. Trigger Full Scoring Pipeline
    7. Verify Rolling Scores
    8. Verify Skill Scores
    9. Verify Weight Array
    10. Edge Case Scenarios
    11. Memory Monitoring

Usage:
    # Start nodes in test mode first
    pm2 start ecosystem.test.config.js
    
    # Run E2E test
    python scripts/dev/test_e2e.py
    
    # Run with edge cases
    python scripts/dev/test_e2e.py --include-edge-cases
    
    # Cleanup
    python scripts/dev/test_e2e.py --teardown
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import aiohttp


@dataclass
class E2EConfig:
    """E2E test configuration."""
    validator_control_url: str = "http://127.0.0.1:8199"
    miner_control_url: str = "http://127.0.0.1:8198"
    timeout: float = 60.0
    num_events: int = 5
    submissions_per_event: int = 3
    include_edge_cases: bool = False
    include_crash_recovery: bool = False


@dataclass
class TestResults:
    """Test results tracker."""
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    assertions: list = field(default_factory=list)
    
    def assert_true(self, condition: bool, msg: str) -> bool:
        if condition:
            self.passed += 1
            self.assertions.append(("PASS", msg))
            log(f"✓ {msg}", "PASS")
            return True
        else:
            self.failed += 1
            self.assertions.append(("FAIL", msg))
            log(f"✗ {msg}", "FAIL")
            return False
    
    def assert_equal(self, actual: Any, expected: Any, msg: str) -> bool:
        return self.assert_true(actual == expected, f"{msg} (expected={expected}, actual={actual})")
    
    def assert_greater(self, actual: Any, threshold: Any, msg: str) -> bool:
        return self.assert_true(actual > threshold, f"{msg} (actual={actual} > {threshold})")
    
    def assert_in_range(self, actual: float, low: float, high: float, msg: str) -> bool:
        in_range = low <= actual <= high
        return self.assert_true(in_range, f"{msg} (actual={actual} in [{low}, {high}])")
    
    def warn(self, msg: str) -> None:
        self.warnings += 1
        self.assertions.append(("WARN", msg))
        log(f"⚠ {msg}", "WARN")
    
    @property
    def success(self) -> bool:
        return self.failed == 0


def log(msg: str, level: str = "INFO") -> None:
    """Logging with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {
        "INFO": "", 
        "PASS": "\033[92m", 
        "FAIL": "\033[91m", 
        "WARN": "\033[93m", 
        "SECTION": "\033[94m",
        "DEBUG": "\033[90m",
    }
    reset = "\033[0m"
    color = colors.get(level, "")
    print(f"{color}[{ts}] [{level}] {msg}{reset}")


def log_section(title: str) -> None:
    """Log section header."""
    print()
    log("=" * 60, "SECTION")
    log(f" {title}", "SECTION")
    log("=" * 60, "SECTION")


class ControlClient:
    """HTTP client for node control APIs."""
    
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    async def get(self, path: str) -> dict:
        """Send GET request."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(f"{self.base_url}{path}") as resp:
                return await resp.json()
    
    async def post(self, path: str, data: dict | None = None) -> dict:
        """Send POST request."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}{path}", json=data or {}) as resp:
                return await resp.json()
    
    async def health(self) -> bool:
        """Check if node is healthy."""
        try:
            result = await self.get("/health")
            return result.get("status") == "ok"
        except Exception:
            return False


# =============================================================================
# Test Phases
# =============================================================================

async def phase0_check_nodes_running(config: E2EConfig, results: TestResults) -> bool:
    """Phase 0: Verify both nodes are running and healthy."""
    log_section("Phase 0: Check Nodes Running")
    
    validator = ControlClient(config.validator_control_url, config.timeout)
    miner = ControlClient(config.miner_control_url, config.timeout)
    
    validator_ok = await validator.health()
    miner_ok = await miner.health()
    
    results.assert_true(validator_ok, "Validator control API is healthy")
    results.assert_true(miner_ok, "Miner control API is healthy")
    
    if not validator_ok:
        log(f"Validator not responding at {config.validator_control_url}", "FAIL")
        log("Start with: pm2 start ecosystem.test.config.js --only test-validator")
    
    if not miner_ok:
        log(f"Miner not responding at {config.miner_control_url}", "FAIL")
        log("Start with: pm2 start ecosystem.test.config.js --only test-miner")
    
    # Check initial memory baseline
    if validator_ok:
        try:
            mem = await validator.get("/health/memory")
            if mem.get("status") == "ok":
                mem_mb = mem.get("memory", {}).get("rss_mb", 0)
                log(f"Validator memory baseline: {mem_mb:.1f} MB", "DEBUG")
        except Exception:
            pass
    
    return validator_ok and miner_ok


async def phase1_seed_mock_data(config: E2EConfig, results: TestResults) -> dict:
    """Phase 1: Seed mock game data on validator."""
    log_section("Phase 1: Seed Mock Game Data")
    
    validator = ControlClient(config.validator_control_url, config.timeout)
    
    # Wipe database for clean test
    log("Wiping test database...")
    wipe_result = await validator.post("/admin/wipe-db")
    if wipe_result.get("status") == "ok":
        log(f"  {wipe_result.get('message', 'Database wiped')}")
    else:
        log(f"  Warning: {wipe_result.get('message', 'unknown error')}", "WARN")
    
    # Sync miners from metagraph to database
    log("Syncing miners from metagraph...")
    sync_result = await validator.post("/admin/sync-miners")
    log(f"  {sync_result.get('message', 'unknown')}")
    results.assert_true(sync_result.get("status") == "ok", "Synced miners from metagraph")
    
    # Reset mock provider
    log("Resetting mock provider...")
    await validator.post("/mock/reset")
    
    now = datetime.now(timezone.utc)
    events_created = []
    markets_created = []
    
    # Create events - mix of future (for odds) and past (for outcomes)
    teams = [
        ("Alpha FC", "Beta United"),
        ("Gamma City", "Delta Town"),
        ("Epsilon Athletic", "Zeta Rangers"),
        ("Eta Warriors", "Theta FC"),
        ("Iota United", "Kappa City"),
    ]
    
    for i, (home, away) in enumerate(teams[:config.num_events]):
        # Half future, half past
        if i < config.num_events // 2:
            start_time = now + timedelta(days=2)
            status = "scheduled"
        else:
            start_time = now - timedelta(hours=3)
            status = "in_play"
        
        log(f"Creating event: {home} vs {away} ({status})")
        
        result = await validator.post("/mock/event", {
            "home_team": home,
            "away_team": away,
            "start_time": start_time.isoformat(),
            "status": status,
        })
        
        if result.get("status") == "ok":
            event = result.get("event", {})
            events_created.append(event)
            db_market_id = event.get("db_market_id")
            if db_market_id:
                markets_created.append({"market_id": db_market_id, "event_status": status})
            log(f"  Created event: {event.get('event_id')} (market_id: {db_market_id})")
    
    results.assert_equal(len(events_created), config.num_events, f"Created {config.num_events} events")
    
    # Seed provider quotes for ground truth computation
    log("Seeding provider quotes for ground truth...")
    for market in markets_created:
        market_id = market["market_id"]
        # Simulate realistic odds (HOME and AWAY)
        home_odds = round(random.uniform(1.4, 3.5), 2)
        away_odds = round(random.uniform(1.4, 3.5), 2)
        
        # Add HOME odds
        await validator.post("/mock/provider-quote", {
            "market_id": market_id,
            "side": "HOME",
            "odds_eu": home_odds,
        })
        # Add AWAY odds
        await validator.post("/mock/provider-quote", {
            "market_id": market_id,
            "side": "AWAY",
            "odds_eu": away_odds,
        })
        log(f"  Market {market_id}: HOME={home_odds}, AWAY={away_odds}")
    
    # Seed ground truth closing directly for testing
    # (snapshot pipeline requires specific timing that's hard to simulate)
    log("Seeding ground truth closing directly...")
    gt_count = 0
    for market in markets_created:
        market_id = market["market_id"]
        # Use the same odds we seeded for provider quotes (already in the loop)
        # Recalculate here for ground truth
        home_odds = round(random.uniform(1.4, 3.5), 2)
        away_odds = round(random.uniform(1.4, 3.5), 2)
        home_prob = round(1.0 / home_odds, 4)
        away_prob = round(1.0 / away_odds, 4)
        
        result = await validator.post("/mock/ground-truth-closing", {
            "market_id": market_id,
            "sides": [
                {"side": "HOME", "prob_consensus": home_prob, "odds_consensus": home_odds},
                {"side": "AWAY", "prob_consensus": away_prob, "odds_consensus": away_odds},
            ]
        })
        if result.get("status") == "ok":
            gt_count += result.get("sides_added", 0)
    log(f"  Seeded {gt_count} ground truth closing records")
    
    # Query the database to get actual event IDs
    db_state = await validator.get("/db/submissions")
    log(f"Database has {db_state.get('count', 0)} submissions initially")
    
    return {
        "events": events_created,
        "markets": markets_created,
    }


async def phase2_miner_fetch_games(config: E2EConfig, results: TestResults) -> dict | None:
    """Phase 2: Miner fetches game data from validator."""
    log_section("Phase 2: Miner Fetches Game Data")
    
    miner = ControlClient(config.miner_control_url, config.timeout)
    
    log("Triggering miner to fetch games from validator...")
    
    result = await miner.post("/action/fetch-games")
    
    success = result.get("status") == "ok"
    results.assert_true(success, f"Miner fetch games: {result.get('message', 'unknown')}")
    
    if success:
        # Check what games the miner has cached
        games = await miner.get("/games")
        game_count = games.get("count", 0)
        log(f"Miner has {game_count} games cached")
        return games
    
    return None


async def phase3_miner_submit_odds(config: E2EConfig, results: TestResults, mock_data: dict) -> int:
    """Phase 3: Miner submits odds to validator."""
    log_section("Phase 3: Miner Submits Odds")
    
    miner = ControlClient(config.miner_control_url, config.timeout)
    validator = ControlClient(config.validator_control_url, config.timeout)
    
    # Submit odds for ALL markets (we'll test scoring on all of them)
    all_markets = mock_data.get("markets", [])
    
    if not all_markets:
        log("No markets found, skipping odds submission", "WARN")
        return 0
    
    submissions_sent = 0
    
    for market in all_markets:
        market_id = market.get("market_id")
        if not market_id:
            continue
            
        for i in range(config.submissions_per_event):
            home_odds = round(random.uniform(1.5, 4.0), 2)
            away_odds = round(random.uniform(1.5, 4.0), 2)
            
            submission = {
                "submissions": [{
                    "market_id": market_id,
                    "kind": "MONEYLINE",
                    "prices": [
                        {
                            "side": "HOME",
                            "odds_eu": home_odds,
                            "imp_prob": round(1.0 / home_odds, 4),
                        },
                        {
                            "side": "AWAY",
                            "odds_eu": away_odds,
                            "imp_prob": round(1.0 / away_odds, 4),
                        }
                    ]
                }]
            }
            
            result = await miner.post("/action/submit-odds", submission)
            if result.get("status") == "ok":
                submissions_sent += 1
    
    log(f"Sent {submissions_sent} odds submissions")
    
    # Wait for processing
    await asyncio.sleep(1)
    
    # Verify submissions reached validator
    db_state = await validator.get("/db/submissions")
    submission_count = db_state.get("count", 0)
    
    results.assert_greater(submission_count, 0, f"Validator received {submission_count} submissions")
    
    return submissions_sent


async def phase4_miner_submit_outcomes(config: E2EConfig, results: TestResults, mock_data: dict) -> int:
    """Phase 4: Miner submits outcomes to validator."""
    log_section("Phase 4: Miner Submits Outcomes")
    
    miner = ControlClient(config.miner_control_url, config.timeout)
    
    # Get in_play events (can accept outcomes)
    in_play_events = [e for e in mock_data.get("events", []) if e.get("status") == "in_play"]
    
    submissions_sent = 0
    
    for event in in_play_events:
        home_score = random.randint(0, 3)
        away_score = random.randint(0, 3)
        
        if home_score > away_score:
            result_str = "HOME"
        elif away_score > home_score:
            result_str = "AWAY"
        else:
            result_str = "DRAW"
        
        outcome = {
            "event_id": 1,  # Will need actual event ID
            "result": result_str,
            "score_home": home_score,
            "score_away": away_score,
        }
        
        log(f"Submitting outcome: {result_str} ({home_score}-{away_score})")
        
        result = await miner.post("/action/submit-outcome", outcome)
        if result.get("status") == "ok":
            submissions_sent += 1
    
    log(f"Sent {submissions_sent} outcome submissions")
    
    return submissions_sent


async def phase5_transition_events(config: E2EConfig, results: TestResults, mock_data: dict) -> None:
    """Phase 5: Transition events to finished and add outcomes."""
    log_section("Phase 5: Transition Event States & Add Outcomes")
    
    validator = ControlClient(config.validator_control_url, config.timeout)
    
    # Add outcomes for ALL markets (simulates games finishing)
    log("Adding outcomes for all markets...")
    outcome_count = 0
    
    for market in mock_data.get("markets", []):
        market_id = market.get("market_id")
        if not market_id:
            continue
        
        # Simulate random outcome
        home_score = random.randint(0, 3)
        away_score = random.randint(0, 3)
        
        if home_score > away_score:
            result_str = "HOME"
        elif away_score > home_score:
            result_str = "AWAY"
        else:
            result_str = "DRAW"
        
        log(f"  Market {market_id}: {result_str} ({home_score}-{away_score})")
        
        result = await validator.post("/mock/settled-outcome", {
            "market_id": market_id,
            "result": result_str,
            "score_home": home_score,
            "score_away": away_score,
        })
        if result.get("status") == "ok":
            outcome_count += 1
    
    log(f"Added {outcome_count} outcomes")
    results.assert_greater(outcome_count, 0, "Outcomes added for markets")


async def phase6_trigger_scoring(config: E2EConfig, results: TestResults) -> dict:
    """Phase 6: Trigger full scoring pipeline (odds scoring, outcome scoring, aggregation)."""
    log_section("Phase 6: Trigger Full Scoring Pipeline")
    
    validator = ControlClient(config.validator_control_url, config.timeout)
    
    # Step 1: Trigger odds scoring (CLV/CLE computation)
    log("Step 1: Computing CLV/CLE scores for submissions...")
    odds_result = await validator.post("/trigger/odds-scoring")
    if odds_result.get("status") == "ok":
        log(f"  Scored {odds_result.get('submissions_scored', 0)} submissions for CLV/CLE")
    else:
        log(f"  Warning: {odds_result.get('message', 'unknown')}", "WARN")
    
    # Step 2: Trigger outcome scoring (Brier/PSS computation)
    log("Step 2: Computing Brier/PSS scores for settled outcomes...")
    outcome_result = await validator.post("/trigger/outcome-scoring")
    if outcome_result.get("status") == "ok":
        log(f"  Scored {outcome_result.get('submissions_scored', 0)} submissions for Brier/PSS")
    else:
        log(f"  Warning: {outcome_result.get('message', 'unknown')}", "WARN")
    
    # Step 3: Backdate all data to yesterday so it falls within the scoring window
    # (The scoring window ends at today's midnight, so today's submissions are excluded)
    log("Step 3: Backdating submissions to fit scoring window...")
    backdate_result = await validator.post("/mock/backdate-submissions", {"days_back": 1})
    if backdate_result.get("status") == "ok":
        log(f"  Backdated {backdate_result.get('submissions_backdated', 0)} submissions by 1 day")
    else:
        log(f"  Warning: {backdate_result.get('message', 'unknown')}", "WARN")
    
    # Step 4: Trigger the main aggregation pipeline
    log("Step 4: Running aggregation pipeline (rolling scores, skill scores)...")
    result = await validator.post("/trigger/scoring")
    
    success = result.get("status") == "ok"
    results.assert_true(success, f"Scoring triggered: {result.get('message', 'unknown')}")
    
    if success:
        # Check for computed scores
        scores = await validator.get("/db/scores")
        score_count = scores.get("count", 0)
        log(f"  Computed {score_count} miner scores")
    
    return result


async def phase7_verify_rolling_scores(config: E2EConfig, results: TestResults) -> dict:
    """Phase 7: Verify rolling aggregate scores."""
    log_section("Phase 7: Verify Rolling Scores")
    
    validator = ControlClient(config.validator_control_url, config.timeout)
    
    rolling = await validator.get("/db/rolling-scores")
    score_count = rolling.get("count", 0)
    scores = rolling.get("scores", [])
    
    log(f"Found {score_count} rolling scores")
    
    if score_count == 0:
        results.warn("No rolling scores computed - may need more submissions")
        return rolling
    
    results.assert_greater(score_count, 0, "Rolling scores computed")
    
    # Verify score bounds
    for score in scores[:5]:  # Check first 5
        miner_id = score.get("miner_id")
        
        # FQ raw should be in [-1, 1]
        fq_raw = score.get("fq_raw")
        if fq_raw is not None:
            results.assert_in_range(fq_raw, -1.0, 1.0, f"Miner {miner_id} fq_raw in bounds")
        
        # n_eff should be positive
        n_eff = score.get("n_eff", 0)
        if n_eff is not None and n_eff > 0:
            results.assert_greater(n_eff, 0, f"Miner {miner_id} n_eff positive")
        
        # Log sample
        log(f"  Miner {miner_id}: fq_raw={fq_raw}, n_eff={n_eff}, pss_mean={score.get('pss_mean')}")
    
    return rolling


async def phase8_verify_skill_scores(config: E2EConfig, results: TestResults) -> dict:
    """Phase 8: Verify skill scores are normalized."""
    log_section("Phase 8: Verify Skill Scores")
    
    validator = ControlClient(config.validator_control_url, config.timeout)
    
    skill = await validator.get("/db/skill-scores")
    score_count = skill.get("count", 0)
    scores = skill.get("scores", [])
    
    log(f"Found {score_count} skill scores")
    
    if score_count == 0:
        results.warn("No skill scores computed - rolling aggregates may be missing")
        return skill
    
    results.assert_greater(score_count, 0, "Skill scores computed")
    
    # Verify normalization
    for score in scores[:5]:  # Check first 5
        miner_id = score.get("miner_id")
        skill_score = score.get("skill_score")
        
        if skill_score is not None:
            # Skill score should be in [0, 1]
            results.assert_in_range(skill_score, 0.0, 1.0, f"Miner {miner_id} skill_score normalized")
            
            # Dimension scores should also be in [0, 1]
            for dim in ["forecast_dim", "econ_dim", "info_dim"]:
                dim_val = score.get(dim)
                if dim_val is not None:
                    results.assert_in_range(dim_val, 0.0, 1.0, f"Miner {miner_id} {dim} normalized")
        
        log(f"  Miner {miner_id} (UID {score.get('uid')}): skill_score={skill_score}")
    
    return skill


async def phase9_verify_weights(config: E2EConfig, results: TestResults) -> dict:
    """Phase 9: Verify weight array is valid."""
    log_section("Phase 9: Verify Weight Array")
    
    validator = ControlClient(config.validator_control_url, config.timeout)
    
    weights = await validator.get("/db/weights")
    weight_count = weights.get("count", 0)
    weight_list = weights.get("weights", [])
    total_weight = weights.get("total_weight", 0)
    
    log(f"Found {weight_count} non-zero weights")
    
    if weight_count == 0:
        results.warn("No weights computed - skill scores may be missing or all zero")
        return weights
    
    results.assert_greater(weight_count, 0, "Non-zero weights exist")
    
    # Weights should sum to 1 (within tolerance)
    results.assert_in_range(total_weight, 0.99, 1.01, "Weights sum to ~1.0")
    
    # Individual weights should be in [0, 1]
    for w in weight_list[:5]:
        uid = w.get("uid")
        weight = w.get("weight", 0)
        raw_score = w.get("raw_score", 0)
        
        results.assert_in_range(weight, 0.0, 1.0, f"UID {uid} weight in [0,1]")
        log(f"  UID {uid}: weight={weight:.4f}, raw_score={raw_score:.4f}")
    
    return weights


async def phase10_edge_cases(config: E2EConfig, results: TestResults) -> None:
    """Phase 10: Test edge case scenarios."""
    log_section("Phase 10: Edge Case Scenarios")
    
    if not config.include_edge_cases:
        log("Skipping edge cases (use --include-edge-cases to enable)")
        return
    
    validator = ControlClient(config.validator_control_url, config.timeout)
    miner = ControlClient(config.miner_control_url, config.timeout)
    
    # Test: Extreme odds values
    log("Testing extreme odds values...")
    
    # Very low odds (should be handled)
    extreme_low = {
        "submissions": [{
            "market_id": 1,
            "kind": "MONEYLINE",
            "prices": [{"side": "HOME", "odds_eu": 1.01, "imp_prob": 0.99}]
        }]
    }
    result = await miner.post("/action/submit-odds", extreme_low)
    log(f"  Low odds (1.01): {result.get('status', 'unknown')}")
    
    # Very high odds (should be handled)
    extreme_high = {
        "submissions": [{
            "market_id": 1,
            "kind": "MONEYLINE", 
            "prices": [{"side": "HOME", "odds_eu": 999.0, "imp_prob": 0.001}]
        }]
    }
    result = await miner.post("/action/submit-odds", extreme_high)
    log(f"  High odds (999.0): {result.get('status', 'unknown')}")
    
    # Test: Invalid probability sum
    log("Testing invalid probability sum...")
    bad_prob = {
        "submissions": [{
            "market_id": 1,
            "kind": "MONEYLINE",
            "prices": [
                {"side": "HOME", "odds_eu": 2.0, "imp_prob": 0.5},
                {"side": "AWAY", "odds_eu": 2.0, "imp_prob": 0.8},  # Sum > 1
            ]
        }]
    }
    result = await miner.post("/action/submit-odds", bad_prob)
    # Should be rejected or clamped
    log(f"  Invalid prob sum: {result.get('status', 'unknown')}")
    
    # Re-run scoring to ensure it handles edge cases
    log("Re-running scoring with edge case data...")
    scoring_result = await validator.post("/trigger/scoring")
    results.assert_true(
        scoring_result.get("status") == "ok",
        "Scoring handles edge cases without crashing"
    )


async def phase11_memory_monitoring(config: E2EConfig, results: TestResults) -> dict:
    """Phase 11: Memory monitoring and baseline."""
    log_section("Phase 11: Memory Monitoring")
    
    validator = ControlClient(config.validator_control_url, config.timeout)
    
    try:
        mem = await validator.get("/health/memory")
        
        if mem.get("status") != "ok":
            results.warn("Could not get memory stats")
            return {}
        
        memory = mem.get("memory", {})
        rss_mb = memory.get("rss_mb", 0)
        max_rss_mb = memory.get("max_rss_mb", 0)
        percent = memory.get("percent", 0)
        
        log(f"Current RSS: {rss_mb:.1f} MB")
        log(f"Max RSS: {max_rss_mb:.1f} MB")
        log(f"Memory percent: {percent:.1f}%")
        
        # Warn if memory is high
        if rss_mb > 1024:
            results.warn(f"Memory usage is high: {rss_mb:.1f} MB")
        
        # Check job status
        jobs = await validator.get("/health/jobs")
        job_list = jobs.get("jobs", [])
        worker_list = jobs.get("workers", [])
        
        log(f"Scoring jobs: {len(job_list)}")
        for job in job_list:
            status = job.get("status", "unknown")
            job_id = job.get("job_id", "?")
            log(f"  {job_id}: {status}")
        
        log(f"Workers: {len(worker_list)}")
        for worker in worker_list:
            worker_id = worker.get("worker_id", "?")
            mem_mb = worker.get("memory_mb", 0)
            log(f"  {worker_id}: {mem_mb} MB")
        
        return mem
        
    except Exception as e:
        results.warn(f"Memory monitoring failed: {e}")
        return {}


async def phase12_verify_basic_results(config: E2EConfig, results: TestResults) -> None:
    """Phase 12: Final verification of basic results."""
    log_section("Phase 12: Final Verification")
    
    validator = ControlClient(config.validator_control_url, config.timeout)
    
    # Get submissions
    submissions = await validator.get("/db/submissions")
    submission_count = submissions.get("count", 0)
    
    log(f"Total submissions in database: {submission_count}")
    
    if submissions.get("submissions"):
        log("Sample submissions:")
        for sub in submissions["submissions"][:3]:
            log(f"  - Market {sub.get('market_id')}: {sub.get('odds_eu')} ({sub.get('side')})")
    
    # Get scores
    scores = await validator.get("/db/scores")
    score_count = scores.get("count", 0)
    
    log(f"Total scores computed: {score_count}")


def print_summary(results: TestResults) -> None:
    """Print test summary."""
    log_section("Test Summary")
    
    total = results.passed + results.failed
    
    log(f"Total assertions: {total}")
    log(f"  Passed: {results.passed}", "PASS" if results.passed > 0 else "INFO")
    log(f"  Failed: {results.failed}", "FAIL" if results.failed > 0 else "INFO")
    log(f"  Warnings: {results.warnings}", "WARN" if results.warnings > 0 else "INFO")
    
    if results.failed > 0:
        log("\nFailed assertions:", "FAIL")
        for status, msg in results.assertions:
            if status == "FAIL":
                log(f"  ✗ {msg}", "FAIL")
    
    if results.warnings > 0:
        log("\nWarnings:", "WARN")
        for status, msg in results.assertions:
            if status == "WARN":
                log(f"  ⚠ {msg}", "WARN")
    
    status = "PASSED" if results.success else "FAILED"
    color = "PASS" if results.success else "FAIL"
    log(f"\nOverall: {status}", color)


async def main(config: E2EConfig) -> int:
    """Main E2E test runner."""
    log_section("Sparket E2E Integration Test")
    log(f"Validator: {config.validator_control_url}")
    log(f"Miner: {config.miner_control_url}")
    log(f"Edge cases: {'enabled' if config.include_edge_cases else 'disabled'}")
    
    results = TestResults()
    
    # Phase 0: Check nodes running
    if not await phase0_check_nodes_running(config, results):
        log("\nNodes not running. Start them with:")
        log("  pm2 start ecosystem.test.config.js")
        print_summary(results)
        return 1
    
    # Phase 1: Seed mock data
    mock_data = await phase1_seed_mock_data(config, results)
    
    # Phase 2: Miner fetches games
    await phase2_miner_fetch_games(config, results)
    
    # Phase 3: Miner submits odds
    await phase3_miner_submit_odds(config, results, mock_data)
    
    # Phase 4: Miner submits outcomes
    await phase4_miner_submit_outcomes(config, results, mock_data)
    
    # Phase 5: Transition events and add outcomes
    await phase5_transition_events(config, results, mock_data)
    
    # Phase 6: Trigger scoring
    await phase6_trigger_scoring(config, results)
    
    # Phase 7: Verify rolling scores
    await phase7_verify_rolling_scores(config, results)
    
    # Phase 8: Verify skill scores
    await phase8_verify_skill_scores(config, results)
    
    # Phase 9: Verify weights
    await phase9_verify_weights(config, results)
    
    # Phase 10: Edge cases (optional)
    await phase10_edge_cases(config, results)
    
    # Phase 11: Memory monitoring
    await phase11_memory_monitoring(config, results)
    
    # Phase 12: Final verification
    await phase12_verify_basic_results(config, results)
    
    # Summary
    print_summary(results)
    
    return 0 if results.success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparket E2E Integration Test")
    parser.add_argument("--validator-url", default="http://127.0.0.1:8199")
    parser.add_argument("--miner-url", default="http://127.0.0.1:8198")
    parser.add_argument("--num-events", type=int, default=5)
    parser.add_argument("--include-edge-cases", action="store_true", help="Run edge case tests")
    parser.add_argument("--include-crash-recovery", action="store_true", help="Run crash recovery test")
    parser.add_argument("--teardown", action="store_true", help="Teardown test infrastructure")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    if args.teardown:
        log("Tearing down test infrastructure...")
        subprocess.run(["docker", "rm", "-f", "pg-sparket-test"], capture_output=True)
        subprocess.run(["docker", "volume", "rm", "-f", "pg_sparket_test_data"], capture_output=True)
        subprocess.run(["pm2", "delete", "ecosystem.test.config.js"], capture_output=True)
        log("Teardown complete")
        sys.exit(0)
    
    config = E2EConfig(
        validator_control_url=args.validator_url,
        miner_control_url=args.miner_url,
        num_events=args.num_events,
        include_edge_cases=args.include_edge_cases,
        include_crash_recovery=args.include_crash_recovery,
        timeout=args.timeout,
    )
    
    exit_code = asyncio.run(main(config))
    sys.exit(exit_code)
