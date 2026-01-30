"""Live test script - fetch real markets and generate predictions without submitting.

Fetches current markets from The-Odds-API, runs the ensemble model,
and displays detailed analysis of what would be submitted.

Usage:
    # Set API key first
    export SPARKET_CUSTOM_MINER__ODDS_API_KEY="your_key"

    # Run live test
    uv run python -m sparket.miner.custom.live_test

    # Specific sport
    uv run python -m sparket.miner.custom.live_test --sport NBA

    # Verbose output
    uv run python -m sparket.miner.custom.live_test -v
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sparket.miner.base.data.teams import get_team_by_name
from sparket.miner.custom.config import CustomMinerConfig
from sparket.miner.custom.data.fetchers.odds_api import OddsAPIFetcher
from sparket.miner.custom.data.seed_elo import seed_elo_ratings
from sparket.miner.custom.models.calibration.isotonic import IsotonicCalibrator
from sparket.miner.custom.models.engines.elo import EloEngine
from sparket.miner.custom.models.engines.ensemble import EnsembleEngine
from sparket.miner.custom.models.engines.poisson import PoissonEngine


def normalize_team_name(name: str, sport: str) -> str:
    """Convert full team name to code for Elo lookup.

    Args:
        name: Full team name (e.g., "New England Patriots")
        sport: Sport code (e.g., "NFL")

    Returns:
        Team code (e.g., "NE") or original name if not found
    """
    code = get_team_by_name(sport, name)
    return code if code else name


def format_odds(prob: float, vig: float = 0.02) -> str:
    """Format probability as decimal odds."""
    implied = prob + (vig / 2)
    implied = max(0.001, min(0.999, implied))
    odds = 1.0 / implied
    return f"{odds:.2f}"


def format_american(prob: float) -> str:
    """Format probability as American odds."""
    if prob >= 0.5:
        american = -100 * prob / (1 - prob)
    else:
        american = 100 * (1 - prob) / prob
    return f"{american:+.0f}"


async def run_live_test(
    sport: str = "NFL",
    verbose: bool = False,
    max_events: int = 10,
) -> None:
    """Run live test against real markets.

    Args:
        sport: Sport to test (NFL, NBA, MLB, NHL)
        verbose: Show detailed output
        max_events: Maximum events to analyze
    """
    print("=" * 70)
    print("LIVE MARKET TEST - Ensemble Model")
    print("=" * 70)
    print(f"Sport: {sport}")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    # Load config
    config = CustomMinerConfig.from_env()

    if not config.odds_api_key:
        print("\nERROR: No API key set!")
        print("Set: export SPARKET_CUSTOM_MINER__ODDS_API_KEY='your_key'")
        return

    # Initialize components
    data_dir = Path.home() / ".sparket" / "custom_miner"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\nInitializing models...")

    # Elo engine with seeded ratings
    elo = EloEngine(
        config=config.elo,
        vig=config.vig,
        ratings_path=str(data_dir / "elo_ratings.json"),
    )

    # Seed Elo if empty
    if not elo._store._ratings:
        print("Seeding Elo ratings...")
        seed_elo_ratings(elo)

    # Poisson engine
    poisson = PoissonEngine(
        data_path=str(data_dir / "poisson_profiles.json"),
    )

    # Ensemble
    ensemble = EnsembleEngine(
        elo_engine=elo,
        poisson_engine=poisson,
        base_weights=config.engine_weights,
        confidence_scaling=True,
    )

    # Calibrator (load if exists)
    calibrator = IsotonicCalibrator(
        min_samples=config.calibration.min_samples,
        data_path=str(data_dir / "calibration.json"),
    )

    # Odds API fetcher
    odds_api = OddsAPIFetcher(
        api_key=config.odds_api_key,
        cache_ttl_seconds=60,
    )

    print(f"API requests remaining: {odds_api.remaining_requests or 'unknown'}")

    # Fetch live odds
    print(f"\nFetching live {sport} markets...")

    try:
        events = await odds_api.get_all_games(sport)
    except Exception as e:
        print(f"ERROR fetching events: {e}")
        await odds_api.close()
        return

    if not events:
        print(f"No upcoming {sport} events found.")
        await odds_api.close()
        return

    print(f"Found {len(events)} upcoming events")
    print(f"API requests remaining: {odds_api.remaining_requests}")

    # Analyze each event
    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)

    results = []

    for i, event in enumerate(events[:max_events]):
        # event is a MarketOdds object
        home_team = event.home_team
        away_team = event.away_team
        game_time = event.commence_time

        if not home_team or not away_team:
            continue

        # Calculate hours to game
        try:
            hours_to_game = (game_time - datetime.now(timezone.utc)).total_seconds() / 3600
        except:
            hours_to_game = 24.0

        # Use the MarketOdds directly (already has consensus)
        market_odds = event

        # Convert team names to codes for Elo lookup
        home_code = normalize_team_name(home_team, sport)
        away_code = normalize_team_name(away_team, sport)

        # Build market dict with codes for Elo
        market = {
            "market_id": i + 1,
            "kind": "MONEYLINE",
            "home_team": home_code,
            "away_team": away_code,
            "sport": sport,
        }

        # Get ensemble prediction
        pred = ensemble.predict(market, market_odds=market_odds)

        if pred is None:
            if verbose:
                print(f"\n{i+1}. {away_team} @ {home_team}")
                print("   Could not generate prediction")
            continue

        # Apply calibration if fitted
        cal_home = pred.home_prob
        cal_away = pred.away_prob
        if calibrator.is_fitted:
            cal_home, cal_away = calibrator.calibrate_pair(cal_home, cal_away)

        # Calculate edge vs market
        edge = None
        market_prob = None
        if market_odds:
            market_prob = market_odds.home_prob
            edge = cal_home - market_prob

        # Store result
        results.append({
            "home_team": home_team,
            "away_team": away_team,
            "our_prob": cal_home,
            "market_prob": market_prob,
            "edge": edge,
            "confidence": pred.confidence,
            "dominant": pred.dominant_source,
            "agreed": pred.models_agreed,
            "hours_to_game": hours_to_game,
            "game_time": game_time,
        })

        # Print result
        print(f"\n{i+1}. {away_team} @ {home_team}")
        if game_time:
            print(f"   Game: {game_time.strftime('%a %b %d, %I:%M %p')} ({hours_to_game:.1f}h)")

        print(f"   Our prediction:    {home_team} {cal_home*100:.1f}% | {away_team} {cal_away*100:.1f}%")
        print(f"   Our odds:          {home_team} {format_odds(cal_home)} ({format_american(cal_home)}) | {away_team} {format_odds(cal_away)} ({format_american(cal_away)})")

        if market_prob:
            print(f"   Market consensus:  {home_team} {market_prob*100:.1f}% | {away_team} {(1-market_prob)*100:.1f}%")
            print(f"   Market odds:       {home_team} {format_odds(market_prob)} ({format_american(market_prob)}) | {away_team} {format_odds(1-market_prob)} ({format_american(1-market_prob)})")

            edge_pct = edge * 100
            edge_symbol = "+" if edge > 0 else ""
            edge_color = "✓" if abs(edge) > 0.02 else "~"
            print(f"   Edge vs market:    {edge_symbol}{edge_pct:.1f}% {edge_color}")

        if verbose:
            print(f"   Confidence:        {pred.confidence:.2f}")
            print(f"   Dominant model:    {pred.dominant_source}")
            print(f"   Models agreed:     {pred.models_agreed}")
            if pred.components:
                print("   Components:")
                for comp in pred.components:
                    if comp.home_prob is not None:
                        print(f"     - {comp.source}: {comp.home_prob*100:.1f}% (weight: {comp.weight:.2f})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        edges = [r["edge"] for r in results if r["edge"] is not None]
        if edges:
            avg_edge = sum(edges) / len(edges)
            positive_edges = sum(1 for e in edges if e > 0.02)
            negative_edges = sum(1 for e in edges if e < -0.02)

            print(f"Events analyzed:      {len(results)}")
            print(f"Avg edge vs market:   {avg_edge*100:+.2f}%")
            print(f"Positive edges (>2%): {positive_edges}")
            print(f"Negative edges (<-2%): {negative_edges}")
            print(f"Neutral edges:        {len(edges) - positive_edges - negative_edges}")

            # Best opportunities
            sorted_by_edge = sorted(
                [r for r in results if r["edge"] is not None],
                key=lambda x: abs(x["edge"]),
                reverse=True
            )

            if sorted_by_edge:
                print("\nBest opportunities (by edge magnitude):")
                for r in sorted_by_edge[:3]:
                    team = r["home_team"] if r["edge"] > 0 else r["away_team"]
                    print(f"  {team}: {r['edge']*100:+.1f}% edge ({r['away_team']} @ {r['home_team']})")

    print(f"\nAPI requests remaining: {odds_api.remaining_requests}")

    # Cleanup
    await odds_api.close()


def main():
    parser = argparse.ArgumentParser(description="Live test against real markets")
    parser.add_argument("--sport", default="NFL", choices=["NFL", "NBA", "MLB", "NHL"],
                        help="Sport to test")
    parser.add_argument("--max-events", type=int, default=10,
                        help="Maximum events to analyze")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    asyncio.run(run_live_test(
        sport=args.sport,
        verbose=args.verbose,
        max_events=args.max_events,
    ))


if __name__ == "__main__":
    main()
