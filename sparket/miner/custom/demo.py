#!/usr/bin/env python
"""Demo the custom miner without Bittensor network.

This runs the odds generation pipeline locally to show it working.

Usage:
    uv run python -m sparket.miner.custom.demo
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone

from sparket.miner.custom.config import CustomMinerConfig
from sparket.miner.custom.models.engines.elo import EloEngine
from sparket.miner.custom.data.fetchers.odds_api import OddsAPIFetcher, blend_with_market
from sparket.miner.custom.data.seed_elo import seed_elo_ratings


async def main():
    print("=" * 60)
    print("Custom Miner Demo (no network required)")
    print("=" * 60)

    # Load config
    config = CustomMinerConfig.from_env()

    # Initialize Elo engine with seeded ratings
    print("\n1. Initializing Elo engine with real ratings...")
    elo = EloEngine(config=config.elo, vig=config.vig)
    count = seed_elo_ratings(elo, sports=["NFL", "NBA"])
    print(f"   Seeded {count} team ratings")

    # Initialize Odds API if key available
    odds_api = None
    api_key = config.odds_api_key or os.getenv("ODDS_API_KEY")
    if api_key:
        print("\n2. Connecting to The-Odds-API...")
        odds_api = OddsAPIFetcher(api_key=api_key)
        print("   Connected!")
    else:
        print("\n2. No API key - skipping market data")
        print("   (Set SPARKET_CUSTOM_MINER__ODDS_API_KEY or ODDS_API_KEY to enable)")

    # Demo: Generate odds for sample games
    print("\n3. Generating odds for sample matchups...")
    print("-" * 60)

    sample_games = [
        {"home_team": "KC", "away_team": "BUF", "sport": "NFL"},
        {"home_team": "SF", "away_team": "DAL", "sport": "NFL"},
        {"home_team": "CAR", "away_team": "DET", "sport": "NFL"},
        {"home_team": "BOS", "away_team": "LAL", "sport": "NBA"},
        {"home_team": "MIL", "away_team": "PHX", "sport": "NBA"},
    ]

    for game in sample_games:
        home = game["home_team"]
        away = game["away_team"]
        sport = game["sport"]

        # Get Elo prediction
        market = {"market_id": 1, "kind": "MONEYLINE", **game}
        elo_odds = elo.get_odds_sync(market)

        if elo_odds is None:
            continue

        print(f"\n{away} @ {home} ({sport})")
        print(f"  Elo Model:    {home} {elo_odds.home_prob:.1%} | {away} {elo_odds.away_prob:.1%}")
        print(f"  Elo Odds:     {elo_odds.home_odds_eu:.2f} / {elo_odds.away_odds_eu:.2f}")

        # Get market consensus if available
        if odds_api:
            try:
                market_odds = await odds_api.get_consensus_odds(sport, home, away)
                if market_odds:
                    print(f"  Market:       {home} {market_odds.home_prob:.1%} | {away} {market_odds.away_prob:.1%}")
                    print(f"  Market Odds:  {market_odds.home_odds:.2f} / {market_odds.away_odds:.2f}")
                    print(f"  Books: {market_odds.num_books} (Pinnacle: {'✓' if market_odds.has_pinnacle else '✗'})")

                    # Blend
                    blended = blend_with_market(
                        model_prob=elo_odds.home_prob,
                        market_prob=market_odds.home_prob,
                        model_weight=0.4,
                        vig=config.vig,
                    )
                    print(f"  BLENDED:      {home} {blended.home_prob:.1%} | {away} {blended.away_prob:.1%}")
                    print(f"  Final Odds:   {blended.home_odds_eu:.2f} / {blended.away_odds_eu:.2f}")
                else:
                    print(f"  Market:       (game not found)")
            except Exception as e:
                print(f"  Market:       (error: {e})")

    # Show API usage
    if odds_api:
        print("\n" + "-" * 60)
        print(f"API requests remaining: {odds_api.remaining_requests}")
        await odds_api.close()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nTo run on real network:")
    print("  1. Create wallet: btcli wallet new_coldkey")
    print("  2. Register: btcli subnet register --netuid <NETUID>")
    print("  3. Run: uv run python -m sparket.miner.custom.runner")


if __name__ == "__main__":
    asyncio.run(main())
