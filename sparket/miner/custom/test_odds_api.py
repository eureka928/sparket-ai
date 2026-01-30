#!/usr/bin/env python
"""Test The-Odds-API integration.

Usage:
    # With API key
    export ODDS_API_KEY=your_key_here
    python -m sparket.miner.custom.test_odds_api

    # Or pass directly
    python -m sparket.miner.custom.test_odds_api --api-key your_key_here
"""

import argparse
import asyncio
import os

from sparket.miner.custom.data.fetchers.odds_api import (
    OddsAPIFetcher,
    blend_with_market,
)


async def main():
    parser = argparse.ArgumentParser(description="Test The-Odds-API")
    parser.add_argument("--api-key", help="API key (or set ODDS_API_KEY env var)")
    parser.add_argument("--sport", default="NFL", help="Sport to test")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ODDS_API_KEY") or os.getenv("SPARKET_CUSTOM_MINER__ODDS_API_KEY")

    if not api_key:
        print("ERROR: No API key provided")
        print()
        print("Get a free key at: https://the-odds-api.com/")
        print()
        print("Then run:")
        print("  export ODDS_API_KEY=your_key_here")
        print("  python -m sparket.miner.custom.test_odds_api")
        return

    print(f"Testing The-Odds-API for {args.sport}...")
    print("=" * 60)

    fetcher = OddsAPIFetcher(api_key=api_key)

    try:
        # Get all games
        games = await fetcher.get_all_games(args.sport)

        if not games:
            print(f"No games found for {args.sport}")
            print("(Season may be off, or sport code invalid)")
            return

        print(f"Found {len(games)} upcoming games:\n")

        for game in games[:5]:  # Show first 5
            print(f"{game.away_team} @ {game.home_team}")
            print(f"  Start: {game.commence_time}")
            print(f"  Consensus: {game.home_team} {game.home_prob:.1%} | {game.away_team} {game.away_prob:.1%}")
            print(f"  Odds: {game.home_odds} / {game.away_odds}")
            print(f"  Books: {game.num_books} (Pinnacle: {'✓' if game.has_pinnacle else '✗'})")

            # Show individual books
            if game.book_odds:
                print("  Book odds:")
                for book, odds in list(game.book_odds.items())[:3]:
                    print(f"    {book}: {odds['home']:.2f} / {odds['away']:.2f}")

            print()

        # Test blending
        if games:
            game = games[0]
            print("=" * 60)
            print("Blend Example:")
            print(f"  Model predicts: {game.home_team} 55%")
            print(f"  Market says: {game.home_team} {game.home_prob:.1%}")

            blended = blend_with_market(
                model_prob=0.55,
                market_prob=game.home_prob,
                model_weight=0.4,
            )
            print(f"  Blended (40/60): {game.home_team} {blended.home_prob:.1%}")
            print(f"  Final odds: {blended.home_odds_eu}")

        print()
        print("=" * 60)
        print(f"API requests remaining: {fetcher.remaining_requests}")

    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
