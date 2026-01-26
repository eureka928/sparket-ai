#!/usr/bin/env python3
"""Diagnose SportsDataIO API connectivity and data availability.

This script tests each configured league endpoint and reports:
- HTTP connectivity
- Number of games/events returned
- Sample data for verification

Usage:
    # Test all leagues
    uv run python scripts/dev/diagnose_sdio.py
    
    # Test specific leagues
    uv run python scripts/dev/diagnose_sdio.py --league nba --league nfl
    
    # Test with verbose output
    uv run python scripts/dev/diagnose_sdio.py -v
"""

import argparse
import asyncio
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sparket.providers.sportsdataio.client import SportsDataIOClient
from sparket.providers.sportsdataio.config import LeagueConfig, build_default_config


class DiagnosticResult:
    def __init__(self, league_code: str, sport: str):
        self.league_code = league_code
        self.sport = sport
        self.schedule_ok = False
        self.schedule_count = 0
        self.schedule_error: Optional[str] = None
        self.schedule_sample: List[Dict] = []
        self.odds_ok = False
        self.odds_count = 0
        self.odds_error: Optional[str] = None
        self.odds_sample: Dict = {}
        self.teams_ok = False
        self.teams_count = 0
        self.teams_error: Optional[str] = None


async def test_league(
    client: SportsDataIOClient,
    config: LeagueConfig,
    verbose: bool = False,
) -> DiagnosticResult:
    """Test a single league's endpoints."""
    result = DiagnosticResult(config.code.value, config.sport_code)
    now = datetime.now(timezone.utc)
    
    # Test schedule endpoint
    print(f"\n  📅 Schedule ({config.schedule_mode})...", end=" ", flush=True)
    try:
        if config.schedule_mode == "season":
            # For NFL, need to construct season code
            year = now.year
            season_type = config.season_type or "REG"
            fmt = config.season_format or "{year}"
            season_code = fmt.format(year=year, season_type=season_type, SEASONTYPE=season_type, YEAR=year)
            games = await client.fetch_schedule_season(config, season_code, season_type=season_type)
        else:
            # Date-based: check today and next 7 days
            games = []
            for days_ahead in range(8):
                target = date.today() + timedelta(days=days_ahead)
                day_games = await client.fetch_schedule_by_date(config, target)
                games.extend(day_games)
                if day_games and verbose:
                    print(f"\n      {target}: {len(day_games)} games", end="")
        
        result.schedule_ok = True
        result.schedule_count = len(games)
        
        # Get sample with key info
        for game in games[:3]:
            result.schedule_sample.append({
                "game_id": game.game_id,
                "home": game.home_team,
                "away": game.away_team,
                "date": str(game.date_time)[:19] if game.date_time else None,
                "status": game.status.value if game.status else None,
            })
        
        if games:
            print(f"✅ {len(games)} games")
        else:
            print(f"⚠️  0 games (may be off-season)")
            
    except Exception as e:
        result.schedule_error = str(e)
        print(f"❌ {e}")
    
    # Test odds endpoint (if we have games)
    if result.schedule_count > 0 and result.schedule_sample:
        game_id = result.schedule_sample[0].get("game_id")
        if game_id:
            print(f"  📊 Odds (game {game_id})...", end=" ", flush=True)
            try:
                odds_set = await client.fetch_line_history(config, game_id)
                if odds_set and odds_set.pregame:
                    result.odds_ok = True
                    result.odds_count = len(odds_set.pregame)
                    # Sample first odds entry
                    first = odds_set.pregame[0]
                    result.odds_sample = {
                        "sportsbook": first.sportsbook,
                        "moneyline": {
                            "home": first.moneyline.home if first.moneyline else None,
                            "away": first.moneyline.away if first.moneyline else None,
                        } if first.moneyline else None,
                        "updated": str(first.updated)[:19] if first.updated else None,
                    }
                    print(f"✅ {len(odds_set.pregame)} line movements")
                else:
                    print(f"⚠️  No odds data available")
            except Exception as e:
                result.odds_error = str(e)
                print(f"❌ {e}")
    
    # Test teams endpoint
    if config.teams_url:
        print(f"  👥 Teams...", end=" ", flush=True)
        try:
            teams = await client.fetch_team_catalog(config)
            result.teams_ok = True
            result.teams_count = len(teams)
            print(f"✅ {len(teams)} teams")
        except Exception as e:
            result.teams_error = str(e)
            print(f"❌ {e}")
    
    return result


async def run_diagnostics(
    leagues: Optional[List[str]] = None,
    include_soccer: bool = False,
    verbose: bool = False,
) -> List[DiagnosticResult]:
    """Run diagnostics on all configured leagues."""
    
    # Check for API key
    api_key = os.getenv("SDIO_API_KEY")
    if not api_key:
        print("\n❌ ERROR: SDIO_API_KEY environment variable not set!")
        print("   Set it with: export SDIO_API_KEY=your_key_here")
        return []
    
    print(f"\n🔑 API Key: {api_key[:8]}...{api_key[-4:]}")
    
    config = build_default_config(include_soccer=include_soccer)
    
    # Filter leagues if specified
    if leagues:
        allowed = {code.lower() for code in leagues}
        config.leagues = [lc for lc in config.leagues if lc.code.value.lower() in allowed]
        missing = allowed - {lc.code.value.lower() for lc in config.leagues}
        if missing:
            print(f"\n⚠️  Unknown leagues: {missing}")
    
    print(f"\n📋 Testing {len(config.leagues)} leagues...")
    
    results: List[DiagnosticResult] = []
    
    async with SportsDataIOClient(config=config) as client:
        for league_config in config.leagues:
            print(f"\n{'='*50}")
            print(f"🏆 {league_config.code.value.upper()} ({league_config.sport_code})")
            print(f"   Schedule URL: {league_config.schedule_url[:60]}...")
            
            result = await test_league(client, league_config, verbose=verbose)
            results.append(result)
            
            if verbose and result.schedule_sample:
                print(f"\n  📝 Sample games:")
                for game in result.schedule_sample:
                    print(f"      {game['home']} vs {game['away']} @ {game['date']}")
    
    return results


def print_summary(results: List[DiagnosticResult]):
    """Print summary table of results."""
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"{'League':<10} {'Sport':<12} {'Schedule':<12} {'Odds':<10} {'Teams':<10}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    
    for r in results:
        sched = f"✅ {r.schedule_count}" if r.schedule_ok else f"❌"
        odds = f"✅ {r.odds_count}" if r.odds_ok else ("⚠️" if r.schedule_count == 0 else "❌")
        teams = f"✅ {r.teams_count}" if r.teams_ok else "❌"
        print(f"{r.league_code.upper():<10} {r.sport:<12} {sched:<12} {odds:<10} {teams:<10}")
    
    # Overall status
    total = len(results)
    ok = sum(1 for r in results if r.schedule_ok)
    with_games = sum(1 for r in results if r.schedule_count > 0)
    
    print(f"\n✅ {ok}/{total} leagues responding")
    print(f"📅 {with_games}/{total} leagues have upcoming games")
    
    # Warnings
    no_games = [r for r in results if r.schedule_ok and r.schedule_count == 0]
    if no_games:
        print(f"\n⚠️  Leagues with no games (may be off-season):")
        for r in no_games:
            print(f"   - {r.league_code.upper()}")
    
    errors = [r for r in results if r.schedule_error]
    if errors:
        print(f"\n❌ Leagues with errors:")
        for r in errors:
            print(f"   - {r.league_code.upper()}: {r.schedule_error}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose SportsDataIO API connectivity")
    parser.add_argument(
        "--league", "-l",
        dest="leagues",
        action="append",
        help="Specific league codes to test (e.g., nfl, nba). Repeat for multiple.",
    )
    parser.add_argument(
        "--soccer", "-s",
        action="store_true",
        help="Include soccer leagues (requires separate SDIO subscription)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including sample games",
    )
    args = parser.parse_args()
    
    print("🔍 SportsDataIO Diagnostic Tool")
    print(f"   Time: {datetime.now(timezone.utc).isoformat()}")
    
    results = asyncio.run(run_diagnostics(
        leagues=args.leagues,
        include_soccer=args.soccer,
        verbose=args.verbose,
    ))
    
    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
