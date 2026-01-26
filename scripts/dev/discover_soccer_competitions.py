#!/usr/bin/env python3
"""Discover available soccer competitions from SportsDataIO.

Run with:
    python scripts/dev/discover_soccer_competitions.py

Outputs a list of accessible competitions with their keys/IDs for config wiring.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sparket.providers.sportsdataio.client import SportsDataIOClient


async def main():
    async with SportsDataIOClient() as client:
        competitions = await client.fetch_soccer_competitions()
        print(f"\n{'='*60}")
        print(f"Found {len(competitions)} soccer competitions")
        print(f"{'='*60}\n")
        
        # Group by area
        by_area: dict[str, list] = {}
        for comp in competitions:
            area = comp.area_name or "Unknown"
            by_area.setdefault(area, []).append(comp)
        
        for area in sorted(by_area.keys()):
            print(f"\n## {area}")
            for comp in sorted(by_area[area], key=lambda c: c.name or ""):
                active_str = "✓" if comp.active else "✗"
                print(f"  [{active_str}] {comp.name:<40} key={comp.key!r:<20} id={comp.competition_id}")
        
        # Output config snippet for active competitions
        active = [c for c in competitions if c.active]
        print(f"\n{'='*60}")
        print(f"Active competitions: {len(active)}")
        print(f"{'='*60}\n")
        
        print("# Suggested LeagueCode enum additions:")
        for comp in sorted(active, key=lambda c: c.key or ""):
            if comp.key:
                enum_name = comp.key.upper().replace("-", "_").replace(" ", "_")
                print(f"    {enum_name} = \"{comp.key.lower()}\"")


if __name__ == "__main__":
    asyncio.run(main())
