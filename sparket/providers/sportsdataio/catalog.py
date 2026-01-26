"""Team catalog helpers for SportsDataIO.

Transforms provider team payloads into `reference.Team` rows and supports
building fast lookup indices for resolving event team IDs.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from .types import Team, Game


def sport_rows() -> List[dict]:
    """Return canonical sport rows for reference.Sport upsert.

    Codes should be stable lowercase identifiers used across the system.
    """
    return [
        {"code": "nfl", "name": "NFL"},
        {"code": "nba", "name": "NBA"},
        {"code": "mlb", "name": "MLB"},
        {"code": "nhl", "name": "NHL"},
        {"code": "soccer", "name": "Soccer"},
    ]

def team_rows_from_catalog(teams: Iterable[Team], league_id: int) -> List[dict]:
    """Transform provider teams into rows for `reference.Team`.

    Output fields match docs/validator_schema/reference.Team columns.
    - name: prefer City + Name if both present; else Name
    - abbrev: Key
    - ext_ref['sportsdataio']: {'TeamID': team_id, 'Key': key}
    """
    rows: List[dict] = []
    for t in teams:
        if t.city and t.name:
            display_name = f"{t.city} {t.name}"
        else:
            display_name = t.name or t.key
        rows.append(
            {
                "league_id": league_id,
                "name": display_name,
                "abbrev": t.key,
                "ext_ref": {
                    "sportsdataio": {
                        "TeamID": t.team_id,
                        "Key": t.key,
                        "Conference": t.conference,
                        "Division": t.division,
                    }
                },
            }
        )
    return rows


def build_team_index_by_sdio(rows: Iterable[dict]) -> Dict[str, int]:
    """Build a lookup: provider team key -> internal team_id.

    Expects each row to include: team_id, ext_ref['sportsdataio']['Key']
    """
    index: Dict[str, int] = {}
    for r in rows:
        ext = (r.get("ext_ref") or {}).get("sportsdataio") or {}
        key = ext.get("Key")
        if key:
            index[key] = r["team_id"]
    return index


def resolve_game_team_ids(game: Game, team_index_by_key: Dict[str, int]) -> Tuple[Optional[int], Optional[int]]:
    """Resolve home/away team_ids from a provider `Game` and index by provider `Key`.

    Returns (home_team_id, away_team_id), allowing None if not found.
    """
    home_id = team_index_by_key.get(game.home_team)
    away_id = team_index_by_key.get(game.away_team)
    return home_id, away_id


__all__ = [
    "sport_rows",
    "team_rows_from_catalog",
    "build_team_index_by_sdio",
    "resolve_game_team_ids",
]


