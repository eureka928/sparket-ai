"""League and competition catalog models/mappers for SportsDataIO.

These are used to populate `reference.League` rows and build indices that
map provider keys/ids to internal `league_id`s.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LeagueEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore", frozen=True)

    league_id: Optional[int] = Field(default=None, alias="LeagueID")
    key: Optional[str] = Field(default=None, alias="Key")
    code: Optional[str] = Field(default=None, alias="Code")
    name: Optional[str] = Field(default=None, alias="Name")
    active: Optional[bool] = Field(default=None, alias="Active")


class SoccerCompetition(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore", frozen=True)

    competition_id: Optional[int] = Field(default=None, alias="CompetitionId")
    key: Optional[str] = Field(default=None, alias="Key")
    name: Optional[str] = Field(default=None, alias="Name")
    area_name: Optional[str] = Field(default=None, alias="AreaName")
    active: Optional[bool] = Field(default=None, alias="Active")


def league_rows_from_catalog(entries: Iterable[LeagueEntry], sport_id: int) -> List[dict]:
    rows: List[dict] = []
    for e in entries:
        if not (e.name or e.key or e.code):
            continue
        rows.append(
            {
                "sport_id": sport_id,
                "code": (e.code or e.key or e.name or ""),
                "name": (e.name or e.code or e.key or ""),
                "ext_ref": {
                    "sportsdataio": {
                        "LeagueID": e.league_id,
                        "Key": e.key,
                        "Code": e.code,
                        "Active": e.active,
                    }
                },
            }
        )
    return rows


def league_rows_from_soccer_competitions(entries: Iterable[SoccerCompetition], sport_id: int) -> List[dict]:
    rows: List[dict] = []
    for e in entries:
        if not (e.name or e.key or e.competition_id):
            continue
        rows.append(
            {
                "sport_id": sport_id,
                "code": (e.key or str(e.competition_id) or e.name or ""),
                "name": e.name or e.key or str(e.competition_id) or "",
                "ext_ref": {
                    "sportsdataio": {
                        "CompetitionId": e.competition_id,
                        "Key": e.key,
                        "AreaName": e.area_name,
                        "Active": e.active,
                    }
                },
            }
        )
    return rows


def build_league_index_by_sdio(rows: Iterable[dict]) -> Dict[str, int]:
    """Build a lookup from provider key/code -> internal league_id."""
    index: Dict[str, int] = {}
    for r in rows:
        ext = (r.get("ext_ref") or {}).get("sportsdataio") or {}
        for k in (ext.get("Key"), ext.get("Code")):
            if k:
                index[k] = r["league_id"]
    return index


__all__ = [
    "LeagueEntry",
    "SoccerCompetition",
    "league_rows_from_catalog",
    "league_rows_from_soccer_competitions",
    "build_league_index_by_sdio",
]


