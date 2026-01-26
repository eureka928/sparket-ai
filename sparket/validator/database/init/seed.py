"""Reference data seeding for validator database."""

from __future__ import annotations

import bittensor as bt
from sqlalchemy import text

import json

from sparket.validator.config.config import Config
from sparket.validator.database.dbm import DBM
from sparket.providers.sportsdataio.catalog import sport_rows

__all__ = ["_seed_reference_minimal"]


async def _seed_reference_minimal(config: Config) -> None:
    """Ensure minimal reference rows exist after migrations."""
    bt.logging.info({"db_init": {"seed_reference_data": "starting"}})
    dbm = DBM.create_worker(config, pool_size=1, max_overflow=0, echo=False)
    try:
        # 1. Seed Provider
        exists = await dbm.read(
            text("select 1 from provider where code = :code limit 1"),
            {"code": "SDIO"},
        )
        if not exists:
            await dbm.write(
                text(
                    "insert into provider (code, name) values (:code, :name) on conflict (code) do nothing"
                ),
                {"code": "SDIO", "name": "SportsDataIO"},
            )
            bt.logging.info(
                {"db_init": {"seed_reference_data": "provider_seeded", "code": "SDIO"}}
            )

        # 2. Seed Sports
        sports = sport_rows()
        for s in sports:
            await dbm.write(
                text(
                    "insert into sport (code, name) values (:code, :name) on conflict (code) do nothing"
                ),
                {"code": s["code"], "name": s["name"]},
            )

        # 3. Seed Leagues for major sports
        # Core US sports (included in base SDIO subscription)
        league_specs = [
            {"sport_code": "nfl", "code": "nfl", "name": "NFL", "ext_key": "NFL"},
            {"sport_code": "nba", "code": "nba", "name": "NBA", "ext_key": "NBA"},
            {"sport_code": "mlb", "code": "mlb", "name": "MLB", "ext_key": "MLB"},
            {"sport_code": "nhl", "code": "nhl", "name": "NHL", "ext_key": "NHL"},
            # Soccer leagues (require separate SDIO soccer subscription)
            {"sport_code": "soccer", "code": "epl", "name": "Premier League", "ext_key": "EPL"},
            {"sport_code": "soccer", "code": "ucl", "name": "UEFA Champions League", "ext_key": "UCL"},
            {"sport_code": "soccer", "code": "uel", "name": "UEFA Europa League", "ext_key": "UEL"},
            {"sport_code": "soccer", "code": "esp", "name": "La Liga", "ext_key": "ESP"},
            {"sport_code": "soccer", "code": "deb", "name": "Bundesliga", "ext_key": "DEB"},
            {"sport_code": "soccer", "code": "itsa", "name": "Serie A", "ext_key": "ITSA"},
            {"sport_code": "soccer", "code": "frl1", "name": "Ligue 1", "ext_key": "FRL1"},
            {"sport_code": "soccer", "code": "mls", "name": "Major League Soccer", "ext_key": "MLS"},
            {"sport_code": "soccer", "code": "nle", "name": "Eredivisie", "ext_key": "NLE"},
        ]

        for spec in league_specs:
            sport_row = await dbm.read(
                text("select sport_id from sport where code = :code limit 1"),
                {"code": spec["sport_code"]},
                mappings=True,
            )
            if not sport_row:
                bt.logging.warning(
                    {
                        "db_init": {
                            "seed_reference_data": "sport_missing_for_league",
                            "league": spec["code"],
                            "sport_code": spec["sport_code"],
                        }
                    }
                )
                continue
            sport_id = sport_row[0]["sport_id"]
            exists_league = await dbm.read(
                text("select 1 from league where sport_id = :sport_id and code = :code limit 1"),
                {"sport_id": sport_id, "code": spec["code"]},
            )
            if exists_league:
                continue
            await dbm.write(
                text(
                    """
                    insert into league (sport_id, code, name, ext_ref)
                    values (:sport_id, :code, :name, :ext_ref)
                    on conflict (sport_id, code) do nothing
                    """
                ),
                {
                    "sport_id": sport_id,
                    "code": spec["code"],
                    "name": spec["name"],
                    "ext_ref": json.dumps({"sportsdataio": {"Key": spec["ext_key"]}}),
                },
            )
            bt.logging.info(
                {"db_init": {"seed_reference_data": "league_seeded", "code": spec["code"]}}
            )

    finally:
        await dbm.dispose()
        bt.logging.info({"db_init": {"seed_reference_data": "finished"}})
