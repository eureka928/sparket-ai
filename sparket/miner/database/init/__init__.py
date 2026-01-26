from __future__ import annotations

import os
from typing import Optional

import bittensor as bt
from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from sparket.miner.config.config import Config


def alembic_env_path() -> str:
    """Return path to miner Alembic env (created alongside this package)."""
    return os.path.join(os.path.dirname(__file__), "..", "alembic")


def alembic_ini_path() -> str:
    return os.path.join(alembic_env_path(), "alembic.ini")


def initialize(config: Config, neuron_path: str | None = None) -> None:
    """
    Ensure the miner database exists and Alembic migrations are applied.

    Args:
        config: Miner application config.
        neuron_path: Deprecated - now uses project-relative data directory.
    """
    # Use project-relative data directory based on test mode
    test_mode = getattr(getattr(config, "core", None), "test_mode", False)
    db_path = config.miner.database_path(test_mode)
    
    # Ensure data directory exists
    data_dir = os.path.dirname(db_path)
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(db_path):
        bt.logging.info({"miner_db": {"message": "creating sqlite db", "path": db_path, "test_mode": test_mode}})
        open(db_path, "a", encoding="utf-8").close()

    upgrade_database(db_path=db_path)


def upgrade_database(*, db_path: str) -> None:
    """
    Run Alembic upgrade head for miner database inside given neuron path.
    """
    db_path = os.path.abspath(db_path)
    # Use synchronous sqlite driver for migrations to avoid async engine quirks.
    db_url = f"sqlite:///{db_path}"
    ini_path = alembic_ini_path()

    alembic_config = AlembicConfig(ini_path)
    alembic_config.set_main_option("sqlalchemy.url", db_url)

    script_directory = ScriptDirectory.from_config(alembic_config)
    head_revision = script_directory.get_current_head()
    current_revision = _get_database_revision(db_url)

    if head_revision is not None and current_revision == head_revision:
        bt.logging.info(
            {
                "miner_db": {
                    "event": "alembic_upgrade_skip",
                    "db_path": db_path,
                    "revision": current_revision,
                }
            }
        )
        return

    bt.logging.info(
        {
            "miner_db": {
                "message": "running alembic upgrade head",
                "config": ini_path,
                "db_path": db_path,
            }
        }
    )

    from time import monotonic

    started = monotonic()
    bt.logging.info(
        {
            "miner_db": {
                "event": "alembic_upgrade_start",
                "url": db_url,
            }
        }
    )
    try:
        command.upgrade(alembic_config, "head")
    except Exception as exc:
        bt.logging.error(
            {
                "miner_db": {
                    "event": "alembic_upgrade_error",
                    "error": str(exc),
                }
            }
        )
        raise
    else:
        elapsed = monotonic() - started
        bt.logging.info(
            {
                "miner_db": {
                    "event": "alembic_upgrade_complete",
                    "elapsed_seconds": round(elapsed, 3),
                }
            }
        )


def _get_database_revision(db_url: str) -> str | None:
    engine = create_engine(db_url, future=True)
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            if "alembic_version" not in inspector.get_table_names():
                return None
            result = connection.execute(text("select version_num from alembic_version limit 1"))
            return result.scalar()
    except Exception:
        return None
    finally:
        engine.dispose()

