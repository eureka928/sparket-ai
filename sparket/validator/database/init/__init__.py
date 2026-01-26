"""Database bootstrap utilities for the validator.

Responsibilities:
- Ensure Docker/Postgres services are available
- Ensure target database exists
- Run Alembic migrations to head
- Seed minimal reference data

Notes:
- Target platform: Ubuntu. We include a best-effort apt-based flow for dev
  environments but will no-op if not root or not Ubuntu.
"""

from __future__ import annotations

import asyncio
import os
import traceback
from typing import Optional

import bittensor as bt

from sparket.validator.config.config import Config

from .alembic import _alembic_upgrade_head, _get_current_revision
from .bootstrap import _bootstrap_role_and_database, _ensure_database_exists
from .docker import _ensure_docker_postgres_running
from .seed import _seed_reference_minimal
from .system import _ensure_postgres_running
from .utils import _command_exists, _is_root, _run, _sanitize_url_for_log

__all__ = [
    "initialize",
    "_sanitize_url_for_log",
    "_ensure_postgres_running",
    "_ensure_docker_postgres_running",
    "_ensure_database_exists",
    "_bootstrap_role_and_database",
    "_alembic_upgrade_head",
    "_get_current_revision",
    "_seed_reference_minimal",
    "_command_exists",
    "_run",
    "_is_root",
]


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def initialize(config: Config) -> None:
    """Initialize database: ensure services, migrations, and seeds."""
    bt.logging.info({"db_init": {"step": "initialize_start"}})

    # Determine test mode from config
    core = getattr(config, "core", None)
    test_mode = getattr(getattr(core, "runtime", None), "test_mode", False)
    if not test_mode:
        test_mode = getattr(core, "test_mode", False) if core else False
    bt.logging.info({"db_init": {"test_mode": test_mode}})

    bt.logging.info({"db_init": {"step": "docker_postgres_setup"}})
    core_db = getattr(core, "database", None)
    if core_db is not None:
        try:
            bt.logging.info({"db_init": {"docker_setup": "starting", "test_mode": test_mode}})
            _ensure_docker_postgres_running(core_db, test_mode=test_mode)
            bt.logging.info({"db_init": {"docker_setup": "complete"}})
        except Exception as exc:
            bt.logging.warning({"db_init": {"docker_setup_error": str(exc)}})
            bt.logging.debug(
                {"db_init": {"docker_setup_traceback": traceback.format_exc()}}
            )
    else:
        bt.logging.info({"db_init": {"docker_setup": "skipped_no_core_db"}})

    bt.logging.info({"db_init": {"step": "system_postgres_check"}})
    try:
        _ensure_postgres_running()
        bt.logging.info({"db_init": {"system_postgres_check": "complete"}})
    except Exception as exc:
        bt.logging.warning({"db_init": {"system_postgres_check_error": str(exc)}})

    bt.logging.info({"db_init": {"step": "resolve_database_config"}})
    database_url: Optional[str] = getattr(config, "database_url", None) or os.getenv(
        "DATABASE_URL"
    )
    db_name: Optional[str] = getattr(config, "database_name", None)

    if core_db:
        if not database_url:
            database_url = getattr(core_db, "url", None)
        if not db_name:
            # Use test_name when in test mode
            if test_mode:
                db_name = getattr(core_db, "test_name", None) or getattr(core_db, "name", None)
            else:
                db_name = getattr(core_db, "name", None)

    bt.logging.info(
        {
            "db_init": {
                "database_config": {
                    "url_set": bool(database_url),
                    "url_redacted": _sanitize_url_for_log(database_url)
                    if database_url
                    else None,
                    "db_name": db_name,
                }
            }
        }
    )

    if not database_url:
        bt.logging.error({"db_init": {"error": "no_database_url"}})
        raise ValueError("database_url is required but not found in config or environment")

    bt.logging.info({"db_init": {"step": "bootstrap_role_database"}})
    try:
        bootstrap_cfg = getattr(core_db, "bootstrap", None) if core_db else None
        admin_url = getattr(bootstrap_cfg, "admin_url", None) if bootstrap_cfg else None
        auto_create = bool(getattr(bootstrap_cfg, "auto_create", False)) if bootstrap_cfg else False
        if auto_create and admin_url:
            bt.logging.info(
                {
                    "db_init": {
                        "bootstrap": "enabled",
                        "admin_url_redacted": _sanitize_url_for_log(admin_url),
                    }
                }
            )
            _run_async(
                _bootstrap_role_and_database(
                    admin_url=admin_url,
                    user=getattr(core_db, "user", None) if core_db else None,
                    password=getattr(core_db, "password", None) if core_db else None,
                    db_name=db_name,
                )
            )
            bt.logging.info({"db_init": {"bootstrap": "complete"}})
        else:
            bt.logging.info(
                {
                    "db_init": {
                        "bootstrap": "skipped",
                        "auto": auto_create,
                        "admin_url_set": bool(admin_url),
                    }
                }
            )
    except Exception as exc:
        bt.logging.warning({"db_init": {"bootstrap_error": str(exc)}})
        bt.logging.debug({"db_init": {"bootstrap_traceback": traceback.format_exc()}})

    bt.logging.info({"db_init": {"step": "ensure_database_exists"}})
    try:
        _run_async(_ensure_database_exists(database_url, db_name))
        bt.logging.info({"db_init": {"ensure_database_exists": "complete"}})
    except Exception as exc:
        bt.logging.error(
            {
                "db_init": {
                    "ensure_database_exists_failed": True,
                    "error": str(exc),
                }
            }
        )
        bt.logging.debug(
            {"db_init": {"ensure_database_exists_traceback": traceback.format_exc()}}
        )
        raise

    bt.logging.info({"db_init": {"step": "alembic_migrations"}})
    try:
        _alembic_upgrade_head(database_url=database_url)
        bt.logging.info({"db_init": {"alembic_migrations": "complete"}})
    except Exception as exc:
        bt.logging.error({"db_init": {"alembic_migrations_error": str(exc)}})
        bt.logging.debug(
            {"db_init": {"alembic_migrations_traceback": traceback.format_exc()}}
        )
        raise

    bt.logging.info({"db_init": {"step": "seed_reference_data"}})
    try:
        _run_async(_seed_reference_minimal(config))
        bt.logging.info({"db_init": {"seed_reference_data": "complete"}})
    except Exception as exc:
        bt.logging.warning({"db_init": {"seed_reference_data_error": str(exc)}})
        bt.logging.debug(
            {"db_init": {"seed_reference_data_traceback": traceback.format_exc()}}
        )

    bt.logging.info({"db_init": {"initialize_complete": True}})

