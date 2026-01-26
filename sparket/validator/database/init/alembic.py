"""Alembic helpers for validator database initialization."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import bittensor as bt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from .utils import _command_exists, _sanitize_url_for_log

__all__ = [
    "_get_current_revision",
    "_alembic_upgrade_head",
]


def _get_current_revision(database_url: Optional[str]) -> Optional[str]:
    if not database_url:
        return None

    async def _probe_async(url: str) -> Optional[str]:
        engine = create_async_engine(url, future=True)
        try:
            async with engine.connect() as conn:
                result = await conn.execute(
                    text("select version_num from alembic_version")
                )
                return result.scalar_one_or_none()
        finally:
            await engine.dispose()

    try:
        try:
            return asyncio.run(_probe_async(database_url))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_probe_async(database_url))
            finally:
                loop.close()
    except Exception as exc:
        bt.logging.debug(
            {
                "db_init": {
                    "alembic_revision_probe_failed": str(exc),
                    "url": _sanitize_url_for_log(database_url),
                }
            }
        )
        return None


def _alembic_upgrade_head(database_url: Optional[str] = None) -> None:
    """Run Alembic migrations to head with robust path resolution."""
    alembic_root = Path(__file__).resolve().parent.parent
    alembic_ini = alembic_root / "alembic.ini"
    alembic_scripts = alembic_root / "alembic"

    if not alembic_ini.exists():
        raise FileNotFoundError(f"Alembic config not found at {alembic_ini}")
    if not alembic_scripts.exists():
        raise FileNotFoundError(
            f"Alembic scripts directory not found at {alembic_scripts}"
        )

    bt.logging.info(
        {
            "db_init": {
                "alembic_prepare": {
                    "ini": str(alembic_ini),
                    "scripts": str(alembic_scripts),
                    "url_redacted": _sanitize_url_for_log(database_url)
                    if database_url
                    else None,
                }
            }
        }
    )

    versions_dir = alembic_scripts / "versions"
    if not versions_dir.exists():
        bt.logging.info({"db_init": {"alembic_migrations": "skipped_no_versions_dir"}})
        return
    version_files = sorted(f.name for f in versions_dir.glob("*.py"))
    if not version_files:
        bt.logging.info(
            {"db_init": {"alembic_migrations": "skipped_no_migration_files"}}
        )
        return
    bt.logging.info(
        {
            "db_init": {
                "alembic_versions_detected": version_files,
            }
        }
    )
    # NOTE: We don't try to determine "head" from filenames - alphabetical order
    # doesn't match dependency order. Let Alembic determine head via its API.

    try:
        from alembic import command
        from alembic.config import Config as AlembicConfig

        cfg = AlembicConfig(str(alembic_ini))
        cfg.set_main_option("script_location", str(alembic_scripts))
        if database_url:
            cfg.set_main_option("sqlalchemy.url", database_url)
        os.environ.setdefault("ALEMBIC_SKIP_GEN_API", "1")
        before_revision = _get_current_revision(database_url)
        # Always let Alembic determine if upgrade is needed - it knows the real head
        bt.logging.info(
            {
                "db_init": {
                    "alembic_api": {
                        "event": "command.upgrade_start",
                        "from_revision": before_revision,
                        "url": _sanitize_url_for_log(database_url)
                        if database_url
                        else None,
                    }
                }
            }
        )
        started_at = time.monotonic()
        command.upgrade(cfg, "head")
        # CRITICAL: Alembic's logging.basicConfig() disables the 'bittensor' logger
        # Re-enable it after alembic runs
        import logging as std_logging
        std_logging.getLogger("bittensor").disabled = False
        
        elapsed = time.monotonic() - started_at
        after_revision = _get_current_revision(database_url)
        bt.logging.info(
            {
                "db_init": {
                    "alembic_api": {
                        "event": "command.upgrade_complete",
                        "from_revision": before_revision,
                        "to_revision": after_revision,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                }
            }
        )
    except Exception as err:
        bt.logging.error({"db_init": {"alembic_api_error": str(err)}})
        if not _command_exists("alembic"):
            raise

        env = os.environ.copy()
        env.setdefault("ALEMBIC_SKIP_GEN_API", "1")
        if database_url:
            env.setdefault("DATABASE_URL", database_url)
        try:
            cmd = [
                "alembic",
                "-c",
                str(alembic_ini),
                "upgrade",
                "head",
            ]
            env_overrides: dict[str, Optional[str]] = {}
            if "DATABASE_URL" in env:
                env_overrides["DATABASE_URL"] = _sanitize_url_for_log(
                    env["DATABASE_URL"]
                )
            if "ALEMBIC_SKIP_GEN_API" in env:
                env_overrides["ALEMBIC_SKIP_GEN_API"] = env["ALEMBIC_SKIP_GEN_API"]
            bt.logging.info(
                {
                    "db_init": {
                        "alembic_cli": {
                            "event": "starting",
                            "command": cmd,
                            "env_overrides": env_overrides,
                        }
                    }
                }
            )
            started_at = time.monotonic()
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                cwd=str(alembic_root),
                env=env,
            )
            elapsed = time.monotonic() - started_at
            bt.logging.info(
                {
                    "db_init": {
                        "alembic_cli": {
                            "event": "completed",
                            "returncode": result.returncode,
                            "elapsed_seconds": round(elapsed, 3),
                        }
                    }
                }
            )
            result.check_returncode()
            if result.stdout:
                bt.logging.info(
                    {
                        "db_init": {
                            "alembic_cli_stdout": result.stdout[-1000:],
                        }
                    }
                )
            after_revision = _get_current_revision(database_url)
            # Re-enable bittensor logger after alembic CLI fallback too
            import logging as std_logging
            std_logging.getLogger("bittensor").disabled = False
            bt.logging.info(
                {
                    "db_init": {
                        "alembic_cli": {
                            "event": "post_upgrade_revision",
                            "revision": after_revision,
                        }
                    }
                }
            )
        except subprocess.CalledProcessError as cli_err:
            bt.logging.error(
                {
                    "db_init": {
                        "alembic_cli_error": cli_err.stderr[-1000:]
                        if cli_err.stderr
                        else str(cli_err),
                        "returncode": cli_err.returncode,
                    }
                }
            )
            raise


