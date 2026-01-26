"""Database bootstrap helpers (role creation and database provisioning)."""

from __future__ import annotations

import asyncio
import re
from typing import Optional

import bittensor as bt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from .utils import _sanitize_url_for_log

__all__ = [
    "_ensure_database_exists",
    "_bootstrap_role_and_database",
]


async def _ensure_database_exists(database_url: str, db_name: Optional[str]) -> None:
    """Ensure target database exists, creating it if necessary."""
    bt.logging.info({"db_init": {"step": "ensure_database_exists", "db_name": db_name}})

    if not database_url.startswith("postgresql+asyncpg://"):
        bt.logging.warning(
            {
                "db_init": {
                    "skip": "invalid_url_scheme",
                    "url_redacted": _sanitize_url_for_log(database_url),
                }
            }
        )
        return

    conn_url = database_url
    try:
        head, tail = database_url.rsplit("/", 1)
        if head and tail:
            conn_url = head + "/postgres"
            bt.logging.info(
                {
                    "db_init": {
                        "using_postgres_db": True,
                        "conn_url_redacted": _sanitize_url_for_log(conn_url),
                    }
                }
            )
    except ValueError:
        bt.logging.warning({"db_init": {"url_parse_warning": "could_not_split_url"}})

    max_retries = 5
    retry_delay = 2.0
    last_error: Exception | None = None

    for attempt in range(max_retries):
        engine = None
        try:
            bt.logging.info(
                {
                    "db_init": {
                        "connection_attempt": attempt + 1,
                        "max_retries": max_retries,
                    }
                }
            )
            engine = create_async_engine(
                conn_url,
                pool_size=1,
                max_overflow=0,
                future=True,
            )
            async with engine.begin() as conn:
                if db_name:
                    if not re.match(r"^[A-Za-z0-9_]+$", db_name):
                        raise ValueError(
                            "Invalid database_name; only letters, numbers, and underscore allowed."
                        )
                    exists_query = text(
                        "select 1 from pg_database where datname = :name limit 1"
                    )
                    res = await conn.execute(exists_query, {"name": db_name})
                    if not res.scalar():
                        bt.logging.info({"db_init": {"creating_database": db_name}})
                        await conn.execute(text(f'create database "{db_name}"'))
                        bt.logging.info({"db_init": {"database_created": db_name}})
                    else:
                        bt.logging.info({"db_init": {"database_exists": db_name}})
                else:
                    bt.logging.info({"db_init": {"database_exists": db_name}})

                bt.logging.info({"db_init": {"ensure_database_exists_complete": True}})
                return
        except Exception as exc:
            last_error = exc
            bt.logging.warning(
                {
                    "db_init": {
                        "connection_failed": True,
                        "attempt": attempt + 1,
                        "error": str(exc),
                    }
                }
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                bt.logging.error(
                    {
                        "db_init": {
                            "connection_failed_after_retries": True,
                            "error": str(last_error),
                        }
                    }
                )
                raise
        finally:
            if engine is not None:
                await engine.dispose()


async def _bootstrap_role_and_database(
    *,
    admin_url: Optional[str],
    user: Optional[str],
    password: Optional[str],
    db_name: Optional[str],
) -> None:
    if not admin_url or not user or not db_name:
        return
    engine = create_async_engine(admin_url, pool_size=1, max_overflow=0, future=True)
    try:
        async with engine.begin() as conn:
            stmts = []
            if password:
                stmts.append(
                    text(
                        """
                        DO $$
                        DECLARE uname text := :uname; upwd text := :upwd;
                        BEGIN
                          IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = uname) THEN
                            EXECUTE 'CREATE ROLE ' || quote_ident(uname) || ' LOGIN PASSWORD ' || quote_literal(upwd);
                          ELSE
                            EXECUTE 'ALTER ROLE ' || quote_ident(uname) || ' LOGIN PASSWORD ' || quote_literal(upwd);
                          END IF;
                        END $$;
                        """
                    )
                )
            else:
                stmts.append(
                    text(
                        """
                        DO $$
                        DECLARE uname text := :uname;
                        BEGIN
                          IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = uname) THEN
                            EXECUTE 'CREATE ROLE ' || quote_ident(uname) || ' LOGIN';
                          END IF;
                        END $$;
                        """
                    )
                )
            for stmt in stmts:
                await conn.execute(stmt, {"uname": user, "upwd": password})

            await conn.execute(
                text(
                    """
                    DO $$
                    DECLARE dname text := :dname; uname text := :uname;
                    BEGIN
                      IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = dname) THEN
                        EXECUTE 'CREATE DATABASE ' || quote_ident(dname) || ' OWNER ' || quote_ident(uname);
                      END IF;
                    END $$;
                    """
                ),
                {"dname": db_name, "uname": user},
            )
    finally:
        await engine.dispose()


