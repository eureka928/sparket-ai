"""
Database manager for the miner.

Miner requirements are much lighter than validator, so we can use sqlite instead of postgres, and simpler configuration.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.engine import Engine, Result
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql.elements import TextClause, ClauseElement

from sparket.miner.config import Config


# Enable WAL mode for aiosqlite connections
# This is a per-connection pragma, so we need to set it on each connect.
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any):
    # Only run this for sqlite connections
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()


def build_sqlite_url(path: str) -> str:
    return f"sqlite+aiosqlite:///{os.path.abspath(path)}"


class DBM:
    def __init__(self, config: Config, neuron_path: str | None = None):
        self.config = config
        # Use project-relative data directory based on test mode
        test_mode = getattr(getattr(config, "core", None), "test_mode", False)
        db_path = config.miner.database_path(test_mode)
        self.db_path = db_path
        db_url = build_sqlite_url(db_path)

        self.engine: AsyncEngine = create_async_engine(
            db_url,
            echo=False,  # Keep logs clean
            future=True,
        )

        self.session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    @asynccontextmanager
    async def session(self) -> AsyncSession:
        async with self.session_maker() as session:
            yield session

    async def read(self, query: Any, params: dict | None = None) -> list[Any]:
        """Execute a read-only statement and return all rows."""
        if isinstance(query, str):
            raise TypeError("Raw SQL strings are disallowed. Use sqlalchemy.text().")
        if not isinstance(query, (TextClause, ClauseElement)):
            raise TypeError("Query must be a SQLAlchemy TextClause or ClauseElement.")

        async with self.session() as session:
            result: Result = await session.execute(query, params or {})
            return result.mappings().all()

    async def write(self, query: Any, params: dict | None = None) -> int:
        """Execute a write statement inside a transaction and return row count."""
        if isinstance(query, str):
            raise TypeError("Raw SQL strings are disallowed. Use sqlalchemy.text().")
        if not isinstance(query, (TextClause, ClauseElement)):
            raise TypeError("Query must be a SQLAlchemy TextClause or ClauseElement.")
        if not params:
            raise ValueError("Parameterized writes are required. Provide a params mapping.")

        async with self.session() as session:
            async with session.begin():
                result: Result = await session.execute(query, params)
                return result.rowcount or 0

    async def dispose(self) -> None:
        await self.engine.dispose()