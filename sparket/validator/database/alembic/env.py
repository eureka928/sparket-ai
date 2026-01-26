from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import bittensor as bt

from sqlalchemy import engine_from_config, pool
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
from sparket.validator.database.schema import metadata as target_metadata
from sparket.validator.database.schema.base import (
    market_kind_enum,
    market_result_enum,
    price_side_enum,
    score_component_enum,
    task_type_enum,
)
# --- Custom: API model generation on autogenerate ---
# ...


def get_database_url() -> str:
    # Prefer env var; fall back to alembic.ini
    env = os.getenv("DATABASE_URL") or os.getenv("SPARKET_DATABASE__URL")
    if env:
        return env
    try:
        url = config.get_main_option("sqlalchemy.url")
    except Exception:
        url = ""
    if url:
        return url
    raise RuntimeError("DATABASE_URL is not set for Alembic migrations.")


# Simple sanitization to avoid logging credentials
def _sanitize_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        netloc = host
        if parts.username:
            creds = parts.username
            if parts.password:
                creds += ":***"
            netloc = f"{creds}@{host}"
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return "***"


# --- Custom: API model generation on autogenerate ---

def _maybe_generate_api_models() -> None:
    try:
        xargs = context.get_x_argument(as_dictionary=True)
        if xargs.get("skip_gen_api") == "1":
            bt.logging.debug({"alembic": {"api_models": "skipped_flag"}})
            return
        if os.getenv("ALEMBIC_GEN_API") != "1":
            bt.logging.debug({"alembic": {"api_models": "skipped_default"}})
            return
        # Ensure project root in sys.path
        root = Path(__file__).resolve().parents[4]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # Import and run generator
        from scripts.generate_api_models_from_schema import main as gen_main  # type: ignore

        bt.logging.info({"alembic": {"api_models": "generating"}})
        gen_main()
        bt.logging.info({"alembic": {"api_models": "complete"}})
    except Exception:  # best-effort; don't block migrations
        bt.logging.warning({"alembic": {"api_models": "failed"}})
        pass


def process_revision_directives(context_, revision, directives):
    # Run after autogenerate finished composing directives
    _maybe_generate_api_models()


from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import ENUM as PGEnum

_ENUM_IMPORT_LINE = (
    "from sparket.validator.database.schema.base import "
    "market_kind_enum, market_result_enum, price_side_enum, "
    "score_component_enum, task_type_enum"
)
_ENUM_MAP: dict[str, str] = {
    "market_kind": "market_kind_enum",
    "market_result": "market_result_enum",
    "price_side": "price_side_enum",
    "score_component": "score_component_enum",
    "task_type": "task_type_enum",
}


def _render_item(type_, obj, autogen_context):
    if type_ == "type" and isinstance(obj, (SAEnum, PGEnum)):
        name = getattr(obj, "name", None)
        if name and name in _ENUM_MAP:
            bt.logging.info({"alembic": {"render_enum": name}})
            autogen_context.imports.add(_ENUM_IMPORT_LINE)
            return _ENUM_MAP[name]
    return False


def run_migrations_offline() -> None:
    url = get_database_url()
    bt.logging.info({
        "alembic": {
            "mode": "offline",
            "phase": "start",
            "url": _sanitize_url(url),
        }
    })
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        process_revision_directives=process_revision_directives,
        render_item=_render_item,
    )

    with context.begin_transaction():
        context.run_migrations()
    bt.logging.info({"alembic": {"mode": "offline", "phase": "complete"}})


def run_migrations_online() -> None:
    url = get_database_url()
    bt.logging.info({
        "alembic": {
            "mode": "online",
            "phase": "start",
            "url": _sanitize_url(url),
        }
    })

    if url.startswith("postgresql+asyncpg://"):
        connectable = create_async_engine(url, poolclass=pool.NullPool)

        def _run_migrations(sync_conn):
            context.configure(
                connection=sync_conn,
                target_metadata=target_metadata,
                compare_type=True,
                compare_server_default=True,
                process_revision_directives=process_revision_directives,
                render_item=_render_item,
            )
            with context.begin_transaction():
                context.run_migrations()

        async def do_run_migrations() -> None:
            bt.logging.info({"alembic": {"mode": "online", "phase": "async_connect"}})
            async with connectable.connect() as connection:
                await connection.run_sync(_run_migrations)
                bt.logging.info({"alembic": {"mode": "online", "phase": "async_run_complete"}})

        import asyncio

        asyncio.get_event_loop().run_until_complete(do_run_migrations())
        connectable.sync_engine.dispose()
        bt.logging.info({"alembic": {"mode": "online", "phase": "async_disposed"}})
    else:
        configuration = config.get_section(config.config_ini_section)
        configuration["sqlalchemy.url"] = url
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        def _run_sync_migrations(connection):
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                compare_server_default=True,
                process_revision_directives=process_revision_directives,
                render_item=_render_item,
            )
            with context.begin_transaction():
                context.run_migrations()

        with connectable.connect() as connection:
            _run_sync_migrations(connection)
        bt.logging.info({"alembic": {"mode": "online", "phase": "sync_complete"}})


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()


