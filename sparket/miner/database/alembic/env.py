from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Ensure package root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from sparket.miner.database.schema import Base  # noqa: E402 - imports all models via __init__


config = context.config


def _get_database_url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("sqlalchemy.url must be set for miner migrations (set by upgrade_database).")
    return url


def _configured_section(database_url: str) -> dict[str, str]:
    section = config.get_section(config.config_ini_section, {}).copy()
    section["sqlalchemy.url"] = database_url
    return section


DATABASE_URL: str = _get_database_url()
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = _configured_section(DATABASE_URL)
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    def _run_sync_migrations(connection):
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()

    with connectable.connect() as connection:
        _run_sync_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

