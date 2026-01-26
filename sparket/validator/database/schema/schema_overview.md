# Validator Schema Overview

The validator database models live in the `schema/` package. Splitting the code by concern keeps each file readable while still sharing a single SQLAlchemy `MetaData` for migrations.

## Directory Layout

- `schema/base.py` – shared `DeclarativeBase`, naming-convention `metadata`, and enum declarations (`market_kind`, `price_side`, `market_result`).
- `schema/reference.py` – static lookup tables (`sport`, `league`, `team`, `provider`, `miner`).
- `schema/events.py` – fixtures and trading context (`event`, `market`).
- `schema/outcomes.py` – settlement truth per market (`outcome`).
- `schema/provider.py` – external provider quote history and closing snapshots.
- `schema/miner.py` – miner submissions plus derived scoring tables (`submission_vs_close`, rolling stats, etc.).
- `schema/publication.py` – outbox/inbox tables supporting exactly-once publishing.
- `schema/render_validator_ddl.py` – helper that walks `metadata` and prints PostgreSQL DDL (convenient for draw.io diagrams).
- `schema/cleanup_rules.py` – retention helpers that age out historical provider quotes and miner submissions.

## Alembic Usage

Import `metadata` from `schema` in Alembic’s `env.py`:

```python
from schema import metadata
```

Because each module registers its tables against the shared `metadata`, autogenerate picks up every change automatically. If we add new domains later, just expose their tables in `schema/__init__.py`.

## Adding New Tables

1. Create a module beside the existing ones (or extend an existing domain module).
2. Import `Base`, any mixins, and enums from `schema.base`.
3. Define your models; declare surrogate keys explicitly.
4. Export them via `schema/__init__.py` so migrations, cleanup rules, and other callers can import from a single place.

This structure keeps most files under 500 LOC, reduces import churn, and makes it easy to reason about each slice of validator responsibility.
