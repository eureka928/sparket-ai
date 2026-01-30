# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sparket Subnet is a Bittensor subnet that rewards miners for contributing valuable odds and outcome data for sports markets. The system has two main roles:
- **Validators**: Ingest provider data, score miner submissions, and set chain weights
- **Miners**: Submit odds predictions and outcomes for sporting events

## Common Commands

```bash
# Setup environment
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync --dev
source .venv/bin/activate

# Run tests
pytest                           # All tests
pytest -m "not slow"             # Skip slow tests
pytest -m "not e2e"              # Skip e2e tests
pytest -m "not integration"      # Skip integration tests
pytest tests/path/to/test.py    # Single file
pytest tests/path/to/test.py::test_name  # Single test

# Coverage
uv tool run pytest --cov=sparket tests/

# Database migrations
sparket-alembic upgrade head     # Run migrations
sparket-alembic downgrade -1     # Rollback one migration

# PM2 process management
pm2 start ecosystem.config.js
pm2 logs validator-local
pm2 stop ecosystem.config.js
```

## Architecture

### Core Modules (`sparket/`)

| Module | Purpose |
|--------|---------|
| `entrypoints/` | `validator.py` and `miner.py` entry points |
| `validator/` | Scoring engine, provider ingestion, weight setting |
| `miner/base/` | Template miner implementation |
| `miner/custom/` | Example miner with Elo models + calibration |
| `protocol/` | Synapse definitions and serialization |
| `providers/` | External data sources (SportsDataIO) |
| `config/` | Pydantic settings, YAML and env loading |
| `shared/` | Utilities, logging, probability math |

### Validator Scoring Pipeline

```
Provider Data Ingest → Build Closing Lines → Score Submissions (CLV, CLE, MES, Brier, LogLoss, PSS)
→ Time Decay & Shrinkage → Compute SkillScore → Normalize & Set Weights
```

### Database Schema (`sparket/validator/database/schema/`)

- `reference.py` — Static lookups (sport, league, team, provider, miner)
- `events.py` — Fixtures and markets
- `outcomes.py` — Settlement ground truth
- `provider.py` — Provider quote history and closing snapshots
- `miner.py` — Miner submissions and scoring tables
- `publication.py` — Outbox/inbox for exactly-once publishing

## Configuration

Configuration is loaded with precedence: environment variables > YAML file > defaults.

- YAML config: `sparket/config/sparket.yaml` (copy from `sparket.example.yaml`)
- Environment: `.env` (copy from `sparket/config/env.example`)

Key settings:
- `SPARKET_ROLE` — validator or miner
- `SDIO_API_KEY` — SportsDataIO API key (required for validators)
- `DATABASE_URL` — PostgreSQL connection string

## Key Dependencies

- **bittensor** — Blockchain integration
- **starlette** — Async web framework
- **sqlalchemy** + **alembic** — ORM and migrations
- **asyncpg** — PostgreSQL async driver
- **pydantic** — Type validation and settings

## Documentation

- `docs/validator.md` — Validator setup and operations
- `docs/miner.md` — Miner guide and communication flow
- `docs/im.md` — Incentive mechanism and scoring equations
