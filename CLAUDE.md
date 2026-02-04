# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sparket Subnet is a Bittensor subnet that rewards miners for contributing valuable odds and outcome data for sports markets. Validators ingest provider data, score miner submissions, and set chain weights. Miners submit odds predictions and outcomes for sporting events.

## Common Commands

```bash
# Setup
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync --dev
source .venv/bin/activate

# Tests
pytest                                        # All tests
pytest -m "not slow"                          # Skip slow tests
pytest -m "not e2e"                           # Skip e2e tests
pytest -m "not integration"                   # Skip integration tests
pytest tests/path/to/test.py                  # Single file
pytest tests/path/to/test.py::TestClass::test_name  # Single test
pytest --cov=sparket tests/                   # With coverage

# Database migrations (validator only)
sparket-alembic upgrade head
sparket-alembic downgrade -1

# PM2 process management
pm2 start ecosystem.config.js
pm2 logs validator-local
pm2 stop ecosystem.config.js
```

## Architecture

### Miner System

The miner entrypoint (`sparket/entrypoints/miner.py`) creates a `Miner(BaseMinerNeuron)` which handles Bittensor integration (axon, wallet, metagraph). Two optional miner engines run independently as background asyncio loops:

- **Base Miner** (`sparket/miner/base/`): ESPN stats → team strength → naive odds. Enabled by default (`SPARKET_BASE_MINER__ENABLED`).
- **Custom Miner** (`sparket/miner/custom/`): Ensemble model (Elo 50% + Market 35% + Poisson 15%) with isotonic calibration. Opt-in (`SPARKET_CUSTOM_MINER__ENABLED=true`).

Both engines share the same `ValidatorClient` and `GameDataSync` injected from the entrypoint, and both run two background loops: `_odds_loop()` (generate + batch-submit odds) and `_outcome_loop()` (detect finished games + submit outcomes).

**Communication flow**: Validators push `CONNECTION_INFO_PUSH` synapses containing `{host, port, url, token}`. Miners store this and include the token in all subsequent submissions via a `get_token()` callback.

### Validator System

The validator (`sparket/entrypoints/validator.py`) orchestrates:

1. **Provider Ingestion** — Background task polls SportsDataIO API (~60s), writes events/markets/quotes to PostgreSQL, detects outcomes.
2. **Main Loop** (~12s/step):
   - Sync metagraph (every 5 steps)
   - Broadcast connection info + token to miners (every 5 min)
   - Run scoring pipeline (every 25 steps): `rolling_aggregates` → `calibration_sharpness` → `originality_lead_lag` → `skill_score` with time decay and shrinkage
   - Set chain weights from normalized scores
3. **Axon Handlers** (`sparket/validator/handlers/`): Ingest miner odds/outcome pushes, serve game data. Security middleware (injected before axon start) rejects spam early via HTTP 429/403.

### Scoring Pipeline

```
Provider Data → Closing Lines → Score Submissions (CLV, CLE, MES, Brier, LogLoss, PSS)
→ Time Decay & Shrinkage → SkillScore → Normalize → Set Weights
```

Scoring dimensions: **EconDim** (50%, beat closing lines), **InfoDim** (30%, originality + lead market), **ForecastDim + SkillDim** (20%, outcome accuracy). Scoring jobs run in a multiprocess worker pool (`ScoringWorkerManager`).

### Database

- **Miner**: SQLite via aiosqlite (`sparket/data/{prod|test}/miner.db`). Simple `DBM.read()`/`DBM.write()` helpers. Stores validator endpoints and game cache.
- **Validator**: PostgreSQL via asyncpg. `DBM.get_manager()` singleton (pool_size=50) for main loop, `DBM.create_worker()` (pool_size=10) for scoring processes. Repository pattern in `sparket/validator/database/resolver.py`. Schema models in `sparket/validator/database/schema/`.

### Protocol

Synapse types (`sparket/protocol/protocol.py`): `ODDS_PUSH`, `OUTCOME_PUSH`, `GAME_DATA_REQUEST` (miner→validator), `CONNECTION_INFO_PUSH` (validator→miner). `ValidatorClient` (`sparket/miner/client.py`) handles submission with exponential backoff on 429/not_ready responses.

## Configuration

Precedence: **environment variables > YAML > defaults**.

- **Root config** (`sparket/config/core.py`): YAML search order — `SPARKET_CONFIG_FILE` env var → `./sparket.yaml` → `./config/sparket.yaml` → `./sparket/config/sparket.yaml`. Copy from `sparket.example.yaml`.
- **Miner config** (`sparket/miner/config/config.py`): Combines root `Settings` + YAML `miner:` section.
- **Base/Custom Miner configs**: Pure env-based, no YAML. Prefix `SPARKET_BASE_MINER__*` or `SPARKET_CUSTOM_MINER__*` (e.g., `SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS`).
- **Test mode**: `SPARKET_TEST_MODE=true` switches data directory to `sparket/data/test/`.

Key env vars: `SPARKET_ROLE`, `SDIO_API_KEY` (validators), `DATABASE_URL` (validators), `SPARKET_WALLET__NAME`, `SPARKET_WALLET__HOTKEY`, `SPARKET_NETUID`.

## Key Dependencies

- **bittensor** (>=10.0.0) — Chain integration, axon/dendrite
- **sqlalchemy** (2.0) + **alembic** — ORM and migrations
- **asyncpg** / **aiosqlite** — Async DB drivers (PostgreSQL / SQLite)
- **starlette** — Validator HTTP handlers
- **pydantic** + **pydantic-settings** — Config and validation
- **httpx** — Async HTTP client

## Documentation

- `docs/validator.md` — Validator setup and operations
- `docs/miner.md` — Miner guide and communication flow
- `docs/im.md` — Incentive mechanism and scoring equations
