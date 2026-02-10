# Sparket Subnet

Sparket is a company focused on building a democratized market where
participants contribute data to the markets they wager in. The Sparket subnet
(sparket.ai) is a Bittensor subnet that rewards miners for contributing valuable odds+outcome data. The subnet outputs are targeted to support product development and expansion of the Sparket ecosystem.
The Subnet beta supports two tasks—Odds Origination and Outcome
Verification—with real‑money wagering planned for a later phase.

## Quickstart
Install the Python toolchain and sync dependencies:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync --dev
```

Notes:
- `uv sync` reads `pyproject.toml` and `uv.lock` to create a reproducible environment.
- Activate the venv with `source .venv/bin/activate`.
- Prefer `uv tool run <cmd>` for one-off binaries from dependencies.

## Configuration
Sparket uses typed configuration with defaults, optional YAML, and env overrides.
- Example YAML: `sparket/config/sparket.example.yaml` → `sparket/config/sparket.yaml`
- Example env: `sparket/config/env.example` → `.env`
- Env overrides take precedence, e.g.:
  - `SPARKET_API__IP_ALLOWLIST="127.0.0.1/32,192.168.0.0/16"`
  - `DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/sparket"`

See the validator and miner guides for full setup details.

## Documentation
- Docs index: [`docs/README.md`](docs/README.md)
- Validator guide: [`docs/validator.md`](docs/validator.md) (includes auditor mode setup)
- Miner guide: [`docs/miner.md`](docs/miner.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md) (primary + auditor model)
- Incentive mechanism: [`docs/im.md`](docs/im.md)
- mdbook summary: [`docs/SUMMARY.md`](docs/SUMMARY.md)

Build and serve locally (optional):
```
mdbook serve docs
```

