# Validator Guide

This guide gets a Sparket validator from zero to production. It covers
requirements, setup, configuration, and day-two operations.

## What a validator does
A validator is the scoring engine of the subnet. It:
- Ingests provider data and builds ground truth snapshots.
- Accepts miner submissions and scores them.
- Aggregates scores into rolling metrics and final SkillScore.
- Emits weights back to the chain.

## Requirements
### Hardware
- 4 to 8 CPU cores
- 32 GB RAM
- 500 GB to 1 TB SSD
- Reliable 100 Mbps uplink

### Wallet and chain access
- Bittensor CLI installed
- A coldkey and hotkey created and registered
- Access to the subtensor endpoint for your target netuid

### Data provider subscription (required)
Validators must ingest the same provider data to stay in sync with other
validators. We use SportsDataIO for this subnet.

You will need a SportsDataIO plan that includes:
- Odds (line history and closing lines)
- Schedules
- Final scores / outcomes

Cost is roughly $600 per month for the plan we run.

If you do not want to pay for the provider plan, do not run your own
validator. Instead, child hotkey to the Sparket team validator and
delegate your stake there. This keeps the subnet consistent and avoids
validators drifting on mismatched data.

## Install

### Prerequisites
From a clean Ubuntu/Debian host, install system dependencies:
```bash
# Update packages
sudo apt update && sudo apt upgrade -y

# Install build essentials and git
sudo apt install -y build-essential git curl

# Install Node.js (required for pm2)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install pm2 globally
sudo npm install -g pm2

# Install Docker (for managed Postgres)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# Log out and back in for docker group to take effect
```

### Clone and setup
```bash
git clone https://github.com/sparketlabs/sparket-subnet.git
cd sparket-subnet
```

Install uv and Python, then sync dependencies:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart shell to get uv in PATH
uv python install 3.10
uv sync --dev
```

## Configuration
### 1) Environment file
Copy the example env file and edit it:
```
cp sparket/config/env.example .env
```

Required fields:
- `SPARKET_ROLE=validator`
- `SDIO_API_KEY=...`
- Database settings or `DATABASE_URL`
- Axon host and port

### 2) YAML config
Copy the example validator config:
```
cp sparket/config/sparket.example.yaml sparket/config/sparket.yaml
```

Update these sections:
```
role: validator

wallet:
  name: your-wallet
  hotkey: your-hotkey

subtensor:
  chain_endpoint: ws://your-subtensor:9945

chain:
  netuid: 2

database:
  host: 127.0.0.1
  port: 5435
  user: sparket
  name: sparket
  docker:
    enabled: true
```

Validator worker settings live under `validator`:
```
validator:
  scoring_worker_enabled: true
  scoring_worker_count: 2
  scoring_worker_fallback: true
```

### 3) Optional proxy URL
If you front your validator with a proxy or tunnel, set:
- `.env`: `SPARKET_API__PROXY_URL=https://your-proxy.example.com/axon`
- `sparket/config/sparket.yaml`:
```
api:
  proxy_url: https://your-proxy.example.com/axon
```

## First run (optional but recommended)
Running once in the foreground helps verify your setup before handing off to pm2:
```bash
uv run python sparket/entrypoints/validator.py
```

This will:
- Start Postgres via Docker if enabled
- Create the database if missing
- Run migrations and seed reference data

Stop it with Ctrl+C once you see the validator loop running and no errors.

> **Note:** This step is optional. PM2 runs the same entrypoint and will
> perform the bootstrap automatically. However, running interactively first
> makes it easier to spot configuration errors.

## Run in production
### PM2 (recommended)
```bash
# Start the validator
pm2 start ecosystem.config.js --only validator-local

# Watch logs to verify startup
pm2 logs validator-local

# Save process list so pm2 restarts on reboot
pm2 save

# Enable pm2 startup on boot
pm2 startup
# Follow the printed command (sudo env PATH=... pm2 startup ...)
```

Logs live in `sparket/logs/pm2`.

### Useful pm2 commands
```bash
pm2 status              # Check process status
pm2 restart validator-local
pm2 stop validator-local
pm2 delete validator-local  # Remove from pm2
```

### Systemd (optional)
`scripts/ops/setup_validator.sh` writes a systemd unit at
`scripts/systemd/sparket-validator.service`. Copy it to `/etc/systemd/system`
and enable it if you prefer systemd.

## Multi-worker scoring
The validator can offload scoring to worker processes. Increase
`validator.scoring_worker_count` to match your CPU. If workers become
unhealthy, `scoring_worker_fallback` keeps scoring in the main process.

## Upgrades
```bash
cd sparket-subnet
git pull
uv sync --dev
pm2 restart validator-local
pm2 logs validator-local  # verify clean restart
```

## Troubleshooting
- Missing provider data: check `SDIO_API_KEY` and your SportsDataIO plan.
- DB connection errors: verify host, port, and credentials.
- Miners cannot reach you: set `SPARKET_AXON__PORT` and open the port on the host.
