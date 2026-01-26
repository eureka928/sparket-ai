# Validator Guide

This guide gets a Sparket validator from zero to production. It covers
requirements, setup, configuration, and day-two operations.

## What a validator does
A validator is the scoring engine of the subnet. It:
- Ingests provider data and builds ground truth snapshots.
- Accepts miner submissions and scores them.
- Aggregates scores into rolling metrics and final SkillScore.
- Emits weights back to the chain.

## Minimum Requirements
- 4 to 8 CPU cores
- 32 GB RAM
- 500 GB to 1 TB SSD
- Reliable 100 Mbps uplink (minimum)
- Ubuntu 24.04 LTS (or 22.04 LTS)
- Python 3.10
- Docker


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

## Sportsdata.io Plan Details
You will need to contact sportsdata.io and sign up for a plan which includes these specific products and leagues:
- Competition Feeds:
  - Standings, Rankings & Brackets
  - Teams, Players & Rosters
  - Venues & Officials
  - Utility Endpoints
- Event Feeds
  - Schedules & Game Day Info
- Betting Feeds
  - Game Lines
    - Pre-Game Lines
    - Pre-Game Lines PLus

Sports which you must have these products for:
NFL, MLB, NBA, NHL, Soccer

Please contact our team for help if you have issues with consensus or need clarity on which data products to purchase.


## Install
From a clean host:
```
git clone https://github.com/sparketlabs/sparket-subnet.git
cd sparket-subnet
```

Install uv and Python, then sync dependencies:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
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
  netuid: 57

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

### 3) Optional proxy URL (IP Privacy)
If you want to hide your validator's IP from the blockchain, use a proxy or tunnel
(e.g., Cloudflare Tunnel, ngrok). When `proxy_url` is configured:

1. **Your real IP is NOT registered to the chain** - the `axon.serve()` call is skipped
2. **Miners receive the proxy URL** via periodic `CONNECTION_INFO_PUSH` broadcasts
3. **The axon still accepts connections** on the local port for the proxy to forward to

Set the proxy URL via environment variable or YAML:
- `.env`: `SPARKET_API__PROXY_URL=https://your-proxy.example.com`
- `sparket/config/sparket.yaml`:
```yaml
api:
  proxy_url: https://your-proxy.example.com
```

**Example with Cloudflare Tunnel:**
```bash
# 1. Install cloudflared
# 2. Create tunnel: cloudflared tunnel create sparket-validator
# 3. Route: cloudflared tunnel route dns sparket-validator validator.yourdomain.com
# 4. Run tunnel: cloudflared tunnel run --url http://localhost:8093 sparket-validator
# 5. Set SPARKET_API__PROXY_URL=https://validator.yourdomain.com
```

When the validator starts with proxy mode, you'll see:
```
axon_proxy_mode: {enabled: true, proxy_url: "...", chain_registration: "skipped"}
```

## First run
Run once in the foreground to bootstrap:
```
python sparket/entrypoints/validator.py
```

This will:
- Start Postgres via Docker if enabled
- Create the database if missing
- Run migrations and seed reference data

Stop it with Ctrl+C once you see the validator loop running.

## Run in production
### PM2 (recommended)
```
pm2 start ecosystem.config.js --only validator-local
pm2 logs validator-local
pm2 save
```

Logs live in `sparket/logs/pm2`.

### Systemd (optional)
`scripts/ops/setup_validator.sh` writes a systemd unit at
`scripts/systemd/sparket-validator.service`. Copy it to `/etc/systemd/system`
and enable it if you prefer systemd.

## Multi-worker scoring
The validator can offload scoring to worker processes. Increase
`validator.scoring_worker_count` to match your CPU and RAM. Please reserve 2-4 cores for OS operations and the main application loop, so if you have a 12 core machine, we recommend ~6-8 workers. Future versions of this feature will include auto-optimization of worker count based on configurable resource limits, but for now you will need to manually adjust it.
 
 If workers become
unhealthy, `scoring_worker_fallback` keeps scoring in the main process.

## Upgrades
```
git pull
uv sync --dev
pm2 restart validator-local
```

## Troubleshooting
- Missing provider data: check `SDIO_API_KEY` and your SportsDataIO plan.
- DB connection errors: verify host, port, and credentials.
- Miners cannot reach you: set `SPARKET_AXON__PORT` and open the port on the host.
