# Miner Guide

This guide covers setup and usage for a Sparket miner. Miners generate
odds and outcomes and submit them to validators for scoring.

## What a miner does
A miner:
- Connects to the subnet and exposes an axon.
- Produces odds for active markets.
- Submits outcomes after events are settled.
- Builds a track record that feeds SkillScore and emissions.

## Before you begin
You do not need to be a developer, but you should be able to run basic
commands in a terminal and edit a text file.

You will need:
- A Linux server or VPS (Ubuntu 22.04 recommended).
- Python 3.10+ and the Bittensor CLI.
- A funded coldkey and a registered hotkey.
- A public IP address.
- An open inbound TCP port for your miner axon (default 8094).

The axon port must be reachable from the internet. A common setup issue
is a firewall or cloud security group blocking inbound traffic.

## Install the miner software
Step 1: clone the repository:
```
git clone https://github.com/sparketlabs/sparket-subnet.git
cd sparket-subnet
```

Step 2: install uv (the Python toolchain) and make sure Python 3.10 is available:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
```

Step 3: install all dependencies:
```
uv sync --dev
```

If this is your first time using uv, it will also create a `.venv/` virtual
environment in the repository.

## Wallet and registration
Miners need a coldkey (funds) and a hotkey (miner identity). If you
already have a wallet, you can skip creation and only register.

Create a wallet:
```
btcli wallet create --wallet.name miner-wallet --wallet.hotkey default
```

Register your hotkey on the subnet:
```
btcli subnet register \
  --wallet.name miner-wallet \
  --wallet.hotkey default \
  --netuid 2 \
  --subtensor.chain_endpoint ws://your-subtensor:9945
```

## Configuration
### 1) Environment file (recommended)
Copy the example and edit it with your values:
```
cp sparket/config/env.example .env
```

At minimum, set:
- `SPARKET_ROLE=miner`
- `SPARKET_AXON__HOST=0.0.0.0` (listen on all interfaces)
- `SPARKET_AXON__PORT=8094` (your public axon port)

If your server is behind NAT or a load balancer, also set:
- `SPARKET_AXON__EXTERNAL_IP=<public-ip>`
- `SPARKET_AXON__EXTERNAL_PORT=<public-port>`

### 2) Miner YAML
The repo ships a default miner YAML at `sparket/config/miner.yaml`.
For your own settings, copy it and point to the new path:
```
cp sparket/config/miner.yaml sparket/config/miner.local.yaml
```

Then set:
```
export SPARKET_MINER_CONFIG_FILE="$(pwd)/sparket/config/miner.local.yaml"
```

Key miner settings:
```
miner:
  markets: [123, 456]
  events: ["event-id-1", "event-id-2"]
  cadence:
    odds_seconds: 60
    outcomes_seconds: 120
  rate:
    global_per_minute: 60
    per_market_per_minute: 6
  retry:
    max_attempts: 3
  idempotency:
    bucket_seconds: 60
  allow_connection_info_from_unpermitted_validators: false
  endpoint_override:
    url: null
    host: null
    port: null
```

Use `endpoint_override` if you want to pin the miner to a specific
validator endpoint instead of accepting the announced one.

## Open your axon port (critical)
Validators must be able to reach your miner on the axon port. If this
port is blocked, your miner will not receive traffic and will not score.

### On Ubuntu with UFW
Allow inbound TCP on the axon port:
```
sudo ufw allow 8094/tcp
sudo ufw status
```

### On cloud providers
Most VPS providers use a security group or firewall rule. Add an inbound
rule for TCP port 8094 to your instance.

### Test the port
From another machine, check the port:
```
nc -vz <public-ip> 8094
```
If this fails, the port is still blocked or the miner is not listening.

## Run the miner
Activate the virtual environment first:
```
source .venv/bin/activate
```

Then start the miner:
```
python sparket/entrypoints/miner.py
```

### PM2 (optional)
```
pm2 start ecosystem.miner.config.js
pm2 logs miner-local
pm2 save
```

Logs live in `sparket/logs/pm2`.

## Verify it is reachable
Look for logs showing the axon is started. You can also confirm the port
is listening on the server:
```
ss -lntp | grep 8094
```

If your miner is running but validators cannot connect, re-check:
- The firewall rule or cloud security group
- The public IP and port
- Any NAT or port forwarding rules

## How submissions are produced
The miner uses `MinerService` to submit odds and outcomes on a cadence.
It reads market IDs and event IDs from `miner.markets` and `miner.events`.
Replace that pipeline with your own model or data source if desired.

## Base miner (reference implementation)
The repository includes a base miner that uses free or low-cost sources and
simple heuristics to generate odds. It is intentionally lightweight and
serves as a reference implementation, not a competitive strategy. Expect it
to perform poorly in the scoring system compared with miners that ingest
better data and model the market more accurately.

The base miner is **enabled by default** and will start automatically when
you run the miner. If you want to replace it with your own service, disable it:
```
export SPARKET_BASE_MINER__ENABLED=false
```

Additional base miner settings live in `sparket/miner/base/config.py`.

## Building a competitive miner
Competitive submissions require better data and stronger models than the
reference miner. The highest leverage improvements are:

### 1) Data quality and coverage
Data quality is the primary edge. The scoring system rewards early, accurate,
and original probabilities, so higher-resolution inputs matter.
- **Finer granularity**: line history, player availability, injuries, travel,
  lineup changes, weather, venue effects, and market microstructure.
- **Faster ingestion**: lower latency to new information improves time-to-close
  advantage and lead-lag scores.
- **Coverage depth**: more leagues and market types increase sample size and
  stability of rolling metrics.

Instructional tip: start by logging raw data with timestamps and source IDs.
This makes debugging and calibration much easier when performance drops.

### 2) Classical probability modeling
Solid statistical baselines often outperform naive heuristics and are easier
to debug and calibrate.
- **Elo‑style ratings** with home‑field and rest adjustments.
- **Poisson or bivariate Poisson** for scoreline‑driven markets.
- **Logistic regression / GLMs** for win probabilities with structured covariates.
- **Bayesian updating** to incorporate late news while preserving calibration.

### 3) Machine learning approaches
ML can capture nonlinear effects and interactions, but must remain calibrated.
- **Gradient‑boosted trees** for structured tabular features (injuries, rest, travel).
- **Sequence models** for time‑ordered signals (line moves, recent form).
- **Ensembling** multiple model families to reduce variance and improve stability.
- **Calibration layers** (isotonic regression, Platt scaling) to keep probabilities honest.

Instructional tip: evaluate models with proper scoring rules (Brier, log-loss)
and calibration plots, not just accuracy.

### 4) Market-aware adjustments
The system compares you to closing lines, so understand market structure.
- **De‑vigging** and implied probability normalization.
- **Consensus vs sharp books** weighting.
- **Outlier and stale line filtering** to avoid spurious signals.

### 5) Operational reliability
Consistency matters for rolling metrics and shrinkage.
- Avoid gaps in submission cadence.
- Backfill missed markets only when you have reliable data.
- Track submission timing relative to event start.

## Resources and references
- Proper scoring rules: https://en.wikipedia.org/wiki/Proper_scoring_rule
- Calibration overview (scikit-learn): https://scikit-learn.org/stable/modules/calibration.html
- Elo rating system: https://en.wikipedia.org/wiki/Elo_rating_system
- Poisson models for scores: https://en.wikipedia.org/wiki/Poisson_distribution
- Forecasting fundamentals: https://otexts.com/fpp3/

## Replacing or extending the miner
If you are building a competitive miner, plan to replace the base miner
with your own service. A simple, safe path is:
1. Disable the base miner (`SPARKET_BASE_MINER__ENABLED=false`).
2. Implement a new service that:
   - Collects your data feeds
   - Produces calibrated probabilities
   - Submits odds/outcomes on a cadence
3. Wire your service into the miner entrypoint or run it as a separate process.

Starting points in this repo:
- `sparket/miner/service.py`: the default cadence-based submission loop
- `sparket/miner/base/runner.py`: the base miner reference implementation
- `sparket/miner/utils/payloads.py`: submission payload builders
- `sparket/miner/client.py`: validator submission client

See `docs/miner_custom_service_example.md` for a minimal end-to-end example.

The key outcome is calibrated probabilities that beat the market early.

## Upgrades
```
git pull
uv sync --dev
pm2 restart miner-local
```
