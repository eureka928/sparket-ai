# Sparket Custom Miner Guide

A comprehensive guide to running the Sparket Custom Miner with the ensemble prediction model.

## Overview

The custom miner uses an ensemble model that combines three prediction engines:

| Engine | Default Weight | Description |
|--------|---------------|-------------|
| **Market** | 55% | Sharp book consensus (Pinnacle weighted 3x) |
| **Elo** | 35% | Team strength ratings with home advantage |
| **Poisson** | 10% | Scoring distribution for totals |

The ensemble uses confidence-based weighting, adjusting weights based on each model's sharpness and agreement.

## Prerequisites

1. **Bittensor wallet** with registered hotkey on Sparket subnet
2. **Python 3.10+** with dependencies installed
3. **The-Odds-API key** (recommended) - Get free at https://the-odds-api.com/
4. **PM2** (optional) - For production process management

## Quick Start

### 1. Install Dependencies

```bash
# Clone and setup
cd sparket-ai
uv sync

# Or with pip
pip install -e .
```

### 2. Configure Environment

```bash
# Copy template
cp .env.custom.example .env.custom

# Edit with your settings
nano .env.custom
```

**Required settings in `.env.custom`:**
```bash
BT_WALLET_NAME=your_wallet
BT_WALLET_HOTKEY=your_hotkey
SPARKET_NETUID=1
SPARKET_CUSTOM_MINER__ODDS_API_KEY=your_api_key
```

### 3. Run the Miner

**Option A: PM2 (recommended for production)**

Uses the main miner entrypoint with axon serving + custom ensemble model:
```bash
pm2 start ecosystem.custom.config.js
pm2 logs custom-miner
```

**Option B: Direct execution (with axon)**
```bash
SPARKET_CUSTOM_MINER__ENABLED=true SPARKET_BASE_MINER__ENABLED=false \
  uv run python sparket/entrypoints/miner.py
```

**Option C: Standalone (no axon, for testing only)**
```bash
uv run python -m sparket.miner.custom.runner
```
Note: Option C does NOT serve an axon, so validators cannot reach this miner on the network. Use for local testing only.

## Configuration Reference

### Wallet & Network

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARKET_WALLET__NAME` | `default` | Bittensor wallet name |
| `SPARKET_WALLET__HOTKEY` | `default` | Hotkey name |
| `SPARKET_NETUID` | `1` | Subnet network UID |

### Axon (Network Communication)

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARKET_AXON__HOST` | `0.0.0.0` | Axon listen address (all interfaces) |
| `SPARKET_AXON__PORT` | `8094` | Axon listen port |
| `SPARKET_AXON__EXTERNAL_IP` | (auto) | External IP if behind NAT |
| `SPARKET_AXON__EXTERNAL_PORT` | (auto) | External port if behind NAT |

### Model Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARKET_CUSTOM_MINER__VIG` | `0.045` | Vig/margin for odds (4.5%) |
| `SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__ELO` | `0.35` | Elo model weight |
| `SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__MARKET` | `0.55` | Market consensus weight |
| `SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__POISSON` | `0.10` | Poisson model weight |

### Timing

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARKET_CUSTOM_MINER__TIMING__EARLY_SUBMISSION_DAYS` | `7` | Days before event to submit |
| `SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS` | `21600` | Odds refresh interval (6h) |
| `SPARKET_CUSTOM_MINER__OUTCOME_CHECK_SECONDS` | `300` | Outcome check interval (5m) |

### Calibration

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARKET_CUSTOM_MINER__CALIBRATION__ENABLED` | `true` | Enable isotonic calibration |
| `SPARKET_CUSTOM_MINER__CALIBRATION__MIN_SAMPLES` | `100` | Samples before calibration activates |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARKET_CUSTOM_MINER__RATE_LIMIT_PER_MINUTE` | `60` | Global submissions per minute |
| `SPARKET_CUSTOM_MINER__PER_MARKET_LIMIT_PER_MINUTE` | `10` | Per-market submissions per minute |

## Testing

### Live Test (No Submission)

Test against real markets without submitting predictions:

```bash
# Test NFL markets
uv run python -m sparket.miner.custom.live_test --sport NFL -v

# Test NBA markets
uv run python -m sparket.miner.custom.live_test --sport NBA -v

# Test with more events
uv run python -m sparket.miner.custom.live_test --sport NFL --max-events 20
```

**Example output:**
```
1. Dallas Cowboys @ Philadelphia Eagles
   Game: Sun Jan 26, 4:25 PM (48.3h)
   Our prediction:    Philadelphia Eagles 62.1% | Dallas Cowboys 37.9%
   Our odds:          Philadelphia Eagles 1.55 (-164) | Dallas Cowboys 2.53 (+153)
   Market consensus:  Philadelphia Eagles 58.2% | Dallas Cowboys 41.8%
   Edge vs market:    +3.9% ✓
   Confidence:        0.78
   Dominant model:    market
   Components:
     - elo: 65.3% (weight: 0.38)
     - market: 58.2% (weight: 0.62)
```

### Backtest

Run historical backtests to evaluate model performance:

```bash
# Basic backtest
uv run python -m sparket.miner.custom.backtest --games 100

# With Elo seeding and warmup
uv run python -m sparket.miner.custom.backtest --games 200 --seed-elo --warmup 50

# Reproducible results with fixed seed
uv run python -m sparket.miner.custom.backtest --games 100 --seed-elo --warmup 50 --seed 42
```

**Key metrics:**
- **Brier Score**: Lower is better (0.25 = perfect calibration baseline)
- **CLV (Closing Line Value)**: Positive = beating the market
- **Calibration**: 1.0 = perfectly calibrated probabilities

## PM2 Management

### Start/Stop/Restart

```bash
# Start miner
pm2 start ecosystem.custom.config.js

# View logs
pm2 logs custom-miner

# Real-time monitoring
pm2 monit

# Check status
pm2 status

# Stop miner
pm2 stop custom-miner

# Restart miner
pm2 restart custom-miner

# Remove from PM2
pm2 delete custom-miner
```

### Log Files

Logs are stored in the `logs/` directory:
- `custom-miner-out.log` - Standard output
- `custom-miner-error.log` - Error output
- `custom-miner-combined.log` - Combined logs

### Auto-Startup

To start miner automatically on system boot:

```bash
pm2 startup
pm2 save
```

## Scoring System

The Sparket subnet scores miners on four dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **EconDim** | 50% | Economic performance (P&L, ROI) |
| **InfoDim** | 30% | Information value (SOS, LeadRatio) |
| **ForecastDim** | 10% | Prediction accuracy (Brier score) |
| **SkillDim** | 10% | Calibration quality |

### Optimizing InfoDim

InfoDim rewards **original predictions** that differ from market consensus:

- **SOS (Source of Signal)**: 60% - Difference from market
- **LeadRatio**: 40% - Anticipating market movements

The custom miner's originality tracker helps optimize these by:
1. Tracking submission timing vs market moves
2. Measuring differentiation from consensus
3. Avoiding herding (copying market exactly)

### Optimizing Timing

Submit predictions **early** for maximum time credit:
- 7+ days before event = 100% time credit
- Decreases logarithmically as event approaches
- Set `EARLY_SUBMISSION_DAYS=7` for optimal timing

## Data Storage

The miner stores data in `~/.sparket/custom_miner/`:

| File | Description |
|------|-------------|
| `elo_ratings.json` | Team Elo ratings |
| `poisson_profiles.json` | Team scoring profiles |
| `calibration.json` | Calibration model data |
| `originality_history.json` | Submission tracking |

## Troubleshooting

### No API Key Error

```
ERROR: No API key set!
```

**Solution:** Set your The-Odds-API key:
```bash
export SPARKET_CUSTOM_MINER__ODDS_API_KEY='your_key'
# Or add to .env.custom
```

### Wallet Not Found

```
ERROR: Wallet not found
```

**Solution:** Ensure wallet exists and is registered:
```bash
btcli wallet list
btcli subnet register --netuid 1 --wallet.name your_wallet --wallet.hotkey your_hotkey
```

### Low API Requests Remaining

```
API requests remaining: 12
```

**Solution:** The-Odds-API free tier has 500 requests/month. Options:
1. Increase `REFRESH_INTERVAL_SECONDS` to reduce API calls
2. Upgrade to paid API tier
3. Miner will use cached data when API is exhausted

### Ensemble Only Using One Model

If live test shows only one component (e.g., market weight: 1.00):
- Ensure Elo ratings are seeded (run with `--seed-elo` first)
- Check team name mapping in logs
- Verify sport is supported (NFL, NBA, MLB, NHL)

## Architecture

### How the Axon Works

The axon is the network interface that allows validators to communicate with your miner:

```
Validator ──synapse──> Miner Axon (0.0.0.0:8094)
                          │
                          ├── forward_fn (handles incoming requests)
                          ├── blacklist_fn (filters unauthorized callers)
                          └── priority_fn (prioritizes by stake)
```

**Flow when using PM2 (`ecosystem.custom.config.js`):**

1. Main entrypoint (`sparket/entrypoints/miner.py`) starts
2. `BaseMinerNeuron` creates and serves the axon on the bittensor network
3. Custom miner (`SPARKET_CUSTOM_MINER__ENABLED=true`) initializes alongside
4. Custom miner generates odds via ensemble model and submits via `ValidatorClient`
5. Axon handles incoming validator requests (connection info, synapse forwarding)

The custom miner does NOT need its own axon -- it shares the main miner's axon and network registration.

### Code Structure

```
sparket/miner/custom/
├── runner.py              # Main miner runner
├── config.py              # Configuration loader
├── live_test.py           # Live market testing
├── backtest.py            # Historical backtesting
├── models/
│   ├── engines/
│   │   ├── elo.py         # Elo rating engine
│   │   ├── poisson.py     # Poisson scoring engine
│   │   └── ensemble.py    # Ensemble combiner
│   └── calibration/
│       └── isotonic.py    # Probability calibration
├── data/
│   ├── fetchers/
│   │   └── odds_api.py    # The-Odds-API client
│   └── seed_elo.py        # Initial Elo ratings
└── strategy/
    └── originality.py     # InfoDim optimization
```

## Updating

To update the miner:

```bash
git pull origin main
uv sync
pm2 restart custom-miner
```

## Support

- **Issues**: https://github.com/anthropics/claude-code/issues
- **Docs**: See `docs/` directory for additional guides
- **Logs**: Check `logs/custom-miner-*.log` for debugging
