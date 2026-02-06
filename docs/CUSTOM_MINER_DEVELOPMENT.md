# Sparket AI Subnet - Custom Miner Development Guide

## Overview

Sparket is a Bittensor subnet for crowdsourced sports odds and outcome validation. Miners submit odds predictions for sports events, and validators score them based on accuracy, timing, and originality.

This document covers the custom miner implementation optimized for maximizing emissions through the scoring system.

---

## Scoring System

Understanding the scoring system is critical for optimization:

| Dimension | Weight | What It Measures | How to Win |
|-----------|--------|------------------|------------|
| **EconDim** | 50% | Closing Line Value (CLV) | Submit odds better than closing line |
| **InfoDim** | 30% | Originality + Lead Market | Be different from other miners, submit early |
| **ForecastDim** | 10% | Prediction Accuracy | Low Brier score, good calibration |
| **SkillDim** | 10% | Beat Market Baseline | PSS > 0 (outperform naive market) |

### Time Credit Rules

| Submission Timing | Credit |
|-------------------|--------|
| 7+ days before event | 100% |
| 1 day before | ~66% |
| 1 hour before | 10% (floor) |
| After event starts | 0% |

**Key insight**: Early bad predictions get 70% penalty forgiveness, late bad predictions get 100% penalty.

---

## Architecture

```
sparket/miner/custom/
├── __init__.py              # Package exports
├── config.py                # Configuration classes
├── runner.py                # Main orchestration (CustomMiner class)
├── demo.py                  # Demo without network
├── backtest.py              # Backtesting framework
├── live_test.py             # Live market testing (no submission)
├── test_odds_api.py         # API integration test
├── data/
│   ├── __init__.py
│   ├── seed_elo.py          # Pre-seed Elo ratings
│   ├── storage/
│   │   ├── __init__.py
│   │   └── line_history.py  # Line movement tracking & steam detection
│   └── fetchers/
│       ├── __init__.py
│       └── odds_api.py      # The-Odds-API integration
├── models/
│   ├── __init__.py
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── elo.py           # Enhanced Elo rating system
│   │   ├── poisson.py       # Poisson scoring model (totals)
│   │   └── ensemble.py      # Ensemble combiner (Elo + Market + Poisson)
│   └── calibration/
│       ├── __init__.py
│       └── isotonic.py      # Probability calibration
└── strategy/
    ├── __init__.py
    ├── timing.py            # Submission timing optimization
    └── originality.py       # InfoDim optimization (SOS + LeadRatio)
```

### Axon Integration

The custom miner runs inside the main miner entrypoint (`sparket/entrypoints/miner.py`), sharing its axon for validator communication:

```
Validator ──synapse──> Miner Axon (0.0.0.0:8094)
                          │
                          ├── BaseMinerNeuron (axon, wallet, metagraph)
                          └── CustomMiner (ensemble odds generation)
                                ├── EnsembleEngine
                                │   ├── EloEngine
                                │   ├── PoissonEngine
                                │   └── MarketConsensus (The-Odds-API)
                                ├── IsotonicCalibrator
                                ├── OriginalityTracker
                                ├── LineHistory
                                └── TimingStrategy
```

Enable with `SPARKET_CUSTOM_MINER__ENABLED=true` and `SPARKET_BASE_MINER__ENABLED=false`.

---

## Components

### 1. Ensemble Engine (`models/engines/ensemble.py`)

Combines three prediction sources with confidence-based weighting:

| Engine | Default Weight | Description |
|--------|---------------|-------------|
| **Market** | 55% | Sharp book consensus (Pinnacle weighted 3x) |
| **Elo** | 35% | Team strength ratings with home advantage |
| **Poisson** | 10% | Scoring distribution for totals |

**Weighting algorithm:**
1. Start with base weights from config
2. Apply sharpness bonus (sharper predictions get up to 1.5x weight)
3. Apply confidence scaling (model confidence adjusts weight)
4. Enforce minimum weight floor (5%)
5. Normalize to sum to 1.0

```python
from sparket.miner.custom.models.engines.ensemble import EnsembleEngine

ensemble = EnsembleEngine(
    elo_engine=elo,
    poisson_engine=poisson,
    base_weights={"elo": 0.35, "market": 0.55, "poisson": 0.10},
    confidence_scaling=True,
)

prediction = ensemble.predict(
    market={"home_team": "KC", "away_team": "BUF", "sport": "NFL", "kind": "MONEYLINE"},
    market_odds=market_consensus,
)
# Returns: EnsemblePrediction with home_prob, away_prob, confidence, components
```

### 2. Enhanced Elo Engine (`models/engines/elo.py`)

Sport-specific Elo rating system with:

- **K-factors by sport**: NFL=32, NBA=20, MLB=8, NHL=16
- **Home field advantage**: NFL=2.5pts, NBA=3.0pts, MLB=0.5pts, NHL=0.5pts
- **Margin of Victory (MOV)**: Adjusts rating changes based on game margin
- **Log5 probability**: Bill James formula for matchup probabilities

```python
from sparket.miner.custom.models.engines.elo import EloEngine

elo = EloEngine()
odds = elo.get_odds_sync({
    "market_id": 1,
    "kind": "MONEYLINE",
    "home_team": "KC",
    "away_team": "BUF",
    "sport": "NFL"
})
# Returns: OddsPrices with home_prob, away_prob, home_odds_eu, away_odds_eu
```

### 3. Poisson Engine (`models/engines/poisson.py`)

Scoring distribution model for TOTAL (over/under) markets:

- Predicts expected goals/points for each team
- Uses Poisson distribution to calculate over/under probabilities
- Maintains team scoring profiles (attack/defense strengths)
- Updates from game results

```python
from sparket.miner.custom.models.engines.poisson import PoissonEngine

poisson = PoissonEngine(data_path="poisson_profiles.json")
result = poisson.predict_total(
    home_team="KC", away_team="BUF", sport="NFL", line=47.5
)
# Returns: TotalPrediction with over_prob, under_prob, expected_total
```

### 4. Isotonic Calibration (`models/calibration/isotonic.py`)

Pool Adjacent Violators (PAV) algorithm for probability calibration:

- Fits monotonic function to historical predictions vs outcomes
- Corrects systematic over/under-confidence
- Maintains probability pairs summing to 1.0
- Auto-refits after configurable number of new samples

```python
from sparket.miner.custom.models.calibration.isotonic import IsotonicCalibrator

calibrator = IsotonicCalibrator(min_samples=100)
calibrator.add_sample(predicted=0.65, actual=1.0)
# After enough samples:
calibrated_home, calibrated_away = calibrator.calibrate_pair(0.65, 0.35)
```

### 5. Timing Strategy (`strategy/timing.py`)

Optimizes submission timing for maximum time credit:

- Submit 7+ days early for 100% credit
- Refresh every 6 hours with updated predictions
- Never submit within 1 hour of event (10% credit floor)

```python
from sparket.miner.custom.strategy.timing import TimingStrategy

timing = TimingStrategy()
decision = timing.evaluate(market, current_time)
# Returns: SubmissionDecision with should_submit, time_credit, refresh_at
```

### 6. Originality Tracker (`strategy/originality.py`)

Optimizes InfoDim (30% of total score) by tracking:

- **SOS (Source of Signal)** - 60%: Difference from market consensus
- **LeadRatio** - 40%: Anticipating market movements before they happen

```python
from sparket.miner.custom.strategy.originality import OriginalityTracker

tracker = OriginalityTracker(data_path="originality.json")
tracker.record_submission(market_id=1, our_prob=0.58, market_prob=0.52)
# Later, when market moves:
tracker.record_market_move(market_id=1, new_market_prob=0.56)

stats = tracker.stats()
# Returns: avg_sos, avg_lead_ratio, submission_count
```

### 7. Line History (`data/storage/line_history.py`)

Tracks market line movements for optimal submission timing:

- Records odds snapshots over time
- Detects steam moves (sharp money moving the line)
- Determines optimal submission timing based on line stability
- Cleans up old market data automatically

```python
from sparket.miner.custom.data.storage.line_history import LineHistory

history = LineHistory(data_path="line_history.json")
history.record(market_id=1, home_prob=0.55, away_prob=0.45, source="market")

should_submit, reason = history.should_submit_now(
    market_id=1, our_home_prob=0.58, hours_to_game=48.0
)
steam_moves = history.get_recent_steam_moves(hours=24.0)
```

### 8. The-Odds-API Integration (`data/fetchers/odds_api.py`)

Market consensus from 35+ bookmakers:

- **Sharp-weighted averaging**: Pinnacle (3.0x), Betfair (2.5x), others (1.0x)
- **Consensus odds**: Weighted average across all books
- **Blending function**: Combine model + market predictions

```python
from sparket.miner.custom.data.fetchers.odds_api import OddsAPIFetcher, blend_with_market

fetcher = OddsAPIFetcher(api_key="your_key")
games = await fetcher.get_all_games("NFL")

# Blend model prediction with market
blended = blend_with_market(
    model_prob=0.55,      # Your model says 55%
    market_prob=0.52,     # Market says 52%
    model_weight=0.4,     # 40% model, 60% market
    vig=0.05              # 5% vig
)
```

### 9. Seed Elo Ratings (`data/seed_elo.py`)

Pre-computed ratings for all teams across 4 major sports:

| Sport | Teams | Rating Range |
|-------|-------|--------------|
| NFL | 32 | 1370-1680 |
| NBA | 30 | 1340-1680 |
| MLB | 30 | 1350-1600 |
| NHL | 32 | 1360-1620 |

```bash
# Export ratings to JSON
uv run python -m sparket.miner.custom.data.seed_elo
# Saves to: ~/.sparket/custom_miner/elo_ratings.json
```

---

## Configuration

### Environment Variables

```bash
# ===== REQUIRED =====
SPARKET_WALLET__NAME=default
SPARKET_WALLET__HOTKEY=default
SPARKET_NETUID=1

# ===== Axon =====
SPARKET_AXON__HOST=0.0.0.0
SPARKET_AXON__PORT=8094

# ===== Miner Selection =====
SPARKET_CUSTOM_MINER__ENABLED=true
SPARKET_BASE_MINER__ENABLED=false

# ===== Market Data =====
SPARKET_CUSTOM_MINER__ODDS_API_KEY=your_api_key

# ===== Model =====
SPARKET_CUSTOM_MINER__VIG=0.045
SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__ELO=0.35
SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__MARKET=0.55
SPARKET_CUSTOM_MINER__ENGINE_WEIGHTS__POISSON=0.10

# ===== Timing =====
SPARKET_CUSTOM_MINER__TIMING__EARLY_SUBMISSION_DAYS=7
SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS=21600

# ===== Calibration =====
SPARKET_CUSTOM_MINER__CALIBRATION__ENABLED=true
SPARKET_CUSTOM_MINER__CALIBRATION__MIN_SAMPLES=100

# ===== Rate Limiting =====
SPARKET_CUSTOM_MINER__RATE_LIMIT_PER_MINUTE=60
SPARKET_CUSTOM_MINER__PER_MARKET_LIMIT_PER_MINUTE=10

# ===== Outcome Detection =====
SPARKET_CUSTOM_MINER__OUTCOME_CHECK_SECONDS=300
```

### Config Classes (`config.py`)

```python
@dataclass
class EloConfig:
    k_factor: dict = {"NFL": 20, "NBA": 12, "MLB": 4, "NHL": 10}
    home_advantage: dict = {"NFL": 48, "NBA": 100, "MLB": 24, "NHL": 33}
    mov_multiplier: float = 1.0
    default_rating: float = 1500.0

@dataclass
class TimingConfig:
    early_submission_days: int = 7
    refresh_interval_seconds: int = 21600  # 6 hours
    min_hours_before_event: float = 1.0    # validator min_minutes=60
    cutoff_hours: float = 0.25

@dataclass
class CalibrationConfig:
    enabled: bool = True
    min_samples: int = 100
    retrain_interval: int = 500  # refit every N samples

@dataclass
class CustomMinerConfig:
    enabled: bool = False
    elo: EloConfig
    timing: TimingConfig
    calibration: CalibrationConfig
    vig: float = 0.045
    engine_weights: dict = {"elo": 0.35, "market": 0.55, "poisson": 0.10}
    odds_api_key: str | None = None
    rate_limit_per_minute: int = 60
    per_market_limit_per_minute: int = 10
    outcome_check_seconds: int = 300
```

---

## Running the Miner

### Production Mode (PM2 + Axon)

This is the recommended mode for earning emissions. Uses the main miner entrypoint with axon serving:

```bash
# 1. Create wallet
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

# 2. Register on subnet (need TAO)
btcli subnet register --netuid <NETUID> --wallet.name miner

# 3. Configure
cp .env.custom.example .env.custom
# Edit .env.custom with your wallet and API key

# 4. Run with PM2
pm2 start ecosystem.custom.config.js
pm2 logs custom-miner
```

### Direct Execution (with Axon)

```bash
SPARKET_CUSTOM_MINER__ENABLED=true SPARKET_BASE_MINER__ENABLED=false \
  uv run python sparket/entrypoints/miner.py
```

### Live Test (No Submission)

Test predictions against real markets without submitting:

```bash
# Test NFL markets
uv run python -m sparket.miner.custom.live_test --sport NFL -v

# Test NBA markets
uv run python -m sparket.miner.custom.live_test --sport NBA -v

# Limit events
uv run python -m sparket.miner.custom.live_test --sport NFL --max-events 20
```

Output shows per-game predictions, edge vs market, component weights, and summary statistics.

### Demo Mode (No Network)

```bash
# Test locally without Bittensor
uv run python -m sparket.miner.custom.demo
```

### Backtest Mode

```bash
# Run backtesting simulation
uv run python -m sparket.miner.custom.backtest \
    --games 100 \
    --warmup 50 \
    --seed-elo \
    --seed 42
```

Outputs CLV, Brier score, calibration, and PSS metrics.

### Standalone Runner (No Axon, Testing Only)

```bash
uv run python -m sparket.miner.custom.runner
```

**Warning:** This mode does NOT serve an axon. Validators cannot reach this miner on the network. Use only for local testing.

---

## API Keys

### The-Odds-API

1. Sign up at: https://the-odds-api.com/
2. Free tier: 500 requests/month
3. Add to `.env.custom`:
   ```
   SPARKET_CUSTOM_MINER__ODDS_API_KEY=your_key_here
   ```

### Test API Connection

```bash
uv run python -m sparket.miner.custom.test_odds_api --sport NFL
```

---

## Improvement Roadmap

### Implemented

- [x] Enhanced Elo Engine (sport-specific K-factors, MOV, home advantage)
- [x] Isotonic Calibration (PAV algorithm, auto-refit)
- [x] Timing Strategy (early submission, time credit optimization)
- [x] The-Odds-API Integration (sharp-weighted consensus)
- [x] Seed Elo Ratings (all 4 major sports)
- [x] Line Movement Tracking (steam detection, optimal timing)
- [x] Originality Optimization (SOS + LeadRatio for InfoDim)
- [x] Poisson Engine (totals/over-under markets)
- [x] Ensemble Engine (Elo + Market + Poisson with confidence scaling)
- [x] Dynamic Model Weighting (sharpness bonus, confidence scaling)
- [x] Axon Integration (custom miner runs inside main entrypoint)
- [x] PM2 Configuration (production deployment)

### Not Yet Implemented

#### 1. Enhanced ESPN Fetcher
**Impact: MEDIUM**

Integrate injury/rest data:
- Late scratches (market slow to adjust)
- Back-to-back games (NBA)
- Travel distance
- Rest days advantage

```python
# Proposed: data/fetchers/espn_enhanced.py
class ESPNEnhancedFetcher:
    def get_injuries(self, team) -> list[Injury]
    def get_rest_days(self, team) -> int
    def get_travel_distance(self, team) -> float
```

#### 2. Adaptive Weight Learning
**Impact: MEDIUM**

Online learning for ensemble weights based on historical accuracy:

```python
# Currently uses Brier-style accuracy tracking in EnsembleEngine.update_accuracy()
# Future: automatically adjust base_weights based on rolling accuracy
```

#### 3. Weather / Venue Data
**Impact: LOW-MEDIUM**

Weather conditions for outdoor sports (NFL, MLB):
- Wind speed affecting totals
- Temperature affecting scoring
- Dome vs outdoor venues

---

## Testing

### Unit Tests

```bash
# Run all custom miner tests
pytest tests/miner/custom/ -v

# Specific test files
pytest tests/miner/custom/test_elo.py
pytest tests/miner/custom/test_calibration.py
pytest tests/miner/custom/test_timing.py
```

### Integration Test

```bash
# Test full pipeline with API
uv run python -m sparket.miner.custom.test_odds_api --sport NFL
```

### Live Test

```bash
# Test predictions against real markets
uv run python -m sparket.miner.custom.live_test --sport NBA -v
```

### Backtest Validation

```bash
# Validate against simulated validator scoring
uv run python -m sparket.miner.custom.backtest \
    --games 200 \
    --warmup 50 \
    --seed-elo \
    --seed 42
```

Expected metrics for well-tuned miner:
- CLV: > 0 (positive = beating closing lines)
- Brier: < 0.25 (lower = more accurate)
- Calibration: > 0.8 (closer to 1.0 = better)
- PSS: > 0 (positive = beating market baseline)

---

## Payload Format

When submitting to validators:

```python
{
    "miner_hotkey": str,  # Your wallet hotkey (ss58 address)
    "submissions": [{
        "market_id": int,
        "kind": "moneyline",  # or "spread", "total" (lowercase)
        "priced_at": datetime,  # ISO 8601 format
        "prices": [
            {"side": "home", "odds_eu": 1.91, "imp_prob": 0.52},
            {"side": "away", "odds_eu": 2.05, "imp_prob": 0.48},
        ]
    }],
}
```

**Important Notes:**
- `miner_id` is **not required** in payload - validator derives it from authenticated hotkey
- `odds_eu` must be in range `(1.01, 1000]`
- `imp_prob` must be in range `(0.001, 0.999)`
- `kind` must be lowercase: "moneyline", "spread", or "total"
- `side` must be a valid value: `home`, `away`, `draw`, `over`, `under` (case-insensitive, normalized to uppercase by validator). Invalid sides are silently skipped
- `priced_at` must be within ±5 minutes of submission time
- `token` must be included for validators that require it (rotates every ~1 hour)

---

## Data Storage

The miner stores persistent data in `~/.sparket/custom_miner/`:

| File | Description |
|------|-------------|
| `elo_ratings.json` | Team Elo ratings (updated after each game) |
| `poisson_profiles.json` | Team scoring profiles (attack/defense) |
| `calibration.json` | Calibration model data (prediction vs outcome) |
| `line_history.json` | Market line movement history |
| `originality.json` | Submission tracking for InfoDim |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `sparket/entrypoints/miner.py` | Main miner entrypoint (axon + custom miner integration) |
| `ecosystem.custom.config.js` | PM2 configuration for production |
| `.env.custom.example` | Environment variable template |
| `custom/config.py` | All configuration classes |
| `custom/runner.py` | CustomMiner class (orchestration) |
| `custom/live_test.py` | Live market testing (no submission) |
| `custom/backtest.py` | Historical simulation |
| `custom/demo.py` | Local testing without network |
| `custom/models/engines/ensemble.py` | Ensemble combiner (Elo + Market + Poisson) |
| `custom/models/engines/elo.py` | Elo rating system |
| `custom/models/engines/poisson.py` | Poisson scoring model |
| `custom/models/calibration/isotonic.py` | Probability calibration |
| `custom/strategy/timing.py` | Submission timing |
| `custom/strategy/originality.py` | InfoDim optimization |
| `custom/data/storage/line_history.py` | Line movement tracking |
| `custom/data/fetchers/odds_api.py` | Market data API |
| `custom/data/seed_elo.py` | Pre-computed ratings |

---

## Troubleshooting

### "No API key provided"
```bash
export SPARKET_CUSTOM_MINER__ODDS_API_KEY=your_key
# Or add to .env.custom
```

### "Keyfile does not exist"
```bash
# Create wallet first
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

### "Team not found in Elo ratings"
```bash
# Re-seed ratings
uv run python -m sparket.miner.custom.data.seed_elo
```

### Ensemble only using one component
- Ensure Elo ratings are seeded (run with `--seed-elo`)
- Check team name mapping - API returns full names ("New England Patriots"), Elo uses codes ("NE")
- The `get_team_by_name()` function handles normalization
- Verify sport is supported (NFL, NBA, MLB, NHL)

### Poor backtest results
- Ensure `--seed-elo` flag is used
- Use `--warmup 50` to let calibration stabilize
- Check simulated market noise (should be ~8%)
- Verify calibration has enough samples (min 100)

### Custom miner not starting in PM2
- Check `SPARKET_CUSTOM_MINER__ENABLED=true` is set
- Check `SPARKET_BASE_MINER__ENABLED=false` to avoid conflicts
- Verify wallet exists and is registered: `btcli wallet list`
- Check logs: `pm2 logs custom-miner`

---

## Resources

- [The-Odds-API Documentation](https://the-odds-api.com/liveapi/guides/v4/)
- [Bittensor Documentation](https://docs.bittensor.com/)
- [Elo Rating System](https://en.wikipedia.org/wiki/Elo_rating_system)
- [Log5 Method](https://en.wikipedia.org/wiki/Log5)
- [Isotonic Regression](https://en.wikipedia.org/wiki/Isotonic_regression)

---

*Last updated: February 2026*
