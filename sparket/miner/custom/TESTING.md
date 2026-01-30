# Custom Miner Testing Guide

This guide covers all the ways to test and validate the custom miner before deploying to production.

## Testing Options Overview

| Method | Purpose | Data Source | Effort |
|--------|---------|-------------|--------|
| **Unit Tests** | Verify components work | None | Low |
| **Backtest** | Evaluate predictions | Simulated | Low |
| **Test Mode** | Integration test | Mock provider | Medium |
| **Paper Trading** | Real-world validation | Live markets | High |

---

## 1. Unit Tests

Run the unit tests to verify all components are working:

```bash
# Run all custom miner tests
uv run pytest tests/miner/custom/ -v

# Run specific test file
uv run pytest tests/miner/custom/test_elo.py -v

# Run with coverage
uv run pytest tests/miner/custom/ --cov=sparket.miner.custom
```

**What's tested:**
- Elo engine: K-factors, rating updates, probability calculations
- Calibration: Isotonic regression, monotonicity
- Timing: Submission decisions, time credit calculation

---

## 2. Backtesting

The backtest simulates games and evaluates predictions using **actual validator metrics**.

### Basic Usage

```bash
# Quick test with 100 games
uv run python -m sparket.miner.custom.backtest --games 100

# With warmup to train Elo ratings first
uv run python -m sparket.miner.custom.backtest --games 200 --warmup 300

# Test specific sport
uv run python -m sparket.miner.custom.backtest --games 100 --sport NBA

# Verbose output (per-game results)
uv run python -m sparket.miner.custom.backtest --games 50 -v

# Reproducible results
uv run python -m sparket.miner.custom.backtest --games 200 --seed 42

# Save results to file
uv run python -m sparket.miner.custom.backtest --games 200 --output results.json
```

### Understanding Results

```
============================================================
BACKTEST RESULTS
============================================================
Total Games:        200
Avg Brier Score:    0.2400  (lower is better, 0.25 = random)
Avg Log Loss:       0.6900  (lower is better, 0.693 = random)
Avg CLV:            +0.0150  (positive = beating market)
Avg CLE:            +0.0200  (expected edge per bet)
Avg PSS vs Close:   +0.0500  (positive = better than market)
Win Rate vs Close:  52.0%
Calibration Slope:  0.950   (1.0 = perfect)
============================================================
```

**Key Metrics:**

| Metric | Target | Meaning |
|--------|--------|---------|
| Brier Score | < 0.25 | Lower = better accuracy |
| Log Loss | < 0.693 | Lower = better calibration |
| CLV | > 0 | Positive = beating closing line |
| CLE | > 0 | Positive = profitable edge |
| PSS vs Close | > 0 | Positive = skill over market |
| Win Rate | > 50% | Beat market more than half the time |
| Calibration Slope | ~1.0 | Predictions match actual frequencies |

### Validator Scoring Dimensions

The backtest measures the same dimensions as the validator:

| Dimension | Weight | Backtest Metric |
|-----------|--------|-----------------|
| EconDim | 50% | CLV, CLE |
| InfoDim | 30% | (Not in backtest - requires real-time data) |
| ForecastDim | 10% | Brier, Log Loss, Calibration |
| SkillDim | 10% | PSS vs Closing |

---

## 3. Test Mode (Integration Testing)

Test the full miner with the validator infrastructure using mock data.

### Setup

```bash
# 1. Start the test database (if not running)
docker compose -f docker-compose.test.yml up -d

# 2. Start validator in test mode
pm2 start ecosystem.test.config.js

# 3. Check logs
pm2 logs validator-test
```

### Run Custom Miner Against Test Validator

```bash
# Set environment for test mode
export SPARKET_TEST_MODE=true
export SPARKET_CUSTOM_MINER__ENABLED=true

# Run the custom miner
uv run python -m sparket.miner.custom.runner
```

### Seed Test Data

The test validator has a control API for seeding data:

```bash
# Seed mock events
curl -X POST http://127.0.0.1:8199/seed-events

# Seed ground truth odds
curl -X POST http://127.0.0.1:8199/seed-ground-truth

# Trigger scoring
curl -X POST http://127.0.0.1:8199/trigger-scoring

# Check miner scores
curl http://127.0.0.1:8199/scores
```

### Using Mock Provider Directly

```python
from sparket.devtools.mock_provider import MockProvider

# Get singleton
provider = MockProvider.get()

# Create mock event
provider.create_event(
    event_id=1,
    home_team="KC",
    away_team="BUF",
    sport="NFL",
    start_time=datetime.now(timezone.utc) + timedelta(days=3),
)

# Create market with known true probability
provider.create_market(
    market_id=1,
    event_id=1,
    kind="MONEYLINE",
    true_prob_home=0.65,  # Ground truth
)

# Generate realistic odds time series
provider.generate_odds_series(
    market_id=1,
    true_prob_home=0.65,
    open_time=datetime.now(timezone.utc) - timedelta(days=7),
    close_time=datetime.now(timezone.utc) + timedelta(days=3),
)

# Set outcome after event settles
provider.set_outcome(event_id=1, result="HOME", score_home=27, score_away=17)
```

---

## 4. Paper Trading (Pre-Production Validation)

Run the custom miner alongside the base miner without actually affecting scores.

### Setup

```bash
# Create separate data directory for paper trading
export SPARKET_CUSTOM_MINER__DATA_DIR=~/.sparket/paper_trading

# Disable actual submissions (dry run)
export SPARKET_CUSTOM_MINER__DRY_RUN=true

# Run miner
uv run python -m sparket.miner.custom.runner
```

### Monitor Performance

```python
from sparket.miner.custom import CustomMiner

# Get diagnostics
diagnostics = miner.get_diagnostics()
print(diagnostics)

# Output:
# {
#     "running": True,
#     "submissions_count": 150,
#     "errors_count": 2,
#     "calibration": {
#         "sample_count": 100,
#         "brier_score": 0.23,
#         "calibration_slope": 0.95,
#         "reliability_diagram": [...]
#     },
#     "config": {...}
# }
```

---

## 5. Interpreting Poor Results

If backtest shows poor performance, here's how to diagnose:

### High Brier Score (> 0.25)
- Model predictions are inaccurate
- **Fix**: More warmup games, better features, calibration

### Negative CLV
- Model is worse than market closing line
- **Fix**: This is expected initially; market is efficient

### Low Calibration Slope (< 0.8)
- Model is underconfident (predictions too close to 50%)
- **Fix**: Increase Elo K-factors, reduce regularization

### High Calibration Slope (> 1.2)
- Model is overconfident (predictions too extreme)
- **Fix**: Increase calibration min_samples, add regularization

### Low Win Rate (< 45%)
- Model rarely beats market
- **Fix**: Add more data sources, improve timing strategy

---

## 6. Expected Performance

### Initial (No Training)
- Brier: ~0.25 (random)
- CLV: ~0 (no edge)
- PSS: ~0 (market level)

### After Warmup (300+ games)
- Brier: 0.22-0.24
- CLV: -0.05 to +0.05
- PSS: -0.1 to +0.1

### With Real Data & Calibration
- Brier: < 0.22
- CLV: > 0 consistently
- PSS: > 0 (beating market)

---

## 7. Improving the Model

### Phase 2 Enhancements (Planned)
1. **Line History Tracking** - Track market movements for CLV analysis
2. **Enhanced ESPN Fetcher** - Add injuries, rest days, recent form
3. **The-Odds-API Integration** - Multiple books for consensus

### Phase 3 Enhancements (Planned)
1. **Poisson Model** - For NHL/soccer totals
2. **Ensemble Engine** - Combine multiple models
3. **Originality Tracker** - Reduce correlation with market

---

## Quick Reference

```bash
# Run all tests
uv run pytest tests/miner/custom/ -v

# Quick backtest
uv run python -m sparket.miner.custom.backtest --games 100 --warmup 200

# Test mode
pm2 start ecosystem.test.config.js && uv run python -m sparket.miner.custom.runner

# Production
pm2 start ecosystem.config.js
```
