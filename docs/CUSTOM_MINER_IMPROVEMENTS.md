# Custom Miner Improvement Plan

This document outlines the prioritized improvement plan for the custom miner based on the Sparket subnet scoring system.

## Scoring System Overview

| Dimension | Weight | Components |
|-----------|--------|------------|
| **EconDim** | 50% | CLV (Closing Line Value), CLE (Expected Edge), MES |
| **InfoDim** | 30% | SOS (Source of Signal) 60%, LeadRatio 40% |
| **ForecastDim** | 10% | Brier Score 60%, Calibration 40% |
| **SkillDim** | 10% | PSS (Probabilistic Skill Score) vs market |

## High Impact Improvements (EconDim 50% + InfoDim 30%)

### 1. Real Historical Data Backtest
**Impact:** High | **Effort:** Medium | **Status:** Planned

Current backtest uses simulated games which doesn't validate real-world performance. Need to:
- Collect real historical game data with market odds
- Implement backtest against actual closing lines
- Validate CLV and CLE metrics accurately

### 2. Originality Tracking (InfoDim 30%)
**Impact:** High | **Effort:** Medium | **Status:** Implemented

Track and optimize for SOS (Source of Signal) and LeadRatio:
- Calculate how different our predictions are from market consensus
- Detect opportunities to anticipate market moves
- Balance originality with accuracy

Key metrics:
- **SOS**: 1 - |correlation with market| (higher = more original)
- **LeadRatio**: Fraction of market moves we anticipated

Implementation: `sparket/miner/custom/strategy/originality.py`

### 3. Sharper Market Blending
**Impact:** Medium | **Effort:** Low | **Status:** Implemented

Weight Pinnacle odds more heavily (3x sharp book weight):
- Pinnacle has the sharpest lines in the market
- Closer alignment with sharp books improves CLV
- Already implemented in `OddsAPIFetcher`

### 4. Early Submission Strategy
**Impact:** Medium | **Effort:** Low | **Status:** Implemented

Maximize time credit and lead ratio:
- Submit 7+ days early for 100% time credit
- Submit before market moves in predicted direction
- Use line movement tracking for optimal timing

Implementation: `sparket/miner/custom/data/storage/line_history.py`

## Medium Impact Improvements (ForecastDim 10% + SkillDim 10%)

### 5. Additional Features
**Impact:** Medium | **Effort:** High | **Status:** Planned

Add predictive signals beyond Elo:
- Injury reports (key player status)
- Rest days (back-to-back detection)
- Home/away performance splits
- Weather conditions (outdoor sports)
- Historical matchup data

### 6. Multi-Sport Models
**Impact:** Medium | **Effort:** Medium | **Status:** Partial

Sport-specific tuning:
- NFL: Implemented (Elo + Poisson for totals)
- NBA: Basic support
- MLB: Basic support
- NHL: Basic support

Needs sport-specific K-factors, home advantage, and scoring patterns.

### 7. Ensemble Model
**Impact:** Medium | **Effort:** Medium | **Status:** Planned

Combine multiple prediction engines:
- Elo for win probabilities
- Poisson for totals
- Market consensus for sharp signal
- Weighted ensemble based on confidence

## Implementation Status

### Completed
- [x] Elo rating system with sport-specific K-factors
- [x] Isotonic calibration for probability accuracy
- [x] Poisson model for TOTAL markets
- [x] Line movement tracking and steam detection
- [x] The-Odds-API integration for market blending
- [x] Originality tracking for InfoDim optimization
- [x] Strategic timing based on time-to-close
- [x] Rate limiting and bucket cleanup

### In Progress
- [ ] Integration of originality tracking into runner

### Planned
- [ ] Real historical data collection
- [ ] Backtest with actual closing lines
- [ ] Injury data integration
- [ ] Weather data integration
- [ ] Enhanced multi-sport models
- [ ] Ensemble model architecture

## Key Files

| File | Purpose |
|------|---------|
| `sparket/miner/custom/runner.py` | Main orchestration |
| `sparket/miner/custom/models/engines/elo.py` | Elo rating model |
| `sparket/miner/custom/models/engines/poisson.py` | Poisson for totals |
| `sparket/miner/custom/models/calibration/isotonic.py` | Probability calibration |
| `sparket/miner/custom/strategy/timing.py` | Submission timing |
| `sparket/miner/custom/strategy/originality.py` | Originality tracking |
| `sparket/miner/custom/data/storage/line_history.py` | Line movement |
| `sparket/miner/custom/data/fetchers/odds_api.py` | Market data |

## Backtest Results (Baseline)

With 200 games, warmup 50, seed 42:

| Metric | Value | Target |
|--------|-------|--------|
| Brier Score | 0.455 | < 0.25 |
| CLV | -0.17% | > 0% |
| CLE | 0.00% | > 0% |
| Win Rate | 50.5% | > 52% |
| Calibration | 0.79 | > 0.9 |

The Elo-only model performs at market level. Improvements above should push CLV positive and improve all metrics.

## Next Steps

1. **Complete originality integration** - Add to runner.py
2. **Collect real data** - Historical games with closing lines
3. **Validate with real backtest** - Measure actual CLV
4. **Add injury signals** - Start with NFL injuries
5. **Build ensemble** - Combine Elo + Poisson + Market
