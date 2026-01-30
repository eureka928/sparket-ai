# How to Improve Custom Miner Results

## Current State

With seeded Elo ratings, the model produces realistic predictions but still underperforms the efficient market. Here's how to improve each scoring dimension.

---

## 1. EconDim (50% of score) - Beat Closing Lines

**Goal**: Positive CLV (your odds better than closing line)

### Quick Wins

```bash
# Use seeded ratings (already done)
python -m sparket.miner.custom.backtest --games 100 --seed-elo

# This gives ~5-10% CLV improvement
```

### Medium Effort: Blend with Market Odds

Add The-Odds-API to get real market consensus, then blend:

```python
# In custom/config.py - adjust blend weight
engine_weights = {
    "elo": 0.40,      # Your model
    "market": 0.50,   # Market consensus (follow sharp money)
    "adjustment": 0.10,  # Your edge factors
}
```

**Implementation**: Create `sparket/miner/custom/data/fetchers/odds_api.py`

```python
import httpx

class OddsAPIFetcher:
    """Fetch odds from The-Odds-API."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_odds(self, sport: str, market: str = "h2h"):
        """Get current odds from multiple books."""
        url = f"{self.BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": market,
            "bookmakers": "pinnacle,draftkings,fanduel,betmgm",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params)
            return resp.json()

    def get_consensus(self, odds_data: list) -> float:
        """Calculate sharp-weighted consensus probability."""
        # Weight Pinnacle highest (sharpest book)
        weights = {"pinnacle": 2.0, "draftkings": 1.0, "fanduel": 1.0}
        # ... calculate weighted average
```

### High Effort: Line Movement Tracking

Track how lines move and bet against overreactions:

```python
# In custom/data/storage/line_history.py
class LineHistory:
    """Track line movements for each market."""

    def record_line(self, market_id: int, timestamp: datetime, odds: float):
        """Record a line observation."""

    def get_movement(self, market_id: int, hours: int = 24) -> float:
        """Get line movement over time period."""
        # Positive = line moving toward home
        # Negative = line moving toward away

    def detect_steam_move(self, market_id: int) -> bool:
        """Detect sharp money steam move."""
        # Rapid line movement across multiple books
```

---

## 2. InfoDim (30% of score) - Originality & Speed

**Goal**: High SOS (uncorrelated predictions), high lead ratio

### Quick Win: Submit Early

The timing strategy already does this, but ensure you're submitting 7+ days out:

```python
# In custom/config.py
timing = TimingConfig(
    early_submission_days=7.0,  # Full time credit
    refresh_interval_seconds=6 * 3600,  # Update every 6 hours
)
```

### Medium Effort: Originality Tracking

Deviate from market consensus to increase SOS:

```python
# In custom/strategy/originality.py
class OriginalityTracker:
    """Track and encourage prediction originality."""

    def __init__(self, min_deviation: float = 0.02):
        self.min_deviation = min_deviation
        self._market_consensus: Dict[int, float] = {}

    def adjust_for_originality(
        self,
        market_id: int,
        model_prob: float,
        market_prob: float,
    ) -> float:
        """Adjust prediction to be more original.

        If model agrees with market, push slightly away.
        If model disagrees, keep the disagreement.
        """
        diff = model_prob - market_prob

        if abs(diff) < self.min_deviation:
            # Too close to market - add some edge
            # Use model's direction but amplify
            if diff >= 0:
                return model_prob + self.min_deviation
            else:
                return model_prob - self.min_deviation

        return model_prob  # Already original enough
```

---

## 3. ForecastDim (10% of score) - Calibration

**Goal**: Brier < 0.25, calibration slope ~1.0

### Quick Win: More Calibration Data

The isotonic calibrator needs 100+ samples to work well:

```python
# In custom/config.py
calibration = CalibrationConfig(
    enabled=True,
    min_samples=100,  # Wait for enough data
    retrain_interval=50,  # Retrain frequently
)
```

### Medium Effort: Sport-Specific Calibration

Different sports have different calibration curves:

```python
# Separate calibrators per sport
self._calibrators = {
    "NFL": IsotonicCalibrator(min_samples=50),
    "NBA": IsotonicCalibrator(min_samples=100),
    "MLB": IsotonicCalibrator(min_samples=200),
    "NHL": IsotonicCalibrator(min_samples=100),
}
```

---

## 4. SkillDim (10% of score) - Beat Market Baseline

**Goal**: Positive PSS (better than closing line)

### Key Insight

PSS measures: `1 - (your_brier / market_brier)`

To beat the market, you need **information the market doesn't have**:

1. **Injuries announced after line set**
2. **Weather changes**
3. **Rest days / travel fatigue**
4. **Motivation factors** (rivalry games, playoff implications)

### Implementation: Enhanced Data

```python
# In custom/data/fetchers/espn_enhanced.py
class EnhancedESPNFetcher:
    """Fetch additional factors from ESPN."""

    async def get_injury_report(self, team: str, sport: str) -> dict:
        """Get current injury status."""
        # Key players out = big impact

    async def get_rest_days(self, team: str, sport: str) -> int:
        """Days since last game."""
        # Back-to-back = disadvantage

    async def get_travel_distance(self, home: str, away: str) -> float:
        """Calculate travel fatigue factor."""
        # Cross-country = slight disadvantage
```

Then adjust Elo:

```python
def adjust_for_factors(self, base_prob: float, factors: dict) -> float:
    """Adjust probability for situational factors."""
    adj = base_prob

    # Injury adjustment
    if factors.get("key_player_out"):
        adj -= 0.05  # 5% penalty

    # Rest advantage
    rest_diff = factors.get("home_rest", 0) - factors.get("away_rest", 0)
    adj += rest_diff * 0.01  # 1% per rest day advantage

    # Travel fatigue
    if factors.get("away_travel_miles", 0) > 2000:
        adj += 0.02  # 2% home boost for long travel

    return max(0.1, min(0.9, adj))
```

---

## Implementation Priority

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| 1 | ✅ Seed Elo ratings | High | Done |
| 2 | The-Odds-API integration | High | Medium |
| 3 | Injury/rest data | Medium | Medium |
| 4 | Line movement tracking | Medium | High |
| 5 | Originality adjustment | Low | Low |
| 6 | Sport-specific calibration | Low | Low |

---

## Testing After Improvements

```bash
# After implementing The-Odds-API:
uv run python -m sparket.miner.custom.backtest \
    --games 200 --seed-elo --seed 42

# Expected improvement:
# CLV: -0.04 → +0.01 to +0.03
# PSS: -0.36 → -0.10 to +0.05
# Win Rate: 40% → 48-52%
```

---

## Production Deployment

Once backtest shows positive CLV consistently:

```bash
# 1. Seed your Elo ratings
uv run python -m sparket.miner.custom.data.seed_elo \
    --output ~/.sparket/custom_miner/elo_ratings.json

# 2. Enable custom miner
export SPARKET_CUSTOM_MINER__ENABLED=true
export SPARKET_CUSTOM_MINER__ODDS_API_KEY=your_key_here

# 3. Run alongside validator
pm2 start ecosystem.config.js
```
