"""CustomMiner - Main orchestration for the custom miner.

This miner is optimized for the scoring system:
- 50% EconDim: Beat closing lines (CLV > 0)
- 30% InfoDim: Originality + lead market
- 20% Outcome accuracy (ForecastDim + SkillDim)

Key strategies:
1. Enhanced Elo model for probability estimation
2. Isotonic calibration for accuracy
3. Strategic timing for maximum time credit
4. Originality tracking for InfoDim
"""

from __future__ import annotations

import asyncio
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import bittensor as bt

from sparket.miner.base.engines.interface import OddsPrices
from sparket.miner.base.fetchers.espn import ESPNFetcher
from sparket.miner.custom.config import CustomMinerConfig
from sparket.miner.custom.data.fetchers.odds_api import OddsAPIFetcher, blend_with_market
from sparket.miner.custom.data.storage.line_history import LineHistory
from sparket.miner.custom.models.calibration.isotonic import IsotonicCalibrator
from sparket.miner.custom.models.engines.elo import EloEngine
from sparket.miner.custom.models.engines.ensemble import EnsembleEngine
from sparket.miner.custom.models.engines.poisson import PoissonEngine
from sparket.miner.custom.strategy.econ_tracker import EconTracker
from sparket.miner.custom.strategy.originality import OriginalityTracker
from sparket.miner.custom.strategy.timing import (
    SubmissionDecision,
    TimingStrategy,
)
from sparket.miner.utils.ratelimit import TokenBucket


class CustomMiner:
    """Custom miner implementation optimized for scoring.

    Features:
    - Elo-based probability model with sport-specific tuning
    - Isotonic calibration for probability accuracy
    - Strategic submission timing for maximum time credit
    - Rate limiting to respect validator limits

    Usage:
        config = CustomMinerConfig.from_env()
        miner = CustomMiner(
            hotkey="5xxx...",
            config=config,
            validator_client=client,
            game_sync=sync,
        )
        await miner.start()
    """

    def __init__(
        self,
        hotkey: str,
        config: CustomMinerConfig,
        validator_client: Any,
        game_sync: Any,
        data_dir: Optional[str] = None,
        get_token: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        """Initialize the custom miner.

        Args:
            hotkey: Miner's hotkey (ss58 address)
            config: Configuration
            validator_client: Client for submitting to validators
            game_sync: GameDataSync for fetching markets
            data_dir: Directory for persistent data (ratings, calibration)
            get_token: Callback to get the current validator push token
        """
        self.hotkey = hotkey
        self.config = config
        self.validator_client = validator_client
        self.game_sync = game_sync
        self._get_token = get_token

        # Set up data directory
        if data_dir:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = Path.home() / ".sparket" / "custom_miner"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._elo = EloEngine(
            config=config.elo,
            vig=config.vig,
            ratings_path=str(self._data_dir / "elo_ratings.json"),
        )

        self._calibrator = IsotonicCalibrator(
            min_samples=config.calibration.min_samples,
            data_path=(
                str(self._data_dir / "calibration.json")
                if config.calibration.enabled
                else None
            ),
        )

        # Poisson engine for TOTAL markets
        self._poisson = PoissonEngine(
            data_path=str(self._data_dir / "poisson_profiles.json"),
        )

        # Line history for tracking market movements
        self._line_history = LineHistory(
            data_path=str(self._data_dir / "line_history.json"),
            max_history_hours=168.0,  # 7 days
        )

        # Originality tracking for InfoDim optimization
        self._originality = OriginalityTracker(
            data_path=str(self._data_dir / "originality.json"),
            max_history_days=30.0,
        )

        # Economic performance tracking for EconDim feedback
        self._econ_tracker = EconTracker(
            data_path=str(self._data_dir / "econ_tracker.json"),
        )

        # Ensemble engine combining all models
        self._ensemble = EnsembleEngine(
            elo_engine=self._elo,
            poisson_engine=self._poisson,
            base_weights=config.engine_weights,
            confidence_scaling=True,
        )

        self._timing = TimingStrategy(config=config.timing)
        self._espn = ESPNFetcher()

        # The-Odds-API for market consensus (optional but recommended)
        self._odds_api: Optional[OddsAPIFetcher] = None
        if config.odds_api_key:
            self._odds_api = OddsAPIFetcher(
                api_key=config.odds_api_key,
                cache_ttl_seconds=300,  # 5 min cache
            )
            bt.logging.info({"custom_miner": "odds_api_enabled"})

        # Rate limiting
        self._global_bucket = TokenBucket(config.rate_limit_per_minute)
        self._per_market_buckets: Dict[int, TokenBucket] = {}

        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._submissions_count = 0
        self._errors_count = 0
        # Track submitted predictions for calibration training and CLV
        # Key: event_id, Value: (home_prob, home_odds_eu, timestamp)
        self._submitted_predictions: Dict[str, Tuple[float, float, float]] = {}
        # Track submitted market metadata for ESPN-based outcome fallback
        # Key: event_id, Value: dict with home_team, away_team, sport, start_time_utc, market_id
        self._submitted_events: Dict[str, Dict[str, Any]] = {}
        # Track event_ids whose outcomes have already been processed
        self._processed_outcomes: set = set()
        # Track previous market odds for recording market moves (lead ratio)
        self._previous_market_odds: Dict[int, float] = {}

    @property
    def is_running(self) -> bool:
        """Whether the miner is currently running."""
        return self._running

    @property
    def submissions_count(self) -> int:
        """Number of submissions made."""
        return self._submissions_count

    @property
    def errors_count(self) -> int:
        """Number of errors encountered."""
        return self._errors_count

    async def start(self) -> None:
        """Start the custom miner background loops."""
        if self._running:
            return

        self._running = True
        bt.logging.info({"custom_miner": "starting"})

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._odds_loop()),
            asyncio.create_task(self._outcome_loop()),
        ]

        bt.logging.info({
            "custom_miner": "started",
            "calibration_samples": self._calibrator.sample_count,
            "elo_ratings_path": str(self._data_dir / "elo_ratings.json"),
        })

    async def stop(self) -> None:
        """Stop the custom miner."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Cleanup
        await self._espn.close()
        if self._odds_api:
            await self._odds_api.close()

        bt.logging.info({
            "custom_miner": "stopped",
            "submissions": self._submissions_count,
            "errors": self._errors_count,
        })

    def start_background(self) -> None:
        """Start in background (non-blocking)."""
        asyncio.create_task(self.start())

    # -------------------------------------------------------------------------
    # Odds Generation Pipeline
    # -------------------------------------------------------------------------

    async def generate_odds(self, market: Dict[str, Any]) -> Optional[OddsPrices]:
        """Generate calibrated odds for a market using ensemble model.

        Pipeline:
        1. Fetch market consensus from The-Odds-API (if available)
        2. Use ensemble engine to combine Elo + Market + Poisson
        2b. Apply originality adjustment for InfoDim SOS
        3. Apply isotonic calibration
        4. Convert to odds with vig

        Args:
            market: Market info dict

        Returns:
            OddsPrices or None if unable to generate
        """
        home_team = market.get("home_team", "")
        away_team = market.get("away_team", "")
        sport = market.get("sport", "NFL")
        kind = market.get("kind", "MONEYLINE").upper()

        if not home_team or not away_team:
            return None

        # 1. Get market consensus from The-Odds-API
        market_odds = None
        if self._odds_api:
            try:
                market_odds = await self._odds_api.get_consensus_odds(
                    sport, home_team, away_team
                )
                if market_odds:
                    # Record market odds in line history for movement tracking
                    market_id = market.get("market_id")
                    if market_id:
                        self._line_history.record(
                            market_id=int(market_id),
                            home_prob=market_odds.home_prob,
                            away_prob=1.0 - market_odds.home_prob,
                            source="market",
                        )
                        # Record market move for lead ratio (InfoDim)
                        old_market_prob = self._previous_market_odds.get(int(market_id))
                        if old_market_prob is not None:
                            self._originality.record_market_move(
                                market_id=int(market_id),
                                old_prob=old_market_prob,
                                new_prob=market_odds.home_prob,
                            )
                        self._previous_market_odds[int(market_id)] = market_odds.home_prob
            except Exception as e:
                bt.logging.debug({
                    "custom_miner": "market_odds_error",
                    "error": str(e),
                })

        # 2. Use ensemble engine to combine all models
        ensemble_pred = self._ensemble.predict(
            market=market,
            market_odds=market_odds,
        )

        if ensemble_pred is None:
            return None

        cal_home = ensemble_pred.home_prob
        cal_away = ensemble_pred.away_prob

        # 2a. Variance-aware shrinkage toward market (improves EconDim ES)
        # When confidence is low, shrink toward market to reduce CLE variance.
        # ES = CLE_mean / CLE_std, so lower variance improves Sharpe ratio.
        shrinkage_applied = 0.0
        if market_odds is not None and ensemble_pred.confidence < 0.8:
            # Shrinkage factor: more shrinkage when confidence is lower
            # At confidence=0.5, shrink 30% toward market
            # At confidence=0.8, shrink 0% (no shrinkage)
            shrink_factor = 0.3 * (0.8 - ensemble_pred.confidence) / 0.3
            shrink_factor = max(0.0, min(0.5, shrink_factor))  # Cap at 50%

            # Only shrink if we differ significantly from market (>3%)
            diff_from_market = abs(cal_home - market_odds.home_prob)
            if diff_from_market > 0.03:
                old_home = cal_home
                cal_home = cal_home * (1 - shrink_factor) + market_odds.home_prob * shrink_factor
                cal_away = 1.0 - cal_home
                shrinkage_applied = shrink_factor

        # 2b. InfoDim: Elo-anchored differentiation for SOS
        # Use Elo's disagreement with market as differentiation direction.
        # Elo updates from game results/ratings (not live market), so it
        # naturally decorrelates our time series from market odds.
        diff_adjustment = 0.0
        diff_strength = 0.0
        market_consensus_prob = None
        if market_odds is not None:
            market_consensus_prob = market_odds.home_prob
        if market_consensus_prob is not None and shrinkage_applied == 0.0:
            diff_strength = self._originality.get_differentiation_strength()
            if diff_strength > 0.0:
                # Find Elo component from ensemble
                elo_comp = next(
                    (c for c in ensemble_pred.components if c.source == "elo"),
                    None,
                )
                if elo_comp and elo_comp.home_prob is not None:
                    # Elo's disagreement with market = our differentiation direction
                    elo_edge = elo_comp.home_prob - market_consensus_prob
                    # Scale: at full strength, shift up to 50% of Elo's edge
                    max_shift = 0.05  # Cap at 5% max adjustment
                    adjustment = elo_edge * diff_strength * 0.5
                    adjustment = max(-max_shift, min(max_shift, adjustment))
                    if abs(adjustment) > 0.005:  # Only apply if meaningful
                        diff_adjustment = adjustment
                        cal_home = max(0.01, min(0.99, cal_home + adjustment))
                        cal_away = 1.0 - cal_home

        # 3. Apply calibration if enabled and fitted
        if self.config.calibration.enabled and self._calibrator.is_fitted:
            cal_home, cal_away = self._calibrator.calibrate_pair(cal_home, cal_away)

        # 4. Convert to odds with vig
        home_odds = self._probability_to_odds(cal_home)
        away_odds = self._probability_to_odds(cal_away)

        # 5. Handle TOTAL market over/under
        over_prob = ensemble_pred.over_prob
        under_prob = ensemble_pred.under_prob
        over_odds = None
        under_odds = None

        if over_prob is not None and under_prob is not None:
            over_odds = self._probability_to_odds(over_prob)
            under_odds = self._probability_to_odds(under_prob)

        calibrated_odds = OddsPrices(
            home_prob=cal_home,
            away_prob=cal_away,
            home_odds_eu=home_odds,
            away_odds_eu=away_odds,
            over_prob=over_prob,
            under_prob=under_prob,
            over_odds_eu=over_odds,
            under_odds_eu=under_odds,
        )

        bt.logging.debug({
            "custom_miner": "ensemble_odds_generated",
            "market_id": market.get("market_id"),
            "kind": kind,
            "final_home_prob": round(cal_home, 3),
            "home_odds": home_odds,
            "ensemble_confidence": round(ensemble_pred.confidence, 3),
            "dominant_source": ensemble_pred.dominant_source,
            "models_agreed": ensemble_pred.models_agreed,
            "variance_shrinkage": round(shrinkage_applied, 3) if shrinkage_applied > 0 else None,
            "calibration_applied": self._calibrator.is_fitted,
            "diff_strength": round(diff_strength, 3) if diff_strength > 0 else None,
            "originality_adj": round(diff_adjustment, 4) if diff_adjustment != 0.0 else None,
        })

        return calibrated_odds

    def _probability_to_odds(self, prob: float) -> float:
        """Convert probability to EU decimal odds with vig.

        Validator bounds: odds_eu in (1.01, 1000], imp_prob in (0.001, 0.999)
        """
        implied_prob = prob + (self.config.vig / 2)
        # Clamp to validator-accepted range (0.001, 0.999)
        implied_prob = max(0.001, min(0.999, implied_prob))
        odds = 1.0 / implied_prob
        # Clamp odds to validator-accepted range (1.01, 1000]
        odds = max(1.01, min(1000.0, odds))
        return round(odds, 2)

    # -------------------------------------------------------------------------
    # Odds Loop
    # -------------------------------------------------------------------------

    async def _odds_loop(self) -> None:
        """Background loop for odds submission."""
        while self._running:
            min_sleep: Optional[int] = None
            try:
                min_sleep = await self._run_odds_cycle()
            except Exception as e:
                self._errors_count += 1
                bt.logging.warning({"custom_miner": "odds_cycle_error", "error": str(e)})

            # Adaptive sleep: use minimum next_check from cycle, floored by min_refresh
            if min_sleep is not None:
                sleep_seconds = max(self.config.timing.min_refresh_seconds, min_sleep)
            else:
                sleep_seconds = self.config.timing.refresh_interval_seconds
            await asyncio.sleep(sleep_seconds)

    def _record_batch_bookkeeping(
        self,
        batch: List[Tuple[Dict[str, Any], OddsPrices]],
    ) -> int:
        """Record post-submission bookkeeping for a batch of markets.

        Updates timing, line history, originality tracking, and calibration
        prediction storage for each successfully submitted market.

        Args:
            batch: List of (market, odds) tuples that were submitted

        Returns:
            Number of items processed
        """
        for market, odds in batch:
            market_id = market.get("market_id", 0)
            self._submissions_count += 1
            self._timing.record_submission(market_id)

            # Record our prediction in line history
            self._line_history.record(
                market_id=market_id,
                home_prob=odds.home_prob,
                away_prob=odds.away_prob,
                source="prediction",
            )

            # Record for originality tracking (InfoDim)
            market_consensus = self._line_history.get_consensus_prob(market_id)
            if market_consensus:
                market_home_prob, _ = market_consensus
                self._originality.record_submission(
                    market_id=market_id,
                    our_prob=odds.home_prob,
                    market_prob=market_home_prob,
                )

            # Store submitted probability, odds, and timestamp for calibration + CLV
            event_id = market.get("event_id")
            if event_id and odds.home_prob is not None:
                eid = str(event_id)
                self._submitted_predictions[eid] = (
                    odds.home_prob,
                    odds.home_odds_eu,
                    _time.time(),
                )
                # Store event metadata for ESPN-based outcome fallback
                if eid not in self._submitted_events:
                    self._submitted_events[eid] = {
                        "event_id": event_id,
                        "market_id": market.get("market_id"),
                        "home_team": market.get("home_team", ""),
                        "away_team": market.get("away_team", ""),
                        "sport": market.get("sport", "NFL"),
                        "start_time_utc": market.get("start_time_utc"),
                    }

        return len(batch)

    async def _run_odds_cycle(self) -> Optional[int]:
        """Run one odds submission cycle with timing strategy and batching.

        Returns:
            Minimum next_check_seconds from timing decisions, or None if no markets.
        """
        # Cleanup stale rate limit buckets to prevent memory buildup
        self._cleanup_stale_buckets()

        # Cleanup old line history (markets that finished)
        self._line_history.cleanup_old_markets()

        # Cleanup stale submitted predictions (14-day TTL)
        self._cleanup_stale_predictions()

        # Get active markets
        markets = await self.game_sync.get_active_markets()

        if not markets:
            bt.logging.debug({"custom_miner": "no_active_markets"})
            return None

        # Prune previous market odds to active markets only
        active_ids = {int(m.get("market_id", 0)) for m in markets}
        stale_odds = [k for k in self._previous_market_odds if k not in active_ids]
        for k in stale_odds:
            del self._previous_market_odds[k]

        # Enrich markets with event data for timing
        enriched_markets = await self._enrich_markets(markets)

        # Get prioritized submission schedule
        schedule = self._timing.get_submission_schedule(enriched_markets)

        if not schedule:
            bt.logging.debug({"custom_miner": "no_markets_due_for_submission"})
            return None

        # Track minimum next_check from timing decisions for adaptive sleep
        min_next_check: Optional[int] = None
        for _, decision in schedule:
            if decision.next_check_seconds > 0:
                if min_next_check is None or decision.next_check_seconds < min_next_check:
                    min_next_check = decision.next_check_seconds

        submitted = 0
        skipped = 0
        deferred = 0
        batches = 0
        batch: List[Tuple[Dict[str, Any], OddsPrices]] = []

        for market, decision in schedule:
            market_id = market.get("market_id", 0)

            # Check rate limits
            if not self._check_rate_limit(market_id):
                skipped += 1
                continue

            try:
                odds = await self.generate_odds(market)
                if odds is None:
                    continue

                # Check line movement for optimal timing
                start_time = market.get("start_time_utc")
                if start_time:
                    now = datetime.now(timezone.utc)
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(
                            start_time.replace("Z", "+00:00")
                        )
                    hours_to_game = (start_time - now).total_seconds() / 3600

                    should_submit, reason = self._line_history.should_submit_now(
                        market_id=market_id,
                        our_home_prob=odds.home_prob,
                        hours_to_game=hours_to_game,
                    )

                    if not should_submit:
                        deferred += 1
                        bt.logging.debug({
                            "custom_miner": "submission_deferred",
                            "market_id": market_id,
                            "reason": reason,
                            "hours_to_game": round(hours_to_game, 1),
                        })
                        continue

                batch.append((market, odds))

                # Submit when batch is full
                if len(batch) >= self.config.batch_size:
                    try:
                        payload = self._build_batch_payload(batch)
                        success = await self.validator_client.submit_odds(payload)
                        if success:
                            submitted += self._record_batch_bookkeeping(batch)
                    except Exception as e:
                        bt.logging.warning({
                            "custom_miner": "batch_submission_failed",
                            "batch_size": len(batch),
                            "error": str(e),
                        })
                    batches += 1
                    batch = []

            except Exception as e:
                bt.logging.debug({
                    "custom_miner": "market_odds_generation_failed",
                    "market_id": market.get("market_id"),
                    "error": str(e),
                })

        # Submit remaining batch
        if batch:
            try:
                payload = self._build_batch_payload(batch)
                success = await self.validator_client.submit_odds(payload)
                if success:
                    submitted += self._record_batch_bookkeeping(batch)
                batches += 1
            except Exception as e:
                bt.logging.warning({
                    "custom_miner": "final_batch_failed",
                    "batch_size": len(batch),
                    "error": str(e),
                })

        if submitted > 0 or skipped > 0 or deferred > 0:
            bt.logging.info({
                "custom_miner": "odds_cycle_complete",
                "submitted": submitted,
                "skipped_rate_limit": skipped,
                "deferred_line_movement": deferred,
                "batches": batches,
                "total_markets": len(schedule),
            })

        return min_next_check

    async def _enrich_markets(
        self,
        markets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Enrich market data with event info for timing strategy."""
        # Markets from game_sync should already have event data
        # Just ensure datetime fields are properly parsed
        enriched = []
        for market in markets:
            m = dict(market)

            # Ensure start_time_utc is a datetime
            start_time = m.get("start_time_utc")
            if isinstance(start_time, str):
                m["start_time_utc"] = datetime.fromisoformat(
                    start_time.replace("Z", "+00:00")
                )

            enriched.append(m)

        return enriched

    def _check_rate_limit(self, market_id: int) -> bool:
        """Check rate limits for a market."""
        # Global rate limit
        if not self._global_bucket.allow():
            return False

        # Per-market rate limit
        bucket = self._per_market_buckets.setdefault(
            market_id,
            TokenBucket(self.config.per_market_limit_per_minute),
        )
        return bucket.allow()

    def _cleanup_stale_buckets(self, max_age_seconds: float = 3600.0) -> int:
        """Remove rate limit buckets that haven't been used recently.

        This prevents memory buildup for long-running miners with many markets.

        Args:
            max_age_seconds: Remove buckets not used in this many seconds (default: 1 hour)

        Returns:
            Number of buckets removed
        """
        now = _time.time()
        stale_ids = [
            market_id
            for market_id, bucket in self._per_market_buckets.items()
            if (now - bucket.last_refill) > max_age_seconds
        ]

        for market_id in stale_ids:
            del self._per_market_buckets[market_id]

        if stale_ids:
            bt.logging.debug({
                "custom_miner": "rate_limit_cleanup",
                "removed": len(stale_ids),
                "remaining": len(self._per_market_buckets),
            })

        return len(stale_ids)

    def _cleanup_stale_predictions(self, max_age_seconds: float = 14 * 86400) -> int:
        """Remove submitted prediction entries older than max_age_seconds.

        Prevents unbounded memory growth when the outcome loop fails
        to process events (e.g., database unavailable, ESPN lookup miss).

        Also cleans up related _submitted_events and _processed_outcomes.

        Args:
            max_age_seconds: Remove entries older than this (default: 14 days)

        Returns:
            Number of entries removed
        """
        now = _time.time()
        stale_keys = [
            key
            for key, (_, _, ts) in self._submitted_predictions.items()
            if (now - ts) > max_age_seconds
        ]

        for key in stale_keys:
            del self._submitted_predictions[key]
            self._submitted_events.pop(key, None)

        # Also prune _submitted_events that have no prediction (orphaned)
        orphaned = [k for k in self._submitted_events if k not in self._submitted_predictions]
        for key in orphaned:
            del self._submitted_events[key]

        # Cap _processed_outcomes to prevent unbounded growth (keep last 10k)
        if len(self._processed_outcomes) > 10000:
            # Keep only outcomes that still have active predictions
            active_ids = set(self._submitted_predictions.keys())
            self._processed_outcomes &= active_ids
            # If still too large after intersection, keep a bounded size
            if len(self._processed_outcomes) > 5000:
                sorted_ids = sorted(self._processed_outcomes)
                self._processed_outcomes = set(sorted_ids[-5000:])

        removed = len(stale_keys) + len(orphaned)
        if removed:
            bt.logging.debug({
                "custom_miner": "stale_predictions_cleanup",
                "removed_predictions": len(stale_keys),
                "removed_orphaned_events": len(orphaned),
                "remaining_predictions": len(self._submitted_predictions),
            })

        return removed

    def _build_prices(self, market: Dict[str, Any], odds: OddsPrices) -> List[Dict[str, Any]]:
        """Build prices list for a single market submission."""
        kind = market.get("kind", "MONEYLINE").upper()

        if kind in ("MONEYLINE", "SPREAD"):
            return [
                {"side": "home", "odds_eu": odds.home_odds_eu, "imp_prob": odds.home_prob},
                {"side": "away", "odds_eu": odds.away_odds_eu, "imp_prob": odds.away_prob},
            ]
        elif kind == "TOTAL":
            return [
                {"side": "over", "odds_eu": odds.over_odds_eu or odds.home_odds_eu, "imp_prob": odds.over_prob or odds.home_prob},
                {"side": "under", "odds_eu": odds.under_odds_eu or odds.away_odds_eu, "imp_prob": odds.under_prob or odds.away_prob},
            ]
        return []

    def _build_batch_payload(
        self,
        markets_with_odds: List[Tuple[Dict[str, Any], OddsPrices]],
    ) -> Dict[str, Any]:
        """Build batched odds submission payload.

        Args:
            markets_with_odds: List of (market, odds) tuples

        Returns:
            Payload dict with multiple submissions and optional token
        """
        now = datetime.now(timezone.utc)

        submissions = []
        for market, odds in markets_with_odds:
            kind = market.get("kind", "MONEYLINE").upper()
            submissions.append({
                "market_id": int(market.get("market_id", 0)),
                "kind": kind.lower(),
                "priced_at": now.isoformat(),
                "prices": self._build_prices(market, odds),
            })

        payload: Dict[str, Any] = {
            "miner_hotkey": self.hotkey,
            "submissions": submissions,
        }

        # Include token for authentication
        if self._get_token is not None:
            token = self._get_token()
            if token:
                payload["token"] = token

        return payload

    def _build_odds_payload(
        self,
        market: Dict[str, Any],
        odds: OddsPrices,
    ) -> Dict[str, Any]:
        """Build odds submission payload for a single market."""
        return self._build_batch_payload([(market, odds)])

    # -------------------------------------------------------------------------
    # Outcome Loop
    # -------------------------------------------------------------------------

    async def _outcome_loop(self) -> None:
        """Background loop for outcome detection and submission."""
        while self._running:
            try:
                await self._run_outcome_cycle()
            except Exception as e:
                self._errors_count += 1
                bt.logging.warning({"custom_miner": "outcome_cycle_error", "error": str(e)})

            await asyncio.sleep(self.config.outcome_check_seconds)

    async def _run_outcome_cycle(self) -> None:
        """Run one outcome detection cycle."""
        # Get potentially finished events
        events = await self._get_finished_events()

        if not events:
            return

        submitted = 0
        for event in events:
            try:
                result = await self._espn.get_result(event)
                if result is None or not result.is_final:
                    continue

                # Update Elo ratings with result
                self._update_ratings_from_result(event, result)

                # Update calibration with prediction vs outcome
                self._update_calibration_from_result(event, result)

                # Submit outcome
                payload = self._build_outcome_payload(event, result)
                success = await self.validator_client.submit_outcome(payload)

                if success:
                    submitted += 1

            except Exception as e:
                bt.logging.debug({
                    "custom_miner": "outcome_submission_failed",
                    "event_id": event.get("event_id"),
                    "error": str(e),
                })

        if submitted > 0:
            bt.logging.info({"custom_miner": "outcomes_submitted", "count": submitted})

    async def _get_finished_events(self) -> List[Dict[str, Any]]:
        """Get events that might be finished.

        Tries the database repository first (most complete data).
        Falls back to in-memory tracked events filtered by start time,
        so outcomes can still be detected via ESPN even when the
        database is unavailable.
        """
        # Try database first
        try:
            from sparket.miner.database.repository import get_past_events

            if hasattr(self.game_sync, 'database') and self.game_sync.database is not None:
                db_events = await asyncio.wait_for(
                    get_past_events(self.game_sync.database), timeout=10.0,
                )
                if db_events:
                    return db_events
        except ImportError:
            pass
        except Exception as e:
            bt.logging.debug({
                "custom_miner": "db_outcome_lookup_failed",
                "error": str(e),
            })

        # Fallback: use in-memory tracked events whose start_time is in the past
        now = datetime.now(timezone.utc)
        candidates = []
        for eid, meta in self._submitted_events.items():
            if eid in self._processed_outcomes:
                continue

            start_time = meta.get("start_time_utc")
            if start_time is None:
                continue

            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                except ValueError:
                    continue

            if hasattr(start_time, 'tzinfo') and start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)

            # Only check events that started at least 2 hours ago (likely finished)
            # and no more than 48 hours ago (avoid very old events)
            hours_since = (now - start_time).total_seconds() / 3600
            if 2.0 <= hours_since <= 48.0:
                candidates.append(meta)

        if candidates:
            bt.logging.debug({
                "custom_miner": "outcome_fallback_candidates",
                "count": len(candidates),
            })

        return candidates

    def _update_ratings_from_result(
        self,
        event: Dict[str, Any],
        result: Any,
    ) -> None:
        """Update Elo and Poisson models after a game result."""
        try:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            sport = event.get("sport", "NFL")
            home_score = result.home_score or 0
            away_score = result.away_score or 0

            if home_team and away_team:
                # Update Elo ratings
                self._elo.update_ratings(
                    home_team=home_team,
                    away_team=away_team,
                    sport=sport,
                    home_score=home_score,
                    away_score=away_score,
                )

                # Update Poisson model with scoring data
                self._poisson.update_from_result(
                    home_team=home_team,
                    away_team=away_team,
                    sport=sport,
                    home_score=home_score,
                    away_score=away_score,
                )
        except Exception as e:
            bt.logging.debug({
                "custom_miner": "model_update_failed",
                "event_id": event.get("event_id"),
                "error": str(e),
            })

    def _update_calibration_from_result(
        self,
        event: Dict[str, Any],
        result: Any,
    ) -> None:
        """Update calibration with prediction vs actual outcome.

        IMPORTANT: We train on the actual submitted probability (blended + calibrated),
        not the raw Elo probability. This ensures the calibrator learns to correct
        the predictions we actually make.
        """
        if not self.config.calibration.enabled:
            return

        try:
            event_id = str(event.get("event_id", ""))

            # Look up the probability we actually submitted for this event
            submitted_record = self._submitted_predictions.get(event_id)
            if submitted_record is None:
                # We didn't submit a prediction for this event, skip calibration
                bt.logging.debug({
                    "custom_miner": "calibration_skip_no_submission",
                    "event_id": event_id,
                })
                return

            submitted_prob, submitted_odds, submit_ts = submitted_record

            # Actual outcome (1 = home won, 0 = away won)
            # Normalize winner to uppercase for comparison
            winner = str(result.winner).upper() if result.winner else ""
            if winner in ("HOME", "H", "1"):
                actual = 1.0
            elif winner in ("AWAY", "A", "2"):
                actual = 0.0
            else:
                # Skip unknown/invalid outcomes - they provide no training signal
                bt.logging.debug({
                    "custom_miner": "calibration_skip_unknown_outcome",
                    "event_id": event_id,
                    "winner": winner,
                })
                return

            # Add calibration sample using the SUBMITTED probability
            self._calibrator.add_sample(
                predicted=submitted_prob,
                actual=actual,
                market_id=event.get("market_id"),
                sport=event.get("sport"),
            )

            # Compute CLV for observational logging (EconDim visibility)
            # Uses same formulas as validator: sparket/validator/scoring/metrics/clv.py
            market_id = event.get("market_id")
            if market_id:
                closing_consensus = self._line_history.get_consensus_prob(market_id)
                if closing_consensus:
                    closing_home_prob, _ = closing_consensus
                    closing_odds = 1.0 / max(0.001, closing_home_prob)
                    # CLE = miner_odds * truth_prob - 1.0
                    cle = submitted_odds * closing_home_prob - 1.0
                    cle = max(-1.0, min(10.0, cle))
                    # CLV_prob = (truth_prob - miner_prob) / truth_prob
                    clv_prob = (closing_home_prob - submitted_prob) / closing_home_prob if closing_home_prob > 0 else 0.0
                    # Record to econ tracker for rolling stats
                    self._econ_tracker.record(
                        market_id=int(market_id),
                        cle=cle,
                        clv_prob=clv_prob,
                    )

                    # Get rolling stats for logging
                    econ_stats = self._econ_tracker.get_stats()

                    bt.logging.info({
                        "custom_miner": "clv_observation",
                        "event_id": event_id,
                        "market_id": market_id,
                        "submitted_prob": round(submitted_prob, 4),
                        "submitted_odds": round(submitted_odds, 2),
                        "closing_prob": round(closing_home_prob, 4),
                        "closing_odds": round(closing_odds, 2),
                        "cle": round(cle, 4),
                        "clv_prob": round(clv_prob, 4),
                        "rolling_cle_mean": round(econ_stats["cle_mean"], 4),
                        "rolling_es_sharpe": round(econ_stats["es_sharpe"], 3),
                        "econ_samples": econ_stats["n_samples"],
                    })

            # Clean up stored prediction and mark outcome as processed
            del self._submitted_predictions[event_id]
            self._submitted_events.pop(event_id, None)
            self._processed_outcomes.add(event_id)

            # Periodically refit calibration
            if self._calibrator.sample_count % self.config.calibration.retrain_interval == 0:
                self._calibrator.fit()
                bt.logging.info({
                    "custom_miner": "calibration_refitted",
                    "samples": self._calibrator.sample_count,
                })

        except Exception as e:
            bt.logging.debug({
                "custom_miner": "calibration_update_failed",
                "event_id": event.get("event_id"),
                "error": str(e),
            })

    def _build_outcome_payload(
        self,
        event: Dict[str, Any],
        result: Any,
    ) -> Dict[str, Any]:
        """Build outcome submission payload."""
        now = datetime.now(timezone.utc)

        payload: Dict[str, Any] = {
            "event_id": int(event.get("event_id", 0)),
            "miner_hotkey": self.hotkey,
            "result": result.winner,
            "score_home": result.home_score,
            "score_away": result.away_score,
            "ts_submit": now.isoformat(),
        }

        # Include token for authentication
        if self._get_token is not None:
            token = self._get_token()
            if token:
                payload["token"] = token

        return payload

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get miner diagnostics and stats."""
        econ_stats = self._econ_tracker.get_stats()
        return {
            "running": self._running,
            "submissions_count": self._submissions_count,
            "errors_count": self._errors_count,
            "calibration": self._calibrator.get_calibration_stats(),
            "line_history": self._line_history.stats(),
            "recent_steam_moves": len(self._line_history.get_recent_steam_moves(24.0)),
            "originality": self._originality.stats(),
            "econ_dim": {
                "cle_mean": round(econ_stats["cle_mean"], 4),
                "cle_std": round(econ_stats["cle_std"], 4),
                "es_sharpe": round(econ_stats["es_sharpe"], 3),
                "mes_mean": round(econ_stats["mes_mean"], 3),
                "samples": econ_stats["n_samples"],
            },
            "ensemble": self._ensemble.get_model_stats(),
            "odds_api": {
                "enabled": self._odds_api is not None,
                "remaining_requests": (
                    self._odds_api.remaining_requests
                    if self._odds_api else None
                ),
            },
            "config": {
                "vig": self.config.vig,
                "refresh_interval": self.config.timing.refresh_interval_seconds,
                "calibration_enabled": self.config.calibration.enabled,
                "market_blend_weight": self.config.engine_weights.get("market", 0.6),
                "batch_size": self.config.batch_size,
            },
        }


async def main() -> None:
    """Standalone entry point for the custom miner.

    Usage:
        python -m sparket.miner.custom.runner

    Or:
        python sparket/miner/custom/runner.py
    """
    import os
    import sys

    # Load configuration
    config = CustomMinerConfig.from_env()

    if not config.enabled:
        print("Custom miner is disabled. Set SPARKET_CUSTOM_MINER__ENABLED=true")
        sys.exit(0)

    # Import bittensor components
    try:
        import bittensor as bt
        from sparket.miner.client import ValidatorClient
        from sparket.miner.sync import GameDataSync
        from sparket.miner.database.dbm import DBM
    except ImportError as e:
        print(f"Failed to import dependencies: {e}")
        print("Run: uv sync --dev")
        sys.exit(1)

    # Initialize bittensor wallet and metagraph
    bt.logging.info("Initializing custom miner...")

    # Get wallet from environment or defaults
    wallet_name = os.getenv("BT_WALLET_NAME", "default")
    wallet_hotkey = os.getenv("BT_WALLET_HOTKEY", "default")
    netuid = int(os.getenv("SPARKET_NETUID", "1"))

    wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
    subtensor = bt.subtensor()
    metagraph = subtensor.metagraph(netuid=netuid)

    # Initialize database using miner config
    from sparket.miner.config.config import Config as MinerAppConfig
    from sparket.miner.database import initialize as init_db

    app_config = MinerAppConfig()

    # Initialize database schema
    try:
        init_db(app_config)
    except Exception as e:
        bt.logging.warning(f"Database init warning: {e}")

    # Create database manager
    dbm = DBM(app_config)

    # Create validator client and game sync
    client = ValidatorClient(wallet=wallet, metagraph=metagraph)
    sync = GameDataSync(database=dbm, client=client)

    # Start sync
    await sync.start()

    # Create and start custom miner
    miner = CustomMiner(
        hotkey=wallet.hotkey.ss58_address,
        config=config,
        validator_client=client,
        game_sync=sync,
    )

    await miner.start()

    # Run until interrupted
    bt.logging.info("Custom miner running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(60)
            # Log diagnostics every minute
            diag = miner.get_diagnostics()
            bt.logging.debug({"custom_miner": "diagnostics", **diag})
    except KeyboardInterrupt:
        bt.logging.info("Shutting down...")
    finally:
        await miner.stop()
        await sync.stop()


if __name__ == "__main__":
    asyncio.run(main())
