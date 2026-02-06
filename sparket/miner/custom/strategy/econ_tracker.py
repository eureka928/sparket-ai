"""Economic performance tracker for EconDim optimization.

Tracks CLE (Closing Line Edge) performance to provide feedback on
how well predictions are beating closing lines. This directly
corresponds to the validator's EconDim scoring (50% of total score).

EconDim formula:
- ES = CLE_mean / CLE_std (Sharpe ratio)
- MES = 1 - min(1, |CLV_prob|)
- EconDim = 0.7 * ES + 0.3 * MES
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CLERecord:
    """Single CLE observation."""

    market_id: int
    cle: float  # Closing Line Edge: miner_odds * closing_prob - 1
    clv_prob: float  # CLV probability: (closing_prob - miner_prob) / closing_prob
    timestamp: float  # Unix timestamp


class EconTracker:
    """Track CLE/CLV performance for EconDim feedback.

    Uses the same 10-day decay half-life as the validator to compute
    rolling statistics that approximate what the validator sees.

    Usage:
        tracker = EconTracker()

        # After each outcome is known
        tracker.record(market_id, cle, clv_prob)

        # Get current performance estimate
        stats = tracker.get_stats()
        print(f"CLE Sharpe: {stats['es_sharpe']:.3f}")
    """

    # Validator uses 10-day half-life decay
    DECAY_HALF_LIFE_DAYS = 10.0
    # Keep 30 days of history (matches validator's rolling window)
    MAX_AGE_DAYS = 30.0

    def __init__(
        self,
        data_path: Optional[str] = None,
    ) -> None:
        """Initialize the tracker.

        Args:
            data_path: Optional path to persist data
        """
        self._data_path = Path(data_path) if data_path else None
        self._records: List[CLERecord] = []

        if self._data_path:
            self._load()

    def record(
        self,
        market_id: int,
        cle: float,
        clv_prob: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a CLE observation.

        Args:
            market_id: Market identifier
            cle: Closing Line Edge (miner_odds * closing_prob - 1)
            clv_prob: CLV probability ((closing_prob - miner_prob) / closing_prob)
            timestamp: Unix timestamp (defaults to now)
        """
        ts = timestamp or datetime.now(timezone.utc).timestamp()

        # Clamp CLE to validator range [-1, 10]
        cle = max(-1.0, min(10.0, cle))

        self._records.append(CLERecord(
            market_id=market_id,
            cle=cle,
            clv_prob=clv_prob,
            timestamp=ts,
        ))

        # Prune old records
        self._prune_old()
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Get rolling CLE statistics.

        Returns dict with:
        - cle_mean: Decay-weighted mean CLE
        - cle_std: Decay-weighted std of CLE
        - es_sharpe: CLE Sharpe ratio (mean/std) - approximates ES
        - mes_mean: Mean MES score
        - n_samples: Number of samples
        - n_eff: Effective sample count after decay
        """
        if not self._records:
            return {
                "cle_mean": 0.0,
                "cle_std": 0.0,
                "es_sharpe": 0.0,
                "mes_mean": 0.5,
                "n_samples": 0,
                "n_eff": 0.0,
            }

        now = datetime.now(timezone.utc).timestamp()

        # Compute decay weights
        weights = []
        cle_values = []
        clv_values = []

        for r in self._records:
            age_days = (now - r.timestamp) / 86400
            # Exponential decay: weight = 0.5^(age / half_life)
            weight = 0.5 ** (age_days / self.DECAY_HALF_LIFE_DAYS)
            weights.append(weight)
            cle_values.append(r.cle)
            clv_values.append(r.clv_prob)

        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0

        # Weighted mean
        cle_mean = sum(w * c for w, c in zip(weights, cle_values)) / total_weight

        # Weighted variance
        var_sum = sum(w * (c - cle_mean) ** 2 for w, c in zip(weights, cle_values))
        cle_var = var_sum / total_weight
        cle_std = math.sqrt(max(0.0, cle_var))

        # Sharpe ratio (ES approximation)
        es_sharpe = cle_mean / cle_std if cle_std > 0.001 else 0.0

        # MES: 1 - min(1, |clv_prob|)
        mes_values = [1.0 - min(1.0, abs(clv)) for clv in clv_values]
        mes_mean = sum(w * m for w, m in zip(weights, mes_values)) / total_weight

        # Log-scaled shrinkage matching validator (k=200)
        SHRINKAGE_K = 200.0
        n_eff = total_weight  # sum of decay weights = effective sample size
        if n_eff > 0:
            shrink_weight = math.log(1 + n_eff) / math.log(1 + n_eff + SHRINKAGE_K)
        else:
            shrink_weight = 0.0

        # Shrink toward neutral defaults (no population available)
        es_sharpe = shrink_weight * es_sharpe  # neutral = 0.0
        mes_mean = shrink_weight * mes_mean + (1 - shrink_weight) * 0.5  # neutral = 0.5

        return {
            "cle_mean": cle_mean,
            "cle_std": cle_std,
            "es_sharpe": es_sharpe,
            "mes_mean": mes_mean,
            "n_samples": len(self._records),
            "n_eff": total_weight,
        }

    def get_econ_dim_estimate(self) -> float:
        """Estimate EconDim score.

        EconDim = 0.7 * ES_norm + 0.3 * MES_norm

        Since we don't have population stats for normalization,
        this returns a rough estimate based on raw values.
        """
        stats = self.get_stats()

        # ES normalization: logistic transform of Sharpe
        # Typical Sharpe range is -1 to +1 for betting
        es_raw = stats["es_sharpe"]
        es_norm = 1.0 / (1.0 + math.exp(-1.0 * es_raw))

        # MES is already in [0, 1]
        mes_norm = max(0.0, min(1.0, stats["mes_mean"]))

        return 0.7 * es_norm + 0.3 * mes_norm

    def _prune_old(self) -> None:
        """Remove records older than MAX_AGE_DAYS."""
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - (self.MAX_AGE_DAYS * 86400)
        self._records = [r for r in self._records if r.timestamp >= cutoff]

    def _load(self) -> None:
        """Load records from disk."""
        if not self._data_path or not self._data_path.exists():
            return

        try:
            with open(self._data_path) as f:
                data = json.load(f)

            self._records = [
                CLERecord(
                    market_id=r["market_id"],
                    cle=r["cle"],
                    clv_prob=r["clv_prob"],
                    timestamp=r["timestamp"],
                )
                for r in data.get("records", [])
            ]
            self._prune_old()
        except (json.JSONDecodeError, KeyError):
            pass

    def _save(self) -> None:
        """Save records to disk."""
        if not self._data_path:
            return

        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "records": [
                {
                    "market_id": r.market_id,
                    "cle": r.cle,
                    "clv_prob": r.clv_prob,
                    "timestamp": r.timestamp,
                }
                for r in self._records[-5000:]  # Keep last 5k records
            ],
        }

        with open(self._data_path, "w") as f:
            json.dump(data, f, indent=2)

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()
        self._save()
