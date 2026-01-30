"""Isotonic regression calibration for probability predictions.

Calibration ensures that when we predict 70% probability,
the outcome actually occurs ~70% of the time. This is crucial
for the ForecastDim scoring component (Brier score, calibration slope).
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CalibrationSample:
    """A single calibration sample: predicted probability and actual outcome."""

    predicted: float  # Our predicted probability (0-1)
    actual: float  # Actual outcome (1.0 = happened, 0.0 = didn't)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    market_id: Optional[int] = None
    sport: Optional[str] = None


class IsotonicCalibrator:
    """Isotonic regression calibrator for probability predictions.

    Isotonic regression fits a non-decreasing step function to the
    (predicted, actual) data, ensuring calibrated outputs.

    Key properties:
    - Monotonic: higher raw predictions always map to higher calibrated
    - Non-parametric: adapts to any calibration curve shape
    - Optimal for probability calibration under squared loss

    Usage:
        calibrator = IsotonicCalibrator()

        # Add historical predictions and outcomes
        calibrator.add_sample(predicted=0.65, actual=1.0)
        calibrator.add_sample(predicted=0.70, actual=0.0)
        ...

        # Fit the calibration function
        calibrator.fit()

        # Calibrate new predictions
        raw_prob = 0.68
        calibrated = calibrator.calibrate(raw_prob)
    """

    def __init__(
        self,
        min_samples: int = 100,
        data_path: Optional[str] = None,
    ) -> None:
        """Initialize the calibrator.

        Args:
            min_samples: Minimum samples before calibration activates
            data_path: Path to persist calibration data
        """
        self.min_samples = min_samples
        self._data_path = Path(data_path) if data_path else None
        self._samples: List[CalibrationSample] = []
        self._fitted = False

        # Isotonic regression output: sorted (x, y) pairs
        self._calibration_x: List[float] = []
        self._calibration_y: List[float] = []

        # Load existing data
        if self._data_path:
            self._load()

    @property
    def sample_count(self) -> int:
        """Number of calibration samples collected."""
        return len(self._samples)

    @property
    def is_fitted(self) -> bool:
        """Whether calibration has been fitted."""
        return self._fitted and len(self._calibration_x) > 0

    def add_sample(
        self,
        predicted: float,
        actual: float,
        market_id: Optional[int] = None,
        sport: Optional[str] = None,
    ) -> None:
        """Add a calibration sample.

        Args:
            predicted: Our predicted probability (0-1)
            actual: Actual outcome (1.0 = happened, 0.0 = didn't)
            market_id: Optional market identifier
            sport: Optional sport/league
        """
        sample = CalibrationSample(
            predicted=max(0.0, min(1.0, predicted)),
            actual=1.0 if actual > 0.5 else 0.0,
            market_id=market_id,
            sport=sport,
        )
        self._samples.append(sample)

        # Invalidate fit when new data arrives
        if self._fitted:
            self._fitted = False

    def fit(self) -> bool:
        """Fit the isotonic regression model.

        Returns:
            True if fitting succeeded, False if not enough samples
        """
        if len(self._samples) < self.min_samples:
            return False

        # Extract predictions and outcomes
        data = [(s.predicted, s.actual) for s in self._samples]

        # Sort by predicted probability
        data.sort(key=lambda x: x[0])

        # Run Pool Adjacent Violators (PAV) algorithm
        self._calibration_x, self._calibration_y = self._pav_algorithm(data)

        self._fitted = True
        self._save()
        return True

    def _pav_algorithm(
        self,
        data: List[Tuple[float, float]],
    ) -> Tuple[List[float], List[float]]:
        """Pool Adjacent Violators algorithm for isotonic regression.

        This finds the monotonic non-decreasing function that minimizes
        squared error to the data points.

        Args:
            data: List of (predicted, actual) tuples, sorted by predicted

        Returns:
            Tuple of (x_values, y_values) for the fitted step function
        """
        if not data:
            return [], []

        # Initialize blocks: each point starts as its own block
        # Block = (sum_y, count, min_x, max_x)
        blocks: List[List[float]] = []

        for x, y in data:
            # Create new block
            blocks.append([y, 1.0, x, x])  # sum_y, count, min_x, max_x

            # Pool adjacent violators
            while len(blocks) > 1:
                # Check if last two blocks violate monotonicity
                b1 = blocks[-2]
                b2 = blocks[-1]
                mean1 = b1[0] / b1[1]
                mean2 = b2[0] / b2[1]

                if mean1 <= mean2:
                    # No violation
                    break

                # Merge blocks
                blocks.pop()
                blocks.pop()
                merged = [
                    b1[0] + b2[0],  # sum_y
                    b1[1] + b2[1],  # count
                    b1[2],  # min_x
                    b2[3],  # max_x
                ]
                blocks.append(merged)

        # Extract step function
        x_values = []
        y_values = []

        for block in blocks:
            mean_y = block[0] / block[1]
            min_x = block[2]
            max_x = block[3]

            # Add left and right endpoints of block
            if not x_values or x_values[-1] < min_x:
                x_values.append(min_x)
                y_values.append(mean_y)
            if min_x < max_x:
                x_values.append(max_x)
                y_values.append(mean_y)

        return x_values, y_values

    def calibrate(self, probability: float) -> float:
        """Calibrate a raw probability prediction.

        Uses linear interpolation between fitted step function points.

        Args:
            probability: Raw predicted probability (0-1)

        Returns:
            Calibrated probability
        """
        if not self.is_fitted:
            # Return raw probability if not fitted
            return probability

        # Clamp input
        prob = max(0.0, min(1.0, probability))

        # Handle edge cases
        if not self._calibration_x:
            return prob
        if prob <= self._calibration_x[0]:
            return self._calibration_y[0]
        if prob >= self._calibration_x[-1]:
            return self._calibration_y[-1]

        # Binary search for bracket
        idx = bisect.bisect_left(self._calibration_x, prob)

        if idx == 0:
            return self._calibration_y[0]
        if idx >= len(self._calibration_x):
            return self._calibration_y[-1]

        # Linear interpolation
        x0, x1 = self._calibration_x[idx - 1], self._calibration_x[idx]
        y0, y1 = self._calibration_y[idx - 1], self._calibration_y[idx]

        if x1 == x0:
            return y0

        t = (prob - x0) / (x1 - x0)
        calibrated = y0 + t * (y1 - y0)

        # Clamp to validator-accepted range (0.001, 0.999)
        return max(0.001, min(0.999, calibrated))

    def calibrate_pair(
        self,
        home_prob: float,
        away_prob: float,
    ) -> Tuple[float, float]:
        """Calibrate a pair of probabilities, maintaining sum = 1.

        Args:
            home_prob: Raw home win probability
            away_prob: Raw away win probability

        Returns:
            Tuple of (calibrated_home, calibrated_away)
        """
        cal_home = self.calibrate(home_prob)
        cal_away = self.calibrate(away_prob)

        # Normalize to sum to 1
        total = cal_home + cal_away
        if total > 0:
            cal_home /= total
            cal_away /= total

        return cal_home, cal_away

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration diagnostics.

        Returns dict with:
        - sample_count: Number of samples
        - brier_score: Mean squared error (lower is better)
        - calibration_slope: Slope of calibration curve (1.0 is perfect)
        - reliability_diagram: Bucketed (expected, observed) pairs
        """
        if not self._samples:
            return {
                "sample_count": 0,
                "brier_score": None,
                "calibration_slope": None,
                "reliability_diagram": [],
            }

        # Calculate Brier score
        brier = sum(
            (s.predicted - s.actual) ** 2 for s in self._samples
        ) / len(self._samples)

        # Calculate calibration slope using least squares
        # y = mx + b where y=actual, x=predicted
        n = len(self._samples)
        sum_x = sum(s.predicted for s in self._samples)
        sum_y = sum(s.actual for s in self._samples)
        sum_xy = sum(s.predicted * s.actual for s in self._samples)
        sum_xx = sum(s.predicted ** 2 for s in self._samples)

        denom = n * sum_xx - sum_x ** 2
        slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 1.0

        # Build reliability diagram (10 bins)
        bins: Dict[int, Tuple[float, float, int]] = {}
        for s in self._samples:
            bucket = min(9, int(s.predicted * 10))
            if bucket not in bins:
                bins[bucket] = (0.0, 0.0, 0)
            sum_pred, sum_actual, count = bins[bucket]
            bins[bucket] = (sum_pred + s.predicted, sum_actual + s.actual, count + 1)

        reliability = []
        for bucket in sorted(bins.keys()):
            sum_pred, sum_actual, count = bins[bucket]
            if count > 0:
                reliability.append({
                    "bucket": bucket / 10,
                    "expected": sum_pred / count,
                    "observed": sum_actual / count,
                    "count": count,
                })

        return {
            "sample_count": n,
            "brier_score": round(brier, 4),
            "calibration_slope": round(slope, 3),
            "reliability_diagram": reliability,
        }

    def _load(self) -> None:
        """Load calibration data from disk."""
        if not self._data_path or not self._data_path.exists():
            return

        try:
            with open(self._data_path) as f:
                data = json.load(f)

            # Load samples
            self._samples = [
                CalibrationSample(
                    predicted=s["predicted"],
                    actual=s["actual"],
                    timestamp=datetime.fromisoformat(s["timestamp"]),
                    market_id=s.get("market_id"),
                    sport=s.get("sport"),
                )
                for s in data.get("samples", [])
            ]

            # Load fitted model
            if data.get("fitted"):
                self._calibration_x = data.get("calibration_x", [])
                self._calibration_y = data.get("calibration_y", [])
                self._fitted = True

        except (json.JSONDecodeError, KeyError):
            pass  # Start fresh if corrupted

    def _save(self) -> None:
        """Save calibration data to disk."""
        if not self._data_path:
            return

        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "samples": [
                {
                    "predicted": s.predicted,
                    "actual": s.actual,
                    "timestamp": s.timestamp.isoformat(),
                    "market_id": s.market_id,
                    "sport": s.sport,
                }
                for s in self._samples[-10000:]  # Keep last 10k samples
            ],
            "fitted": self._fitted,
            "calibration_x": self._calibration_x,
            "calibration_y": self._calibration_y,
        }

        with open(self._data_path, "w") as f:
            json.dump(data, f, indent=2)

    def clear(self) -> None:
        """Clear all calibration data."""
        self._samples.clear()
        self._calibration_x.clear()
        self._calibration_y.clear()
        self._fitted = False
        self._save()
