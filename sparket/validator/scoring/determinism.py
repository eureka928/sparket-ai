"""Determinism utilities for cross-validator consensus.

This module ensures that scoring computations are fully deterministic:
1. Decimal arithmetic instead of floating point
2. Canonical ordering for aggregations
3. Epoch-aligned time windows
4. Deterministic hashing for verification

All validators running the same code with the same data MUST produce
identical scoring outputs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Any, Iterable, List, Tuple, TypeVar

from .types import DECIMAL_PLACES, ValidationError

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# Decimal Arithmetic
# ─────────────────────────────────────────────────────────────────────────────


def to_decimal(value: Any, name: str = "value") -> Decimal:
    """Convert a value to Decimal with validation.

    Args:
        value: Value to convert (int, float, str, Decimal)
        name: Name for error messages

    Returns:
        Decimal representation

    Raises:
        ValidationError: If value cannot be converted or is invalid
    """
    if value is None:
        raise ValidationError(f"{name} is None")

    try:
        if isinstance(value, Decimal):
            d = value
        else:
            d = Decimal(str(value))

        if d.is_nan():
            raise ValidationError(f"{name} is NaN")
        if d.is_infinite():
            raise ValidationError(f"{name} is infinite")

        return d

    except (InvalidOperation, ValueError, TypeError) as e:
        raise ValidationError(f"Cannot convert {name}={value!r} to Decimal: {e}")


def round_decimal(value: Decimal, places: int = DECIMAL_PLACES) -> Decimal:
    """Round a Decimal to specified places using ROUND_HALF_EVEN.

    ROUND_HALF_EVEN (banker's rounding) is deterministic and unbiased.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_EVEN)


def safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning default on zero/invalid.

    Args:
        numerator: Dividend
        denominator: Divisor
        default: Value to return if division is invalid

    Returns:
        numerator / denominator, or default if invalid
    """
    if denominator == Decimal("0"):
        return default
    if denominator.is_nan() or numerator.is_nan():
        return default

    result = numerator / denominator
    if result.is_nan() or result.is_infinite():
        return default

    return result


def safe_sqrt(value: Decimal) -> Decimal:
    """Compute square root of a Decimal safely.

    Returns 0 for negative values.
    """
    if value < Decimal("0"):
        return Decimal("0")
    return value.sqrt()


def safe_ln(value: Decimal, eps: Decimal = Decimal("1e-9")) -> Decimal:
    """Compute natural log of a Decimal safely.

    Clamps value to [eps, 1-eps] to prevent domain errors.
    """
    clamped = max(eps, min(value, Decimal("1") - eps))
    return clamped.ln()


def clamp(value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal:
    """Clamp a Decimal to a range."""
    return max(min_val, min(value, max_val))


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Ordering
# ─────────────────────────────────────────────────────────────────────────────


def sort_by_id(items: Iterable[T], id_key: str = "id") -> List[T]:
    """Sort items by their ID for deterministic processing order.

    Args:
        items: Iterable of dicts or objects
        id_key: Key/attribute name for ID

    Returns:
        Sorted list
    """
    items_list = list(items)
    if not items_list:
        return items_list

    if isinstance(items_list[0], dict):
        return sorted(items_list, key=lambda x: x.get(id_key, 0))
    else:
        return sorted(items_list, key=lambda x: getattr(x, id_key, 0))


def deterministic_weighted_sum(
    items: Iterable[Tuple[int, Decimal, Decimal]],
) -> Tuple[Decimal, Decimal]:
    """Compute weighted sum with deterministic ordering.

    Args:
        items: Iterable of (id, value, weight) tuples

    Returns:
        (weighted_sum, weight_sum)

    Note:
        Items are sorted by ID before summation to ensure
        consistent floating-point accumulation order.
    """
    sorted_items = sorted(items, key=lambda x: x[0])

    weighted_sum = Decimal("0")
    weight_sum = Decimal("0")

    for _, value, weight in sorted_items:
        weighted_sum += value * weight
        weight_sum += weight

    return weighted_sum, weight_sum


def deterministic_weighted_mean(
    items: Iterable[Tuple[int, Decimal, Decimal]],
) -> Decimal:
    """Compute weighted mean with deterministic ordering.

    Args:
        items: Iterable of (id, value, weight) tuples

    Returns:
        Weighted mean, or 0 if no items
    """
    weighted_sum, weight_sum = deterministic_weighted_sum(items)
    return safe_divide(weighted_sum, weight_sum)


def deterministic_variance(
    items: Iterable[Tuple[int, Decimal, Decimal]],
    mean: Decimal,
) -> Decimal:
    """Compute weighted variance with deterministic ordering.

    Args:
        items: Iterable of (id, value, weight) tuples
        mean: Pre-computed weighted mean

    Returns:
        Weighted variance
    """
    sorted_items = sorted(items, key=lambda x: x[0])

    variance_sum = Decimal("0")
    weight_sum = Decimal("0")

    for _, value, weight in sorted_items:
        diff = value - mean
        variance_sum += weight * diff * diff
        weight_sum += weight

    return safe_divide(variance_sum, weight_sum)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Time Windows
# ─────────────────────────────────────────────────────────────────────────────


def get_canonical_window_bounds(
    window_days: int,
    reference_time: datetime | None = None,
) -> Tuple[datetime, datetime]:
    """Get canonical window boundaries aligned to UTC midnight.

    This ensures all validators use the same window boundaries
    regardless of when they run.

    Args:
        window_days: Number of days in the window
        reference_time: End of window (defaults to now, floored to midnight)

    Returns:
        (window_start, window_end) both at UTC midnight
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    # Ensure timezone-aware
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    # Floor to midnight UTC
    end = reference_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=window_days)

    return start, end


def floor_to_bucket(dt: datetime, bucket_seconds: int) -> datetime:
    """Floor a datetime to the nearest bucket boundary.

    Args:
        dt: Datetime to floor
        bucket_seconds: Bucket size in seconds

    Returns:
        Floored datetime (timezone-aware UTC)
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    epoch = int(dt.timestamp())
    bucket = epoch - (epoch % bucket_seconds)
    return datetime.fromtimestamp(bucket, tz=timezone.utc)


def get_epoch_day(dt: datetime) -> int:
    """Get the epoch day number for a datetime.

    Useful for deterministic day-based grouping.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() // 86400)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Hashing
# ─────────────────────────────────────────────────────────────────────────────


def _serialize_for_hash(obj: Any) -> Any:
    """Recursively serialize an object for hashing."""
    if isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_for_hash(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_hash(v) for v in obj]
    else:
        return obj


def compute_hash(data: dict) -> str:
    """Compute a deterministic SHA256 hash of a dictionary.

    Args:
        data: Dictionary to hash

    Returns:
        Hex-encoded SHA256 hash
    """
    serialized = _serialize_for_hash(data)
    canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_scoring_hash(
    miner_id: int,
    window_end: datetime,
    scores: dict,
) -> str:
    """Compute deterministic hash of scoring inputs/outputs.

    Used for cross-validator consensus verification.

    Args:
        miner_id: Miner identifier
        window_end: End of scoring window
        scores: Score values

    Returns:
        Hex-encoded SHA256 hash
    """
    payload = {
        "miner_id": miner_id,
        "window_end": window_end.isoformat(),
        "scores": {k: str(v) for k, v in sorted(scores.items())},
    }
    return compute_hash(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Version Management
# ─────────────────────────────────────────────────────────────────────────────


def increment_version(current: int) -> int:
    """Increment a version number."""
    return current + 1


def get_current_version_timestamp() -> str:
    """Get a timestamp string for version tagging."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


__all__ = [
    # Decimal arithmetic
    "to_decimal",
    "round_decimal",
    "safe_divide",
    "safe_sqrt",
    "safe_ln",
    "clamp",
    # Canonical ordering
    "sort_by_id",
    "deterministic_weighted_sum",
    "deterministic_weighted_mean",
    "deterministic_variance",
    # Time windows
    "get_canonical_window_bounds",
    "floor_to_bucket",
    "get_epoch_day",
    # Hashing
    "compute_hash",
    "compute_scoring_hash",
    # Versioning
    "increment_version",
    "get_current_version_timestamp",
]

