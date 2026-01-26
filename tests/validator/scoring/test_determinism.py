"""Tests for determinism utilities."""

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from sparket.validator.scoring.determinism import (
    to_decimal,
    round_decimal,
    safe_divide,
    safe_sqrt,
    safe_ln,
    clamp,
    sort_by_id,
    deterministic_weighted_sum,
    deterministic_weighted_mean,
    deterministic_variance,
    get_canonical_window_bounds,
    floor_to_bucket,
    get_epoch_day,
    compute_hash,
    compute_scoring_hash,
    increment_version,
    get_current_version_timestamp,
)
from sparket.validator.scoring.types import ValidationError


class TestToDecimal:
    """Tests for to_decimal conversion."""

    def test_from_int(self):
        """Should convert int to Decimal."""
        result = to_decimal(42)
        assert result == Decimal("42")

    def test_from_float(self):
        """Should convert float to Decimal."""
        result = to_decimal(3.14)
        assert result == Decimal("3.14")

    def test_from_str(self):
        """Should convert string to Decimal."""
        result = to_decimal("123.456")
        assert result == Decimal("123.456")

    def test_from_decimal(self):
        """Should pass through Decimal unchanged."""
        d = Decimal("99.99")
        result = to_decimal(d)
        assert result == d

    def test_none_raises(self):
        """None should raise ValidationError."""
        with pytest.raises(ValidationError, match="is None"):
            to_decimal(None)

    def test_nan_raises(self):
        """NaN should raise ValidationError."""
        with pytest.raises(ValidationError, match="is NaN"):
            to_decimal(Decimal("NaN"))

    def test_inf_raises(self):
        """Infinity should raise ValidationError."""
        with pytest.raises(ValidationError, match="is infinite"):
            to_decimal(Decimal("Infinity"))

    def test_invalid_string_raises(self):
        """Invalid string should raise ValidationError."""
        with pytest.raises(ValidationError, match="Cannot convert"):
            to_decimal("not_a_number")

    def test_custom_name_in_error(self):
        """Custom name should appear in error message."""
        with pytest.raises(ValidationError, match="my_value"):
            to_decimal(None, "my_value")


class TestRoundDecimal:
    """Tests for round_decimal."""

    def test_round_default_places(self):
        """Should round to 8 decimal places by default."""
        d = Decimal("1.123456789123")
        result = round_decimal(d)
        assert result == Decimal("1.12345679")

    def test_round_custom_places(self):
        """Should respect custom places."""
        d = Decimal("1.123456789")
        result = round_decimal(d, 2)
        assert result == Decimal("1.12")

    def test_bankers_rounding(self):
        """Should use banker's rounding (ROUND_HALF_EVEN)."""
        # 0.5 -> 0 (round to even)
        assert round_decimal(Decimal("0.5"), 0) == Decimal("0")
        # 1.5 -> 2 (round to even)
        assert round_decimal(Decimal("1.5"), 0) == Decimal("2")
        # 2.5 -> 2 (round to even)
        assert round_decimal(Decimal("2.5"), 0) == Decimal("2")


class TestSafeDivide:
    """Tests for safe_divide."""

    def test_normal_division(self):
        """Should perform normal division."""
        result = safe_divide(Decimal("10"), Decimal("2"))
        assert result == Decimal("5")

    def test_zero_denominator(self):
        """Zero denominator should return default."""
        result = safe_divide(Decimal("10"), Decimal("0"))
        assert result == Decimal("0")

    def test_custom_default(self):
        """Should use custom default."""
        result = safe_divide(Decimal("10"), Decimal("0"), Decimal("-1"))
        assert result == Decimal("-1")

    def test_nan_denominator(self):
        """NaN denominator should return default."""
        result = safe_divide(Decimal("10"), Decimal("NaN"))
        assert result == Decimal("0")

    def test_nan_numerator(self):
        """NaN numerator should return default."""
        result = safe_divide(Decimal("NaN"), Decimal("2"))
        assert result == Decimal("0")

    def test_infinity_denominator(self):
        """Infinity denominator should return 0 (not default, actual result)."""
        # 10 / Inf = 0, which is valid
        result = safe_divide(Decimal("10"), Decimal("Infinity"))
        assert result == Decimal("0")

    def test_inf_numerator_gives_default(self):
        """Inf / x = Inf, which should return default since result is infinite."""
        result = safe_divide(Decimal("Infinity"), Decimal("2"))
        # Result is Infinity, so should return default (0)
        assert result == Decimal("0")


class TestSafeSqrt:
    """Tests for safe_sqrt."""

    def test_positive_value(self):
        """Should compute square root."""
        result = safe_sqrt(Decimal("4"))
        assert result == Decimal("2")

    def test_zero(self):
        """Zero should return zero."""
        result = safe_sqrt(Decimal("0"))
        assert result == Decimal("0")

    def test_negative_returns_zero(self):
        """Negative should return zero."""
        result = safe_sqrt(Decimal("-4"))
        assert result == Decimal("0")


class TestSafeLn:
    """Tests for safe_ln."""

    def test_valid_value(self):
        """Should compute natural log."""
        # Note: safe_ln clamps to [eps, 1-eps], so test with value in range
        result = safe_ln(Decimal("0.5"))
        # ln(0.5) ≈ -0.693
        assert abs(result - Decimal("-0.693147")) < Decimal("0.001")

    def test_clamped_low(self):
        """Low values should be clamped."""
        result = safe_ln(Decimal("0"))
        # Should return ln(eps), not error
        assert result < Decimal("-10")

    def test_clamped_high(self):
        """Values near 1 should be clamped."""
        result = safe_ln(Decimal("1"))
        # Should return ln(1-eps), close to 0
        assert abs(result) < Decimal("0.0001")


class TestClamp:
    """Tests for clamp."""

    def test_within_range(self):
        """Value within range should pass through."""
        result = clamp(Decimal("5"), Decimal("0"), Decimal("10"))
        assert result == Decimal("5")

    def test_below_min(self):
        """Below min should return min."""
        result = clamp(Decimal("-5"), Decimal("0"), Decimal("10"))
        assert result == Decimal("0")

    def test_above_max(self):
        """Above max should return max."""
        result = clamp(Decimal("15"), Decimal("0"), Decimal("10"))
        assert result == Decimal("10")


class TestSortById:
    """Tests for sort_by_id."""

    def test_sort_dicts(self):
        """Should sort list of dicts by id."""
        items = [{"id": 3}, {"id": 1}, {"id": 2}]
        result = sort_by_id(items)
        assert [x["id"] for x in result] == [1, 2, 3]

    def test_sort_objects(self):
        """Should sort objects by attribute."""
        class Item:
            def __init__(self, id):
                self.id = id

        items = [Item(3), Item(1), Item(2)]
        result = sort_by_id(items)
        assert [x.id for x in result] == [1, 2, 3]

    def test_custom_key(self):
        """Should use custom key."""
        items = [{"uid": 3}, {"uid": 1}, {"uid": 2}]
        result = sort_by_id(items, id_key="uid")
        assert [x["uid"] for x in result] == [1, 2, 3]

    def test_empty_list(self):
        """Empty list should return empty."""
        result = sort_by_id([])
        assert result == []


class TestDeterministicWeightedSum:
    """Tests for deterministic_weighted_sum."""

    def test_simple_sum(self):
        """Should compute weighted sum."""
        items = [
            (1, Decimal("10"), Decimal("1")),
            (2, Decimal("20"), Decimal("2")),
        ]
        weighted_sum, weight_sum = deterministic_weighted_sum(items)
        # 10*1 + 20*2 = 50
        assert weighted_sum == Decimal("50")
        assert weight_sum == Decimal("3")

    def test_empty_items(self):
        """Empty items should return zeros."""
        weighted_sum, weight_sum = deterministic_weighted_sum([])
        assert weighted_sum == Decimal("0")
        assert weight_sum == Decimal("0")

    def test_order_independent(self):
        """Result should be same regardless of input order."""
        items1 = [(1, Decimal("10"), Decimal("1")), (2, Decimal("20"), Decimal("2"))]
        items2 = [(2, Decimal("20"), Decimal("2")), (1, Decimal("10"), Decimal("1"))]

        result1 = deterministic_weighted_sum(items1)
        result2 = deterministic_weighted_sum(items2)

        assert result1 == result2


class TestDeterministicWeightedMean:
    """Tests for deterministic_weighted_mean."""

    def test_simple_mean(self):
        """Should compute weighted mean."""
        items = [
            (1, Decimal("10"), Decimal("1")),
            (2, Decimal("20"), Decimal("3")),
        ]
        result = deterministic_weighted_mean(items)
        # (10*1 + 20*3) / (1+3) = 70/4 = 17.5
        assert result == Decimal("17.5")

    def test_empty_items(self):
        """Empty items should return 0."""
        result = deterministic_weighted_mean([])
        assert result == Decimal("0")


class TestDeterministicVariance:
    """Tests for deterministic_variance."""

    def test_simple_variance(self):
        """Should compute weighted variance."""
        items = [
            (1, Decimal("0"), Decimal("1")),
            (2, Decimal("2"), Decimal("1")),
        ]
        mean = Decimal("1")  # (0+2)/2
        result = deterministic_variance(items, mean)
        # ((0-1)^2 * 1 + (2-1)^2 * 1) / 2 = 2/2 = 1
        assert result == Decimal("1")


class TestGetCanonicalWindowBounds:
    """Tests for get_canonical_window_bounds."""

    def test_returns_tuple(self):
        """Should return (start, end) tuple."""
        start, end = get_canonical_window_bounds(7)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)

    def test_window_size(self):
        """Window should be correct size."""
        start, end = get_canonical_window_bounds(7)
        assert (end - start).days == 7

    def test_midnight_aligned(self):
        """Bounds should be at midnight."""
        start, end = get_canonical_window_bounds(7)
        assert start.hour == 0 and start.minute == 0 and start.second == 0
        assert end.hour == 0 and end.minute == 0 and end.second == 0

    def test_utc_timezone(self):
        """Should be UTC timezone."""
        start, end = get_canonical_window_bounds(7)
        assert start.tzinfo == timezone.utc
        assert end.tzinfo == timezone.utc

    def test_custom_reference(self):
        """Should use custom reference time."""
        ref = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        start, end = get_canonical_window_bounds(7, ref)
        assert end == datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_naive_reference_becomes_utc(self):
        """Naive reference should become UTC."""
        ref = datetime(2024, 1, 15, 12, 30, 0)  # Naive
        start, end = get_canonical_window_bounds(7, ref)
        assert end.tzinfo == timezone.utc


class TestFloorToBucket:
    """Tests for floor_to_bucket."""

    def test_exact_boundary(self):
        """Exact boundary should stay same."""
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = floor_to_bucket(dt, 3600)  # 1 hour buckets
        assert result == dt

    def test_floor_to_hour(self):
        """Should floor to hour boundary."""
        dt = datetime(2024, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
        result = floor_to_bucket(dt, 3600)
        assert result == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_floor_to_minute(self):
        """Should floor to minute boundary."""
        dt = datetime(2024, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
        result = floor_to_bucket(dt, 60)
        assert result == datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc)

    def test_naive_becomes_utc(self):
        """Naive datetime should become UTC."""
        dt = datetime(2024, 1, 1, 12, 30, 45)
        result = floor_to_bucket(dt, 3600)
        assert result.tzinfo == timezone.utc


class TestGetEpochDay:
    """Tests for get_epoch_day."""

    def test_epoch_start(self):
        """Epoch start should be day 0."""
        dt = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert get_epoch_day(dt) == 0

    def test_day_one(self):
        """Day after epoch should be 1."""
        dt = datetime(1970, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert get_epoch_day(dt) == 1

    def test_same_day_any_time(self):
        """Same day regardless of time."""
        dt1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 1, 23, 59, 59, tzinfo=timezone.utc)
        assert get_epoch_day(dt1) == get_epoch_day(dt2)

    def test_naive_becomes_utc(self):
        """Naive datetime should become UTC."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = get_epoch_day(dt)
        assert isinstance(result, int)


class TestComputeHash:
    """Tests for compute_hash."""

    def test_deterministic(self):
        """Same input should give same hash."""
        data = {"a": 1, "b": 2}
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)
        assert hash1 == hash2

    def test_order_independent(self):
        """Key order should not affect hash."""
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}
        assert compute_hash(data1) == compute_hash(data2)

    def test_different_data_different_hash(self):
        """Different data should give different hash."""
        hash1 = compute_hash({"a": 1})
        hash2 = compute_hash({"a": 2})
        assert hash1 != hash2

    def test_handles_decimal(self):
        """Should handle Decimal values."""
        data = {"value": Decimal("123.456")}
        result = compute_hash(data)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex

    def test_handles_datetime(self):
        """Should handle datetime values."""
        data = {"time": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)}
        result = compute_hash(data)
        assert isinstance(result, str)

    def test_handles_nested(self):
        """Should handle nested structures."""
        data = {"nested": {"list": [1, 2, 3]}}
        result = compute_hash(data)
        assert isinstance(result, str)


class TestComputeScoringHash:
    """Tests for compute_scoring_hash."""

    def test_deterministic(self):
        """Same input should give same hash."""
        hash1 = compute_scoring_hash(
            miner_id=123,
            window_end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            scores={"a": 1, "b": 2},
        )
        hash2 = compute_scoring_hash(
            miner_id=123,
            window_end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            scores={"a": 1, "b": 2},
        )
        assert hash1 == hash2

    def test_different_miner_different_hash(self):
        """Different miner should give different hash."""
        hash1 = compute_scoring_hash(123, datetime(2024, 1, 1, tzinfo=timezone.utc), {})
        hash2 = compute_scoring_hash(456, datetime(2024, 1, 1, tzinfo=timezone.utc), {})
        assert hash1 != hash2


class TestIncrementVersion:
    """Tests for increment_version."""

    def test_increments(self):
        """Should increment by 1."""
        assert increment_version(0) == 1
        assert increment_version(10) == 11


class TestGetCurrentVersionTimestamp:
    """Tests for get_current_version_timestamp."""

    def test_returns_string(self):
        """Should return string."""
        result = get_current_version_timestamp()
        assert isinstance(result, str)

    def test_format(self):
        """Should be in YYYYMMDDHHMMSS format."""
        result = get_current_version_timestamp()
        assert len(result) == 14
        assert result.isdigit()

