"""Adversarial validation tests - focused on input sanitization.

Tests for:
1. Malformed input structures
2. Type confusion attacks
3. Unicode/encoding tricks
4. Integer overflow scenarios
5. Resource exhaustion attempts
6. Injection-style attacks on hashing
"""

import math
import sys
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from sparket.validator.scoring.validation import (
    SubmissionValidator,
)
from sparket.validator.config.scoring_params import ScoringParams, SecurityBounds
from sparket.validator.scoring.determinism import (
    to_decimal,
    compute_hash,
    safe_divide,
    safe_sqrt,
    safe_ln,
    clamp,
)
from sparket.validator.scoring.types import ValidationError


class TestMalformedInputs:
    """Tests for malformed or unexpected input structures."""

    def test_none_odds(self):
        """None value for odds."""
        validator = SubmissionValidator()

        with pytest.raises(ValidationError):
            validator.validate_odds(None)

    def test_none_probability(self):
        """None value for probability."""
        validator = SubmissionValidator()

        with pytest.raises(ValidationError):
            validator.validate_probability(None)

    def test_empty_probability_vector(self):
        """Empty probability vector."""
        validator = SubmissionValidator()

        with pytest.raises(ValidationError):
            validator.validate_probability_vector([])

    def test_single_outcome_vector(self):
        """Probability vector with only one outcome."""
        validator = SubmissionValidator()

        # Events require at least 2 outcomes
        with pytest.raises(ValidationError):
            validator.validate_probability_vector([Decimal("1.0")])

    def test_nested_list_probabilities(self):
        """Nested lists instead of flat array."""
        validator = SubmissionValidator()

        # [[0.5, 0.5]] instead of [0.5, 0.5]
        with pytest.raises((ValidationError, TypeError)):
            validator.validate_probability_vector([[0.5, 0.5]])

    def test_dict_instead_of_list(self):
        """Dictionary instead of list for probabilities."""
        validator = SubmissionValidator()

        with pytest.raises((ValidationError, TypeError)):
            validator.validate_probability_vector({"a": 0.5, "b": 0.5})


class TestTypeConfusion:
    """Tests for type confusion attacks."""

    def test_string_odds(self):
        """String that looks like a number for odds."""
        validator = SubmissionValidator()

        # Should convert or reject
        result = validator.validate_odds("2.5")
        assert isinstance(result, Decimal)

    def test_complex_number_odds(self):
        """Complex number for odds."""
        validator = SubmissionValidator()

        with pytest.raises((ValidationError, TypeError)):
            validator.validate_odds(complex(2.5, 0))

    def test_boolean_probability(self):
        """Boolean as probability."""
        validator = SubmissionValidator()

        # True = 1, False = 0 - should work or fail gracefully
        with pytest.raises((ValidationError, TypeError)):
            validator.validate_probability(True)

    def test_list_as_single_probability(self):
        """List passed as single probability."""
        validator = SubmissionValidator()

        with pytest.raises((ValidationError, TypeError)):
            validator.validate_probability([0.5])

    def test_numpy_array_single_value(self):
        """NumPy array with single value."""
        validator = SubmissionValidator()

        arr = np.array([0.5])
        result = validator.validate_probability(arr[0])
        # Should work - numpy scalar is numeric
        assert isinstance(result, Decimal)


class TestUnicodeEncoding:
    """Tests for Unicode and encoding edge cases."""

    def test_unicode_number_chars(self):
        """Unicode digit characters."""
        validator = SubmissionValidator()

        # Full-width digits actually work in Python's Decimal - interesting!
        # This is a potential unicode normalization attack vector to be aware of
        # For now, just verify we handle it without crashing
        try:
            result = validator.validate_odds("２.５")  # Full-width 2 and 5
            assert isinstance(result, Decimal)
        except (ValidationError, ValueError):
            pass  # Rejection is also acceptable

    def test_negative_zero(self):
        """Negative zero -0.0."""
        validator = SubmissionValidator()

        # -0.0 converts to Decimal('-0.0') which is < prob_min (0.001)
        # Validator correctly rejects this
        with pytest.raises(ValidationError):
            validator.validate_probability(-0.0)

    def test_special_float_strings(self):
        """Special float string representations."""
        validator = SubmissionValidator()

        # "nan" string
        with pytest.raises(ValidationError):
            validator.validate_odds("nan")

        # "inf" string
        with pytest.raises(ValidationError):
            validator.validate_odds("inf")


class TestIntegerOverflow:
    """Tests for integer overflow scenarios."""

    def test_huge_odds(self):
        """Odds larger than max float."""
        validator = SubmissionValidator()

        # Beyond float64 max
        with pytest.raises((ValidationError, OverflowError)):
            validator.validate_odds(10 ** 400)

    def test_tiny_probability(self):
        """Probability smaller than min float."""
        validator = SubmissionValidator()

        # Very tiny values get rejected (< prob_min)
        with pytest.raises(ValidationError):
            validator.validate_probability(10 ** -400)

    def test_large_exponent_decimal(self):
        """Decimal with huge exponent stays as Decimal."""
        # Decimal can handle huge exponents internally
        d = Decimal("1E+1000000")
        assert isinstance(d, Decimal)

        # Converting to float gives inf (no OverflowError in Python 3)
        f = float(d)
        assert f == float("inf")


class TestResourceExhaustion:
    """Tests for resource exhaustion attempts."""

    def test_very_long_probability_vector(self):
        """Extremely long probability vector."""
        validator = SubmissionValidator()

        # 10,000 outcomes (should be rejected or limited)
        vec = [Decimal("0.0001")] * 10000

        # Either reject or handle
        try:
            result = validator.validate_probability_vector(vec)
            # If it accepts, should still sum to ~1
            assert abs(sum(result) - 1) < validator.bounds.prob_sum_tolerance
        except ValidationError:
            pass  # Rejection is fine

    def test_deeply_nested_hash_input(self):
        """Deeply nested structure for hashing."""
        # Create nested dict
        data = {"level": 0}
        current = data
        for i in range(100):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        # Should hash without stack overflow
        result = compute_hash(data)
        assert isinstance(result, str)


class TestInjectionStyleAttacks:
    """Tests for injection-style attacks."""

    def test_hash_with_special_chars(self):
        """Data with special characters that might break serialization."""
        data = {
            "miner\n\r\t": 1,
            "score": 0.5,
            "weird": "value\x00with\x00nulls",
        }

        # Should not crash
        result = compute_hash(data)
        assert isinstance(result, str)

    def test_hash_with_unicode_keys(self):
        """Unicode keys in hash data."""
        data = {
            "émoji": "🎲",
            "مرحبا": "hello",
            "score": 0.5,
        }

        result = compute_hash(data)
        assert isinstance(result, str)

    def test_decimal_string_injection(self):
        """Attempt to inject via Decimal string conversion."""
        # Malicious string that might exploit parsing
        malicious = "0." + "9" * 1000

        d = to_decimal(malicious, "test")
        assert isinstance(d, Decimal)
        # Should be less than 1
        assert d < Decimal("1")


class TestSafeMathEdgeCases:
    """Edge cases for safe math operations."""

    def test_divide_very_small_by_very_large(self):
        """Divide tiny number by huge number."""
        tiny = Decimal("1E-308")
        huge = Decimal("1E+308")

        result = safe_divide(tiny, huge, Decimal("0"))
        # Should be 0 or very small, not error
        assert result == Decimal("0") or result < Decimal("1E-100")

    def test_sqrt_of_subnormal(self):
        """Square root of subnormal number."""
        subnormal = Decimal("1E-400")

        result = safe_sqrt(subnormal)
        # Should get valid Decimal result
        assert isinstance(result, Decimal)
        assert result >= Decimal("0")

    def test_ln_of_subnormal(self):
        """Natural log of subnormal number."""
        subnormal = Decimal("1E-400")

        result = safe_ln(subnormal, Decimal("-999"))
        # Should return default or very negative number
        assert result <= Decimal("-100")

    def test_clamp_with_inverted_bounds(self):
        """Clamp with min > max."""
        # This shouldn't happen, but let's test
        value = Decimal("0.5")
        min_val = Decimal("0.8")
        max_val = Decimal("0.2")

        # Behavior is undefined, but shouldn't crash
        result = clamp(value, min_val, max_val)
        assert isinstance(result, Decimal)


class TestValidationBoundsEdgeCases:
    """Edge cases for validation bounds configuration."""

    def test_zero_tolerance(self):
        """Zero probability sum tolerance."""
        params = ScoringParams()
        params.security.prob_sum_tolerance = Decimal("0")
        validator = SubmissionValidator(params)

        # Exact sum required - this returns (probs, overround)
        exact = [Decimal("0.5"), Decimal("0.5")]
        probs, overround = validator.validate_probability_vector(exact)
        assert abs(sum(probs) - 1) == 0

    def test_identical_min_max(self):
        """Min equals max for bounds."""
        params = ScoringParams()
        params.security.odds_min = Decimal("2.0")
        params.security.odds_max = Decimal("2.0")
        validator = SubmissionValidator(params)

        # Only one valid odds value
        result = validator.validate_odds(2.0)
        assert result == Decimal("2.0")

        with pytest.raises(ValidationError):
            validator.validate_odds(2.1)


class TestNumpyIntegrationEdgeCases:
    """Edge cases when converting between numpy and validation."""

    def test_numpy_float16(self):
        """Half-precision float."""
        validator = SubmissionValidator()

        val = np.float16(0.5)
        result = validator.validate_probability(val)
        assert isinstance(result, Decimal)

    def test_numpy_float32(self):
        """Single-precision float."""
        validator = SubmissionValidator()

        val = np.float32(0.5)
        result = validator.validate_probability(val)
        assert isinstance(result, Decimal)

    def test_numpy_int_as_probability(self):
        """NumPy integer as probability."""
        validator = SubmissionValidator()

        # Probability 1 is rejected (> max 0.999), which is correct
        # This protects against "certain" predictions
        with pytest.raises(ValidationError):
            validator.validate_probability(np.int64(1))

        # Valid value should work
        result = validator.validate_probability(np.int64(0) + 0.5)
        assert result == Decimal("0.5")

    def test_numpy_inf_values(self):
        """NumPy infinity values."""
        validator = SubmissionValidator()

        with pytest.raises(ValidationError):
            validator.validate_odds(np.inf)

        with pytest.raises(ValidationError):
            validator.validate_odds(-np.inf)

    def test_numpy_nan_values(self):
        """NumPy NaN values."""
        validator = SubmissionValidator()

        with pytest.raises(ValidationError):
            validator.validate_probability(np.nan)

