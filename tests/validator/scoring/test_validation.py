"""Tests for submission validation."""

from decimal import Decimal

import pytest

from sparket.validator.scoring.validation import (
    SubmissionValidator,
    validate_submission_safe,
    validate_probability_safe,
    get_validator,
)
from sparket.validator.scoring.types import ValidationError


@pytest.fixture
def validator():
    """Create a validator instance."""
    return SubmissionValidator()


class TestValidateOdds:
    """Tests for odds validation."""

    def test_valid_odds(self, validator):
        """Valid odds should pass through."""
        result = validator.validate_odds(2.0)
        assert result == Decimal("2.0")

    def test_valid_odds_decimal(self, validator):
        """Decimal odds should pass through."""
        result = validator.validate_odds(Decimal("1.5"))
        assert result == Decimal("1.5")

    def test_valid_odds_string(self, validator):
        """String odds should be converted."""
        result = validator.validate_odds("2.5")
        assert result == Decimal("2.5")

    def test_odds_below_min(self, validator):
        """Odds below minimum should raise."""
        with pytest.raises(ValidationError, match="< min"):
            validator.validate_odds(0.5)

    def test_odds_above_max(self, validator):
        """Odds above maximum should raise."""
        with pytest.raises(ValidationError, match="> max"):
            validator.validate_odds(100000)

    def test_nan_odds(self, validator):
        """NaN odds should raise."""
        with pytest.raises(ValidationError, match="NaN"):
            validator.validate_odds(float("nan"))

    def test_inf_odds(self, validator):
        """Infinity odds should raise."""
        with pytest.raises(ValidationError, match="infinite"):
            validator.validate_odds(float("inf"))

    def test_none_odds(self, validator):
        """None odds should raise."""
        with pytest.raises(ValidationError, match="is None"):
            validator.validate_odds(None)


class TestValidateProbability:
    """Tests for probability validation."""

    def test_valid_probability(self, validator):
        """Valid probability should pass through."""
        result = validator.validate_probability(0.5)
        assert result == Decimal("0.5")

    def test_probability_below_min(self, validator):
        """Probability below minimum should raise."""
        with pytest.raises(ValidationError, match="< min"):
            validator.validate_probability(0.0)

    def test_probability_above_max(self, validator):
        """Probability above maximum should raise."""
        with pytest.raises(ValidationError, match="> max"):
            validator.validate_probability(1.5)

    def test_nan_probability(self, validator):
        """NaN probability should raise."""
        with pytest.raises(ValidationError, match="NaN"):
            validator.validate_probability(float("nan"))


class TestValidateProbabilityVector:
    """Tests for probability vector validation."""

    def test_valid_vector(self, validator):
        """Valid probability vector should pass."""
        probs, overround = validator.validate_probability_vector([0.5, 0.5])
        assert len(probs) == 2
        assert all(isinstance(p, Decimal) for p in probs)
        assert overround == Decimal("0")

    def test_valid_vector_three_way(self, validator):
        """Three-way market should work."""
        probs, overround = validator.validate_probability_vector([0.4, 0.3, 0.3])
        assert len(probs) == 3

    def test_empty_vector(self, validator):
        """Empty vector should raise."""
        with pytest.raises(ValidationError, match="empty"):
            validator.validate_probability_vector([])

    def test_single_outcome(self, validator):
        """Single outcome should raise."""
        with pytest.raises(ValidationError, match=">= 2"):
            validator.validate_probability_vector([1.0])

    def test_sum_deviation(self, validator):
        """Vector with sum too far from 1 should raise."""
        with pytest.raises(ValidationError, match="deviates from 1.0"):
            validator.validate_probability_vector([0.3, 0.3])  # Sum = 0.6

    def test_individual_invalid(self, validator):
        """Invalid individual probability should raise with index."""
        with pytest.raises(ValidationError, match=r"probability\[1\]"):
            validator.validate_probability_vector([0.5, float("nan")])

    def test_overround_computed(self, validator):
        """Overround should be computed correctly."""
        # With small overround within tolerance
        probs, overround = validator.validate_probability_vector([0.505, 0.5])
        assert overround == Decimal("0.005")


class TestValidateCLE:
    """Tests for CLE clamping."""

    def test_normal_cle(self, validator):
        """Normal CLE should pass through."""
        result = validator.validate_cle(Decimal("0.1"))
        assert result == Decimal("0.1")

    def test_cle_clamped_high(self, validator):
        """High CLE should be clamped."""
        result = validator.validate_cle(Decimal("15.0"))
        assert result == validator.bounds.cle_max  # cle_max is 10

    def test_cle_clamped_low(self, validator):
        """Low CLE should be clamped."""
        result = validator.validate_cle(Decimal("-2.0"))
        assert result == validator.bounds.cle_min


class TestOddsProbConversion:
    """Tests for odds/probability conversion."""

    def test_odds_to_prob(self, validator):
        """Odds to prob conversion."""
        result = validator.odds_to_prob(Decimal("2.0"))
        assert result == Decimal("0.5")

    def test_prob_to_odds(self, validator):
        """Prob to odds conversion."""
        result = validator.prob_to_odds(Decimal("0.5"))
        assert result == Decimal("2")

    def test_prob_to_odds_zero(self, validator):
        """Zero prob should raise."""
        with pytest.raises(ValidationError, match="must be > 0"):
            validator.prob_to_odds(Decimal("0"))

    def test_prob_to_odds_negative(self, validator):
        """Negative prob should raise."""
        with pytest.raises(ValidationError, match="must be > 0"):
            validator.prob_to_odds(Decimal("-0.1"))


class TestValidateSubmissionSafe:
    """Tests for safe validation wrappers."""

    def test_valid_returns_value(self):
        """Valid odds should return (value, None)."""
        result, error = validate_submission_safe(2.0)
        assert result == Decimal("2.0")
        assert error is None

    def test_invalid_returns_error(self):
        """Invalid odds should return (None, error)."""
        result, error = validate_submission_safe(float("nan"))
        assert result is None
        assert error is not None
        assert "NaN" in error

    def test_with_custom_validator(self):
        """Should work with custom validator."""
        v = SubmissionValidator()
        result, error = validate_submission_safe(2.0, v)
        assert result == Decimal("2.0")


class TestValidateProbabilitySafe:
    """Tests for safe probability validation."""

    def test_valid_returns_value(self):
        """Valid prob should return (value, None)."""
        result, error = validate_probability_safe(0.5)
        assert result == Decimal("0.5")
        assert error is None

    def test_invalid_returns_error(self):
        """Invalid prob should return (None, error)."""
        result, error = validate_probability_safe(-0.1)
        assert result is None
        assert error is not None

    def test_with_custom_validator(self):
        """Should work with custom validator."""
        v = SubmissionValidator()
        result, error = validate_probability_safe(0.5, v)
        assert result == Decimal("0.5")


class TestGetValidator:
    """Tests for default validator singleton."""

    def test_returns_validator(self):
        """Should return a SubmissionValidator."""
        v = get_validator()
        assert isinstance(v, SubmissionValidator)

    def test_returns_same_instance(self):
        """Should return the same instance."""
        v1 = get_validator()
        v2 = get_validator()
        assert v1 is v2

