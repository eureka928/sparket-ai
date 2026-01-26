"""Input validation for scoring system.

All validation happens BEFORE data enters the scoring pipeline.
Invalid submissions are logged and rejected, not scored.

This module provides defense against adversarial miner inputs:
- NaN, Inf, extreme values
- Out-of-bounds odds/probabilities
- Malformed probability vectors
"""

from __future__ import annotations

from decimal import Decimal
from typing import List, Tuple

from sparket.validator.config.scoring_params import ScoringParams, get_scoring_params

from .determinism import to_decimal
from .types import ValidationError


class SubmissionValidator:
    """Validate miner submissions before scoring.

    All validation is stateless and deterministic.
    """

    def __init__(self, params: ScoringParams | None = None):
        """Initialize validator with scoring parameters."""
        self.params = params or get_scoring_params()
        self.bounds = self.params.security

    def validate_odds(self, odds: object) -> Decimal:
        """Validate and convert odds to Decimal.

        Rejects:
        - NaN, Inf, -Inf
        - Non-numeric types
        - Out of bounds (< odds_min or > odds_max)
        - Negative values

        Args:
            odds: Raw odds value

        Returns:
            Validated Decimal odds

        Raises:
            ValidationError: If odds are invalid
        """
        d = to_decimal(odds, "odds")

        if d < self.bounds.odds_min:
            raise ValidationError(f"odds {d} < min {self.bounds.odds_min}")
        if d > self.bounds.odds_max:
            raise ValidationError(f"odds {d} > max {self.bounds.odds_max}")

        return d

    def validate_probability(self, prob: object) -> Decimal:
        """Validate and convert probability to Decimal.

        Rejects:
        - NaN, Inf, -Inf
        - Non-numeric types
        - Out of bounds (< prob_min or > prob_max)

        Args:
            prob: Raw probability value

        Returns:
            Validated Decimal probability

        Raises:
            ValidationError: If probability is invalid
        """
        d = to_decimal(prob, "probability")

        if d < self.bounds.prob_min:
            raise ValidationError(f"probability {d} < min {self.bounds.prob_min}")
        if d > self.bounds.prob_max:
            raise ValidationError(f"probability {d} > max {self.bounds.prob_max}")

        return d

    def validate_probability_vector(
        self,
        probs: List[object],
    ) -> Tuple[List[Decimal], Decimal]:
        """Validate a vector of probabilities.

        Checks:
        - All individual probabilities valid
        - Sum is reasonable (1.0 ± tolerance)
        - At least 2 outcomes

        Args:
            probs: List of raw probability values

        Returns:
            (validated_probs, overround)

        Raises:
            ValidationError: If vector is invalid
        """
        if not probs:
            raise ValidationError("probability vector is empty")
        if len(probs) < 2:
            raise ValidationError("probability vector must have >= 2 outcomes")

        validated = []
        for i, p in enumerate(probs):
            try:
                validated.append(self.validate_probability(p))
            except ValidationError as e:
                raise ValidationError(f"probability[{i}]: {e}")

        total = sum(validated)
        deviation = abs(total - Decimal("1"))

        if deviation > self.bounds.prob_sum_tolerance:
            raise ValidationError(
                f"probability vector sum {total} deviates from 1.0 by {deviation}"
            )

        overround = total - Decimal("1")
        return validated, overround

    def validate_cle(self, cle: Decimal) -> Decimal:
        """Validate and clamp CLE to prevent gaming.

        Extreme CLE values (from extreme odds) are clamped.

        Args:
            cle: Raw CLE value

        Returns:
            Clamped CLE
        """
        return max(self.bounds.cle_min, min(cle, self.bounds.cle_max))

    def odds_to_prob(self, odds: Decimal) -> Decimal:
        """Convert validated odds to implied probability."""
        return Decimal("1") / odds

    def prob_to_odds(self, prob: Decimal) -> Decimal:
        """Convert validated probability to decimal odds."""
        if prob <= Decimal("0"):
            raise ValidationError("probability must be > 0 for odds conversion")
        return Decimal("1") / prob


def validate_outcome_result(value: object) -> str:
    """Validate and normalize an outcome result string."""
    if value is None:
        raise ValidationError("result is None")
    text = str(value).strip().lower()
    valid = {"home", "away", "draw", "over", "under", "push", "void"}
    if text not in valid:
        raise ValidationError(f"invalid outcome result: {text}")
    return text


def validate_submission_safe(
    odds: object,
    validator: SubmissionValidator | None = None,
) -> Tuple[Decimal | None, str | None]:
    """Safely validate odds, returning None on failure.

    Args:
        odds: Raw odds value
        validator: Optional validator instance

    Returns:
        (validated_odds, error_message) - one will be None
    """
    if validator is None:
        validator = SubmissionValidator()

    try:
        return validator.validate_odds(odds), None
    except ValidationError as e:
        return None, str(e)


def validate_probability_safe(
    prob: object,
    validator: SubmissionValidator | None = None,
) -> Tuple[Decimal | None, str | None]:
    """Safely validate probability, returning None on failure.

    Args:
        prob: Raw probability value
        validator: Optional validator instance

    Returns:
        (validated_prob, error_message) - one will be None
    """
    if validator is None:
        validator = SubmissionValidator()

    try:
        return validator.validate_probability(prob), None
    except ValidationError as e:
        return None, str(e)


# Default validator instance
_default_validator: SubmissionValidator | None = None


def get_validator() -> SubmissionValidator:
    """Get or create the default submission validator."""
    global _default_validator
    if _default_validator is None:
        _default_validator = SubmissionValidator()
    return _default_validator


__all__ = [
    "SubmissionValidator",
    "validate_submission_safe",
    "validate_probability_safe",
    "validate_outcome_result",
    "get_validator",
]

