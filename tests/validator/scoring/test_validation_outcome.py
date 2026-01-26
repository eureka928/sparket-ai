import pytest

from sparket.validator.scoring.validation import validate_outcome_result, ValidationError


def test_validate_outcome_result_accepts_valid():
    assert validate_outcome_result("HOME") == "home"
    assert validate_outcome_result("draw") == "draw"
    assert validate_outcome_result(" Over ") == "over"


def test_validate_outcome_result_rejects_invalid():
    with pytest.raises(ValidationError):
        validate_outcome_result("win")
