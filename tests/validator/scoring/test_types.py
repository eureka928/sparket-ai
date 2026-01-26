"""Tests for scoring types and constants."""

from datetime import datetime
from decimal import Decimal

import pytest

from sparket.validator.scoring.types import (
    DECIMAL_PRECISION,
    DECIMAL_PLACES,
    ValidationError,
    ScoringError,
    BookProbability,
    BiasEstimate,
    ConsensusResult,
    SubmissionMetrics,
    OutcomeMetrics,
    RollingMetrics,
    NormalizedScores,
    CLVResult,
    ProperScoringResult,
    CalibrationResult,
    TimeSeriesCorrelation,
    LeadLagResult,
)


class TestConstants:
    """Tests for module constants."""

    def test_decimal_precision(self):
        """DECIMAL_PRECISION should be a reasonable value."""
        assert isinstance(DECIMAL_PRECISION, int)
        assert DECIMAL_PRECISION >= 15  # Reasonable precision

    def test_decimal_places(self):
        """DECIMAL_PLACES should be positive."""
        assert isinstance(DECIMAL_PLACES, int)
        assert 1 <= DECIMAL_PLACES <= 15


class TestExceptions:
    """Tests for exception classes."""

    def test_validation_error(self):
        """ValidationError should be an Exception."""
        assert issubclass(ValidationError, Exception)
        with pytest.raises(ValidationError):
            raise ValidationError("test")

    def test_scoring_error(self):
        """ScoringError should be an Exception."""
        assert issubclass(ScoringError, Exception)
        with pytest.raises(ScoringError):
            raise ScoringError("test")

    def test_validation_error_message(self):
        """ValidationError should preserve message."""
        try:
            raise ValidationError("custom message")
        except ValidationError as e:
            assert "custom message" in str(e)


class TestTypedDicts:
    """Tests for TypedDict definitions."""

    def test_book_probability_structure(self):
        """BookProbability should have expected keys."""
        bp: BookProbability = {
            "sportsbook_id": 1,
            "prob": Decimal("0.5"),
            "odds": Decimal("2.0"),
            "timestamp": datetime.now(),
        }
        assert bp["sportsbook_id"] == 1

    def test_bias_estimate_structure(self):
        """BiasEstimate should have expected keys."""
        be: BiasEstimate = {
            "sportsbook_id": 1,
            "sport_id": 2,
            "market_kind": "moneyline",
            "bias_factor": Decimal("1.0"),
            "variance": Decimal("0.01"),
            "sample_count": 100,
        }
        assert be["market_kind"] == "moneyline"

    def test_consensus_result_structure(self):
        """ConsensusResult should have expected keys."""
        cr: ConsensusResult = {
            "prob_consensus": Decimal("0.5"),
            "odds_consensus": Decimal("2.0"),
            "contributing_books": 5,
            "min_prob": Decimal("0.45"),
            "max_prob": Decimal("0.55"),
            "std_dev": Decimal("0.02"),
        }
        assert cr["contributing_books"] == 5

    def test_submission_metrics_partial(self):
        """SubmissionMetrics should allow partial data."""
        sm: SubmissionMetrics = {
            "clv_odds": Decimal("0.1"),
        }
        assert sm.get("clv_prob") is None

    def test_outcome_metrics_partial(self):
        """OutcomeMetrics should allow partial data."""
        om: OutcomeMetrics = {
            "brier": Decimal("0.2"),
        }
        assert om.get("logloss") is None


class TestDataclasses:
    """Tests for frozen dataclass definitions."""

    def test_clv_result_frozen(self):
        """CLVResult should be frozen."""
        result = CLVResult(
            clv_odds=Decimal("0.1"),
            clv_prob=Decimal("0.05"),
            cle=Decimal("0.05"),
            minutes_to_close=60,
        )
        with pytest.raises(AttributeError):
            result.clv_odds = Decimal("0.2")

    def test_clv_result_fields(self):
        """CLVResult should have all fields."""
        result = CLVResult(
            clv_odds=Decimal("0.1"),
            clv_prob=Decimal("0.05"),
            cle=Decimal("0.05"),
            minutes_to_close=60,
        )
        assert result.clv_odds == Decimal("0.1")
        assert result.clv_prob == Decimal("0.05")
        assert result.cle == Decimal("0.05")
        assert result.minutes_to_close == 60

    def test_proper_scoring_result_frozen(self):
        """ProperScoringResult should be frozen."""
        result = ProperScoringResult(
            brier_miner=Decimal("0.2"),
            brier_truth=Decimal("0.25"),
            logloss_miner=Decimal("0.5"),
            logloss_truth=Decimal("0.6"),
            pss_brier=Decimal("0.2"),
            pss_log=Decimal("0.17"),
        )
        with pytest.raises(AttributeError):
            result.brier_miner = Decimal("0.3")

    def test_calibration_result_fields(self):
        """CalibrationResult should have all fields."""
        result = CalibrationResult(
            a=Decimal("0.1"),
            b=Decimal("0.9"),
            cal_score=Decimal("0.8"),
            bins_used=8,
        )
        assert result.a == Decimal("0.1")
        assert result.b == Decimal("0.9")
        assert result.cal_score == Decimal("0.8")
        assert result.bins_used == 8

    def test_time_series_correlation_fields(self):
        """TimeSeriesCorrelation should have all fields."""
        result = TimeSeriesCorrelation(
            correlation=Decimal("0.7"),
            sos_score=Decimal("0.3"),
            n_observations=50,
        )
        assert result.correlation == Decimal("0.7")
        assert result.sos_score == Decimal("0.3")
        assert result.n_observations == 50

    def test_lead_lag_result_fields(self):
        """LeadLagResult should have all fields."""
        result = LeadLagResult(
            moves_led=10,
            moves_matched=15,
            lead_ratio=Decimal("0.67"),
            n_truth_moves=20,
        )
        assert result.moves_led == 10
        assert result.moves_matched == 15
        assert result.lead_ratio == Decimal("0.67")
        assert result.n_truth_moves == 20


class TestDataclassEquality:
    """Tests for dataclass equality."""

    def test_clv_result_equality(self):
        """Identical CLVResults should be equal."""
        r1 = CLVResult(
            clv_odds=Decimal("0.1"),
            clv_prob=Decimal("0.05"),
            cle=Decimal("0.05"),
            minutes_to_close=60,
        )
        r2 = CLVResult(
            clv_odds=Decimal("0.1"),
            clv_prob=Decimal("0.05"),
            cle=Decimal("0.05"),
            minutes_to_close=60,
        )
        assert r1 == r2

    def test_clv_result_inequality(self):
        """Different CLVResults should not be equal."""
        r1 = CLVResult(
            clv_odds=Decimal("0.1"),
            clv_prob=Decimal("0.05"),
            cle=Decimal("0.05"),
            minutes_to_close=60,
        )
        r2 = CLVResult(
            clv_odds=Decimal("0.2"),  # Different
            clv_prob=Decimal("0.05"),
            cle=Decimal("0.05"),
            minutes_to_close=60,
        )
        assert r1 != r2

