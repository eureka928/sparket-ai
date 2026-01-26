"""Tests for weighted consensus probability computation."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from sparket.validator.scoring.ground_truth.bias import BiasState, make_bias_key
from sparket.validator.scoring.ground_truth.consensus import (
    BookQuote,
    ConsensusInput,
    ConsensusComputer,
    compute_simple_average,
)


@pytest.fixture
def now():
    """Current timestamp."""
    return datetime.now(timezone.utc)


@pytest.fixture
def computer():
    """Create consensus computer."""
    return ConsensusComputer()


def make_quote(
    sportsbook_id: int,
    prob: float,
    sport_id: int = 1,
    market_kind: str = "moneyline",
    side: str = "home",
    now: datetime = None,
) -> BookQuote:
    """Helper to create a BookQuote."""
    if now is None:
        now = datetime.now(timezone.utc)
    return BookQuote(
        sportsbook_id=sportsbook_id,
        sport_id=sport_id,
        market_kind=market_kind,
        side=side,
        prob=Decimal(str(prob)),
        odds=Decimal(str(1 / prob)) if prob > 0 else Decimal("100"),
        timestamp=now,
    )


class TestConsensusComputer:
    """Tests for ConsensusComputer class."""

    def test_empty_quotes(self, computer):
        """Empty quotes should return None."""
        result = computer.compute_consensus([], {})
        assert result is None

    def test_insufficient_books(self, computer):
        """Too few books should return None."""
        quotes = [make_quote(1, 0.5)]  # Only 1 book
        result = computer.compute_consensus(quotes, {})
        # Default min_books is 3
        assert result is None

    def test_consensus_with_equal_variance(self, computer):
        """Equal variance should give simple average."""
        quotes = [
            make_quote(1, 0.5),
            make_quote(2, 0.5),
            make_quote(3, 0.5),
        ]
        # All equal variances with default bias
        result = computer.compute_consensus(quotes, {})

        assert result is not None
        assert result["prob_consensus"] == Decimal("0.5")
        assert result["contributing_books"] == 3

    def test_consensus_with_varied_probs(self, computer):
        """Should compute weighted average of different probs."""
        quotes = [
            make_quote(1, 0.4),
            make_quote(2, 0.5),
            make_quote(3, 0.6),
        ]
        result = computer.compute_consensus(quotes, {})

        assert result is not None
        # With equal weights, average is 0.5
        assert result["prob_consensus"] == Decimal("0.5")

    def test_bias_correction_applied(self, computer):
        """Bias should adjust probabilities."""
        quotes = [
            make_quote(1, 0.6),  # Will be adjusted
            make_quote(2, 0.6),
            make_quote(3, 0.6),
        ]
        bias_key = make_bias_key(1, 1, "moneyline")
        bias = BiasState(
            sportsbook_id=1,
            sport_id=1,
            market_kind="moneyline",
            bias_factor=Decimal("1.2"),  # Book overestimates
            variance=Decimal("0.01"),
            mse=Decimal("0.01"),
            sample_count=100,
            version=1,
        )

        result_without = computer.compute_consensus(quotes, {})
        result_with = computer.compute_consensus(quotes, {bias_key: bias})

        # With bias correction, book 1's prob adjusted from 0.6 to 0.5
        # So consensus should differ
        assert result_with is not None
        assert result_without is not None
        assert result_with["prob_consensus"] != result_without["prob_consensus"]

    def test_variance_weighting(self, computer):
        """Lower variance books should have more weight."""
        quotes = [
            make_quote(1, 0.4),
            make_quote(2, 0.4),
            make_quote(3, 0.7),
        ]
        # Book 3 has high variance, should be weighted less
        bias_key3 = make_bias_key(3, 1, "moneyline")
        bias3 = BiasState(3, 1, "moneyline", Decimal("1.0"), Decimal("1.0"), Decimal("0"), 100, 1)

        result = computer.compute_consensus(quotes, {bias_key3: bias3})

        assert result is not None
        # Consensus should be closer to 0.4 (low variance books) than 0.7
        assert result["prob_consensus"] < Decimal("0.55")

    def test_invalid_bias_excluded(self, computer):
        """Books with out-of-range bias should be excluded."""
        quotes = [
            make_quote(1, 0.5),
            make_quote(2, 0.5),
            make_quote(3, 0.5),
            make_quote(4, 0.8),  # Will be excluded
        ]
        bias_key4 = make_bias_key(4, 1, "moneyline")
        # Bias outside valid range [0.5, 2.0]
        bias4 = BiasState(4, 1, "moneyline", Decimal("0.3"), Decimal("0.01"), Decimal("0"), 100, 1)

        result = computer.compute_consensus(quotes, {bias_key4: bias4})

        assert result is not None
        assert result["contributing_books"] == 3  # Only 3 valid

    def test_statistics_computed(self, computer):
        """Should compute min, max, std_dev."""
        quotes = [
            make_quote(1, 0.4),
            make_quote(2, 0.5),
            make_quote(3, 0.6),
        ]
        result = computer.compute_consensus(quotes, {})

        assert result is not None
        assert result["min_prob"] == Decimal("0.4")
        assert result["max_prob"] == Decimal("0.6")
        assert result["std_dev"] > Decimal("0")

    def test_prob_clamped(self, computer):
        """Consensus prob should be clamped to valid range."""
        quotes = [
            make_quote(1, 0.0001),
            make_quote(2, 0.0001),
            make_quote(3, 0.0001),
        ]
        result = computer.compute_consensus(quotes, {})

        assert result is not None
        assert result["prob_consensus"] >= Decimal("0.001")


class TestComputeMarketConsensus:
    """Tests for compute_market_consensus method."""

    @pytest.fixture
    def computer(self):
        return ConsensusComputer()

    def test_multiple_sides(self, computer, now):
        """Should compute consensus for each side."""
        market_quotes = {
            "home": [
                make_quote(1, 0.5, side="home", now=now),
                make_quote(2, 0.5, side="home", now=now),
                make_quote(3, 0.5, side="home", now=now),
            ],
            "away": [
                make_quote(1, 0.5, side="away", now=now),
                make_quote(2, 0.5, side="away", now=now),
                make_quote(3, 0.5, side="away", now=now),
            ],
        }
        results = computer.compute_market_consensus(market_quotes, {})

        assert "home" in results
        assert "away" in results

    def test_normalization(self, computer, now):
        """Probabilities should sum to 1 after normalization."""
        market_quotes = {
            "home": [
                make_quote(1, 0.55, side="home", now=now),
                make_quote(2, 0.55, side="home", now=now),
                make_quote(3, 0.55, side="home", now=now),
            ],
            "away": [
                make_quote(1, 0.55, side="away", now=now),
                make_quote(2, 0.55, side="away", now=now),
                make_quote(3, 0.55, side="away", now=now),
            ],
        }
        results = computer.compute_market_consensus(market_quotes, {})

        prob_sum = sum(r["prob_consensus"] for r in results.values())
        assert abs(prob_sum - Decimal("1")) < Decimal("0.01")

    def test_deterministic_ordering(self, computer, now):
        """Results should be deterministic."""
        market_quotes = {
            "away": [
                make_quote(1, 0.4, side="away", now=now),
                make_quote(2, 0.4, side="away", now=now),
                make_quote(3, 0.4, side="away", now=now),
            ],
            "home": [
                make_quote(1, 0.6, side="home", now=now),
                make_quote(2, 0.6, side="home", now=now),
                make_quote(3, 0.6, side="home", now=now),
            ],
        }
        results1 = computer.compute_market_consensus(market_quotes, {})
        results2 = computer.compute_market_consensus(market_quotes, {})

        assert results1["home"]["prob_consensus"] == results2["home"]["prob_consensus"]


class TestComputeSimpleAverage:
    """Tests for compute_simple_average function."""

    def test_empty_quotes(self):
        """Empty quotes should return None."""
        result = compute_simple_average([])
        assert result is None

    def test_single_quote(self):
        """Single quote should return its prob."""
        quotes = [make_quote(1, 0.5)]
        result = compute_simple_average(quotes)
        assert result == Decimal("0.5")

    def test_average(self):
        """Should compute simple average."""
        quotes = [
            make_quote(1, 0.4),
            make_quote(2, 0.5),
            make_quote(3, 0.6),
        ]
        result = compute_simple_average(quotes)
        assert result == Decimal("0.5")

    def test_rounded(self):
        """Result should be rounded."""
        quotes = [
            make_quote(1, 0.333333333),
            make_quote(2, 0.333333333),
            make_quote(3, 0.333333333),
        ]
        result = compute_simple_average(quotes)
        # Should be rounded to 8 decimal places
        assert result == Decimal("0.33333333")

