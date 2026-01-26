"""Tests for audit hashing utilities."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from sparket.validator.scoring.audit.hashing import (
    compute_hash,
    compute_miner_score_hash,
    compute_batch_hash,
    compute_bias_hash,
    compute_ground_truth_hash,
)


class TestComputeHash:
    """Tests for generic compute_hash function."""

    def test_deterministic(self):
        """Same input should produce same hash."""
        data = {"a": 1, "b": 2}
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)
        assert hash1 == hash2

    def test_key_order_independent(self):
        """Key order should not affect hash."""
        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}
        assert compute_hash(data1) == compute_hash(data2)

    def test_different_data_different_hash(self):
        """Different data should produce different hash."""
        hash1 = compute_hash({"a": 1})
        hash2 = compute_hash({"a": 2})
        assert hash1 != hash2

    def test_sha256_format(self):
        """Should return 64-character hex string."""
        result = compute_hash({"test": "data"})
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_handles_decimal(self):
        """Should serialize Decimal values."""
        data = {"value": Decimal("123.456")}
        result = compute_hash(data)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_handles_datetime(self):
        """Should serialize datetime values."""
        data = {"time": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)}
        result = compute_hash(data)
        assert isinstance(result, str)

    def test_handles_nested(self):
        """Should handle nested structures."""
        data = {
            "nested": {"inner": [1, 2, 3]},
            "list": [{"a": 1}, {"b": 2}],
        }
        result = compute_hash(data)
        assert isinstance(result, str)

    def test_handles_none(self):
        """Should handle None values."""
        data = {"value": None}
        result = compute_hash(data)
        assert isinstance(result, str)

    def test_handles_bool(self):
        """Should handle boolean values."""
        data = {"flag": True, "other": False}
        result = compute_hash(data)
        assert isinstance(result, str)

    def test_handles_custom_object(self):
        """Should convert custom objects to string."""
        class Custom:
            def __str__(self):
                return "custom_value"

        data = {"obj": Custom()}
        result = compute_hash(data)
        assert isinstance(result, str)


class TestComputeMinerScoreHash:
    """Tests for miner score hashing."""

    def test_deterministic(self):
        """Same inputs should produce same hash."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        scores = {"skill_score": Decimal("0.75"), "forecast_dim": Decimal("0.8")}

        hash1 = compute_miner_score_hash(
            miner_id=123,
            miner_hotkey="abc123def456",
            as_of=dt,
            window_days=30,
            scores=scores,
        )
        hash2 = compute_miner_score_hash(
            miner_id=123,
            miner_hotkey="abc123def456",
            as_of=dt,
            window_days=30,
            scores=scores,
        )
        assert hash1 == hash2

    def test_different_miner_different_hash(self):
        """Different miner should produce different hash."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        scores = {"skill_score": 0.75}

        hash1 = compute_miner_score_hash(1, "abc", dt, 30, scores)
        hash2 = compute_miner_score_hash(2, "abc", dt, 30, scores)
        assert hash1 != hash2

    def test_different_scores_different_hash(self):
        """Different scores should produce different hash."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)

        hash1 = compute_miner_score_hash(1, "abc", dt, 30, {"skill": 0.5})
        hash2 = compute_miner_score_hash(1, "abc", dt, 30, {"skill": 0.6})
        assert hash1 != hash2


class TestComputeBatchHash:
    """Tests for batch score hashing."""

    def test_deterministic(self):
        """Same batch should produce same hash."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        scores = [
            {"miner_id": 1, "miner_hotkey": "abc", "skill": 0.5},
            {"miner_id": 2, "miner_hotkey": "def", "skill": 0.6},
        ]

        hash1 = compute_batch_hash(dt, 30, scores)
        hash2 = compute_batch_hash(dt, 30, scores)
        assert hash1 == hash2

    def test_order_independent(self):
        """Miner order should not affect hash (sorted internally)."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)

        scores1 = [
            {"miner_id": 2, "miner_hotkey": "def", "skill": 0.6},
            {"miner_id": 1, "miner_hotkey": "abc", "skill": 0.5},
        ]
        scores2 = [
            {"miner_id": 1, "miner_hotkey": "abc", "skill": 0.5},
            {"miner_id": 2, "miner_hotkey": "def", "skill": 0.6},
        ]

        assert compute_batch_hash(dt, 30, scores1) == compute_batch_hash(dt, 30, scores2)

    def test_empty_batch(self):
        """Empty batch should still produce valid hash."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = compute_batch_hash(dt, 30, [])
        assert isinstance(result, str)
        assert len(result) == 64


class TestComputeBiasHash:
    """Tests for bias hash computation."""

    def test_deterministic(self):
        """Same entries should produce same hash."""
        entries = [
            {"sportsbook_id": 1, "sport_id": 1, "market_kind": "ml", "bias": 1.02},
            {"sportsbook_id": 2, "sport_id": 1, "market_kind": "ml", "bias": 0.98},
        ]

        hash1 = compute_bias_hash(entries)
        hash2 = compute_bias_hash(entries)
        assert hash1 == hash2

    def test_order_independent(self):
        """Entry order should not affect hash."""
        entries1 = [
            {"sportsbook_id": 2, "sport_id": 1, "market_kind": "ml", "bias": 0.98},
            {"sportsbook_id": 1, "sport_id": 1, "market_kind": "ml", "bias": 1.02},
        ]
        entries2 = [
            {"sportsbook_id": 1, "sport_id": 1, "market_kind": "ml", "bias": 1.02},
            {"sportsbook_id": 2, "sport_id": 1, "market_kind": "ml", "bias": 0.98},
        ]

        assert compute_bias_hash(entries1) == compute_bias_hash(entries2)

    def test_empty_entries(self):
        """Empty entries should produce valid hash."""
        result = compute_bias_hash([])
        assert isinstance(result, str)
        assert len(result) == 64


class TestComputeGroundTruthHash:
    """Tests for ground truth hash computation."""

    def test_deterministic(self):
        """Same entries should produce same hash."""
        entries = [
            {"market_id": 1, "side": "home", "prob": 0.6},
            {"market_id": 1, "side": "away", "prob": 0.4},
        ]

        hash1 = compute_ground_truth_hash(entries)
        hash2 = compute_ground_truth_hash(entries)
        assert hash1 == hash2

    def test_order_independent(self):
        """Entry order should not affect hash."""
        entries1 = [
            {"market_id": 1, "side": "away", "prob": 0.4},
            {"market_id": 1, "side": "home", "prob": 0.6},
        ]
        entries2 = [
            {"market_id": 1, "side": "home", "prob": 0.6},
            {"market_id": 1, "side": "away", "prob": 0.4},
        ]

        assert compute_ground_truth_hash(entries1) == compute_ground_truth_hash(entries2)

    def test_empty_entries(self):
        """Empty entries should produce valid hash."""
        result = compute_ground_truth_hash([])
        assert isinstance(result, str)
        assert len(result) == 64

