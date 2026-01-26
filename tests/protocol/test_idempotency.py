"""Tests for protocol/mapping/idempotency.py - Idempotency key generation."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from sparket.protocol.mapping.idempotency import (
    floor_time_to_bucket,
    miner_submission_idempotency_key,
    stable_payload_hash,
    inbox_outcome_dedupe_key,
)


class TestFloorTimeToBucket:
    """Tests for floor_time_to_bucket function."""
    
    def test_floors_to_bucket_boundary(self):
        """Floors datetime to bucket boundary."""
        dt = datetime(2025, 12, 8, 14, 33, 45, tzinfo=timezone.utc)
        result = floor_time_to_bucket(dt, bucket_seconds=60)
        
        # Should floor to 14:33:00
        assert result.second == 0
        assert result.minute == 33
        assert result.hour == 14
    
    def test_handles_naive_datetime(self):
        """Treats naive datetime as UTC."""
        dt = datetime(2025, 12, 8, 14, 33, 45)  # Naive
        result = floor_time_to_bucket(dt, bucket_seconds=60)
        
        # Should still work, result is naive UTC
        assert result.tzinfo is None
        assert result.second == 0
    
    def test_converts_timezone_aware_to_utc(self):
        """Converts timezone-aware datetime to UTC before flooring."""
        # EST is UTC-5
        from datetime import timezone as tz
        est = tz(timedelta(hours=-5))
        dt = datetime(2025, 12, 8, 9, 33, 45, tzinfo=est)  # 9:33 EST = 14:33 UTC
        
        result = floor_time_to_bucket(dt, bucket_seconds=60)
        
        # Should floor to 14:33:00 UTC
        assert result.minute == 33
        assert result.hour == 14
    
    def test_different_bucket_sizes(self):
        """Works with different bucket sizes."""
        dt = datetime(2025, 12, 8, 14, 33, 45, tzinfo=timezone.utc)
        
        # 5-minute buckets
        result = floor_time_to_bucket(dt, bucket_seconds=300)
        assert result.minute == 30
        assert result.second == 0
        
        # 15-minute buckets
        result = floor_time_to_bucket(dt, bucket_seconds=900)
        assert result.minute == 30
    
    def test_exact_bucket_boundary(self):
        """Datetime on exact boundary stays the same."""
        dt = datetime(2025, 12, 8, 14, 30, 0, tzinfo=timezone.utc)
        result = floor_time_to_bucket(dt, bucket_seconds=60)
        
        assert result.minute == 30
        assert result.second == 0
    
    def test_one_second_buckets(self):
        """Works with 1-second buckets (no flooring needed)."""
        dt = datetime(2025, 12, 8, 14, 33, 45, tzinfo=timezone.utc)
        result = floor_time_to_bucket(dt, bucket_seconds=1)
        
        assert result.second == 45


class TestMinerSubmissionIdempotencyKey:
    """Tests for miner_submission_idempotency_key function."""
    
    def test_returns_tuple_with_bucketed_time(self):
        """Returns tuple with bucketed timestamp."""
        dt = datetime(2025, 12, 8, 14, 33, 45, tzinfo=timezone.utc)
        result = miner_submission_idempotency_key(
            miner_id=42,
            miner_hotkey="hotkey123",
            market_id=100,
            side="home",
            submitted_at=dt,
            bucket_seconds=60,
        )
        
        assert result[0] == 42  # miner_id
        assert result[1] == "hotkey123"  # miner_hotkey
        assert result[2] == 100  # market_id
        assert result[3] == "home"  # side
        assert result[4].second == 0  # bucketed time
    
    def test_same_bucket_same_key(self):
        """Same bucket window produces same key."""
        dt1 = datetime(2025, 12, 8, 14, 33, 10, tzinfo=timezone.utc)
        dt2 = datetime(2025, 12, 8, 14, 33, 45, tzinfo=timezone.utc)
        
        key1 = miner_submission_idempotency_key(42, "hk", 100, "home", dt1, 60)
        key2 = miner_submission_idempotency_key(42, "hk", 100, "home", dt2, 60)
        
        assert key1 == key2
    
    def test_different_bucket_different_key(self):
        """Different bucket windows produce different keys."""
        dt1 = datetime(2025, 12, 8, 14, 33, 10, tzinfo=timezone.utc)
        dt2 = datetime(2025, 12, 8, 14, 34, 10, tzinfo=timezone.utc)
        
        key1 = miner_submission_idempotency_key(42, "hk", 100, "home", dt1, 60)
        key2 = miner_submission_idempotency_key(42, "hk", 100, "home", dt2, 60)
        
        assert key1 != key2
    
    def test_different_miner_different_key(self):
        """Different miner produces different key."""
        dt = datetime(2025, 12, 8, 14, 33, 10, tzinfo=timezone.utc)
        
        key1 = miner_submission_idempotency_key(42, "hk", 100, "home", dt, 60)
        key2 = miner_submission_idempotency_key(43, "hk", 100, "home", dt, 60)
        
        assert key1 != key2
    
    def test_different_market_different_key(self):
        """Different market produces different key."""
        dt = datetime(2025, 12, 8, 14, 33, 10, tzinfo=timezone.utc)
        
        key1 = miner_submission_idempotency_key(42, "hk", 100, "home", dt, 60)
        key2 = miner_submission_idempotency_key(42, "hk", 101, "home", dt, 60)
        
        assert key1 != key2
    
    def test_different_side_different_key(self):
        """Different side produces different key."""
        dt = datetime(2025, 12, 8, 14, 33, 10, tzinfo=timezone.utc)
        
        key1 = miner_submission_idempotency_key(42, "hk", 100, "home", dt, 60)
        key2 = miner_submission_idempotency_key(42, "hk", 100, "away", dt, 60)
        
        assert key1 != key2


class TestStablePayloadHash:
    """Tests for stable_payload_hash function."""
    
    def test_produces_sha256_hex(self):
        """Produces 64-character SHA256 hex digest."""
        result = stable_payload_hash({"key": "value"})
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
    
    def test_deterministic(self):
        """Same payload produces same hash."""
        payload = {"foo": "bar", "num": 123}
        hash1 = stable_payload_hash(payload)
        hash2 = stable_payload_hash(payload)
        assert hash1 == hash2
    
    def test_key_order_independent(self):
        """Hash is independent of key order."""
        payload1 = {"a": 1, "b": 2, "c": 3}
        payload2 = {"c": 3, "a": 1, "b": 2}
        
        assert stable_payload_hash(payload1) == stable_payload_hash(payload2)
    
    def test_different_values_different_hash(self):
        """Different values produce different hashes."""
        hash1 = stable_payload_hash({"key": "value1"})
        hash2 = stable_payload_hash({"key": "value2"})
        assert hash1 != hash2
    
    def test_handles_nested_objects(self):
        """Handles nested objects correctly."""
        payload = {
            "outer": {
                "inner": {
                    "value": 123
                }
            }
        }
        result = stable_payload_hash(payload)
        assert len(result) == 64
    
    def test_handles_arrays(self):
        """Handles arrays in payload."""
        payload = {"items": [1, 2, 3, 4, 5]}
        result = stable_payload_hash(payload)
        assert len(result) == 64
    
    def test_empty_payload(self):
        """Handles empty payload."""
        result = stable_payload_hash({})
        assert len(result) == 64


class TestInboxOutcomeDedupeKey:
    """Tests for inbox_outcome_dedupe_key function."""
    
    def test_format(self):
        """Produces correctly formatted dedupe key."""
        dt = datetime(2025, 12, 8, 14, 33, 45, tzinfo=timezone.utc)
        result = inbox_outcome_dedupe_key("event_123", "hotkey_abc", dt, 300)
        
        assert result.startswith("outcome:")
        assert "event_123" in result
        assert "hotkey_abc" in result
    
    def test_same_bucket_same_key(self):
        """Same bucket window produces same key."""
        dt1 = datetime(2025, 12, 8, 14, 31, 10, tzinfo=timezone.utc)  # In 14:30:00 bucket
        dt2 = datetime(2025, 12, 8, 14, 34, 59, tzinfo=timezone.utc)  # Also in 14:30:00 bucket
        
        key1 = inbox_outcome_dedupe_key("e1", "hk", dt1, 300)
        key2 = inbox_outcome_dedupe_key("e1", "hk", dt2, 300)
        
        assert key1 == key2
    
    def test_different_bucket_different_key(self):
        """Different bucket windows produce different keys."""
        dt1 = datetime(2025, 12, 8, 14, 30, 0, tzinfo=timezone.utc)
        dt2 = datetime(2025, 12, 8, 14, 35, 0, tzinfo=timezone.utc)
        
        key1 = inbox_outcome_dedupe_key("e1", "hk", dt1, 300)
        key2 = inbox_outcome_dedupe_key("e1", "hk", dt2, 300)
        
        assert key1 != key2
    
    def test_different_event_different_key(self):
        """Different events produce different keys."""
        dt = datetime(2025, 12, 8, 14, 33, 0, tzinfo=timezone.utc)
        
        key1 = inbox_outcome_dedupe_key("event_1", "hk", dt, 300)
        key2 = inbox_outcome_dedupe_key("event_2", "hk", dt, 300)
        
        assert key1 != key2
    
    def test_different_hotkey_different_key(self):
        """Different hotkeys produce different keys."""
        dt = datetime(2025, 12, 8, 14, 33, 0, tzinfo=timezone.utc)
        
        key1 = inbox_outcome_dedupe_key("e1", "hotkey_1", dt, 300)
        key2 = inbox_outcome_dedupe_key("e1", "hotkey_2", dt, 300)
        
        assert key1 != key2
    
    def test_handles_int_event_id(self):
        """Works with integer event IDs."""
        dt = datetime(2025, 12, 8, 14, 33, 0, tzinfo=timezone.utc)
        result = inbox_outcome_dedupe_key(12345, "hk", dt, 300)
        
        assert "12345" in result
