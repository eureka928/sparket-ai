"""Tests for ledger redaction allowlists and data tier classification."""

import pytest

from sparket.validator.ledger.redaction import (
    DataTier,
    SAFE_ACCUMULATOR_FIELDS,
    SAFE_MINER_FIELDS,
    SAFE_OUTCOME_FIELDS,
    SAFE_ROLLING_SCORE_FIELDS,
    SAFE_SETTLED_SUBMISSION_FIELDS,
    TIER3_FIELD_PATTERNS,
    classify_field,
    contains_tier3,
    redact,
)


class TestRedactionAllowlist:
    """Test the allowlist-based field filtering."""

    def test_safe_rolling_score_fields_only_exports_allowlisted(self):
        row = {
            "miner_id": 1,
            "miner_hotkey": "5abc",
            "fq_raw": 0.5,
            "brier_mean": 0.2,
            "skill_score": 0.7,
            # Internal / Tier 3 fields that should be stripped
            "id": 999,
            "created_at": "2024-01-01",
            "close_odds_eu": 1.95,
            "close_imp_prob": 0.51,
            "ext_ref": {"sportsdataio": {"GameID": 123}},
        }
        result = redact(row, SAFE_ROLLING_SCORE_FIELDS)
        assert "miner_id" in result
        assert "fq_raw" in result
        assert "brier_mean" in result
        assert "skill_score" in result
        assert "id" not in result
        assert "created_at" not in result
        assert "close_odds_eu" not in result
        assert "ext_ref" not in result

    def test_unknown_fields_dropped(self):
        row = {"miner_id": 1, "completely_made_up_field": "should_vanish"}
        result = redact(row, SAFE_MINER_FIELDS)
        assert "miner_id" in result
        assert "completely_made_up_field" not in result

    def test_outcome_fields_exclude_ext_ref(self):
        row = {
            "market_id": 10,
            "result": "home",
            "score_home": 3,
            "score_away": 1,
            "settled_at": "2024-01-01T00:00:00Z",
            "ext_ref": {"sportsdataio": {"GameID": 999}},
            "details": {"some": "internal"},
        }
        result = redact(row, SAFE_OUTCOME_FIELDS)
        assert "market_id" in result
        assert "result" in result
        assert "ext_ref" not in result
        assert "details" not in result

    def test_miner_fields_exclude_internal_ids(self):
        row = {
            "miner_id": 1,
            "uid": 5,
            "hotkey": "5abc",
            "active": True,
            "coldkey": "should_not_appear",
            "stake": 100000,
        }
        result = redact(row, SAFE_MINER_FIELDS)
        assert "uid" in result
        assert "hotkey" in result
        assert "coldkey" not in result
        assert "stake" not in result

    def test_empty_row_returns_empty(self):
        assert redact({}, SAFE_MINER_FIELDS) == {}

    def test_redact_is_idempotent(self):
        row = {"miner_id": 1, "uid": 5, "hotkey": "abc", "active": True, "extra": "gone"}
        first = redact(row, SAFE_MINER_FIELDS)
        second = redact(first, SAFE_MINER_FIELDS)
        assert first == second

    def test_none_values_excluded(self):
        row = {"miner_id": 1, "uid": None, "hotkey": "abc", "active": True}
        result = redact(row, SAFE_MINER_FIELDS)
        assert "uid" not in result
        assert "miner_id" in result

    def test_settled_submission_fields(self):
        row = {
            "miner_id": 1,
            "market_id": 10,
            "side": "home",
            "imp_prob": 0.55,
            "brier": 0.2,
            "pss": 0.1,
            "settled_at": "2024-01-01T00:00:00Z",
            # Tier 3 fields
            "odds_eu": 1.82,
            "submitted_at": "2024-01-01T00:00:00Z",
            "payload": {"model": "v1"},
        }
        result = redact(row, SAFE_SETTLED_SUBMISSION_FIELDS)
        assert "imp_prob" in result
        assert "brier" in result
        assert "odds_eu" not in result
        assert "submitted_at" not in result
        assert "payload" not in result


class TestDataTier:
    """Test data tier classification."""

    def test_tier_classification(self):
        assert classify_field("ext_ref") == DataTier.PRIMARY_ONLY
        assert classify_field("close_odds_eu") == DataTier.PRIMARY_ONLY
        assert classify_field("cle") == DataTier.PRIMARY_ONLY
        assert classify_field("miner_id") == DataTier.VALIDATOR_GATED
        assert classify_field("skill_score") == DataTier.VALIDATOR_GATED
        assert classify_field("brier_mean") == DataTier.VALIDATOR_GATED

    def test_tier3_fields_never_in_tier2_export(self):
        """Exhaustive check: no Tier 3 field appears in any Tier 2 allowlist."""
        tier2_all = (
            SAFE_ROLLING_SCORE_FIELDS
            | SAFE_OUTCOME_FIELDS
            | SAFE_MINER_FIELDS
            | SAFE_SETTLED_SUBMISSION_FIELDS
            | SAFE_ACCUMULATOR_FIELDS
        )
        overlap = TIER3_FIELD_PATTERNS & tier2_all
        assert overlap == set(), f"Tier 3 fields found in Tier 2 allowlists: {overlap}"

    def test_contains_tier3_positive(self):
        assert contains_tier3({"ext_ref": {"sdio": 1}, "miner_id": 1})
        assert contains_tier3({"cle": 0.05})
        assert contains_tier3({"close_odds_eu": 1.95})

    def test_contains_tier3_negative(self):
        assert not contains_tier3({"miner_id": 1, "uid": 5})
        assert not contains_tier3({"brier_mean": 0.2, "skill_score": 0.7})
