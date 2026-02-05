"""Tests for signed attestations."""

import pytest
from datetime import datetime, timezone

from sparket.validator.auditor.attestation import create_attestation, verify_attestation
from sparket.validator.auditor.plugin_registry import TaskResult


@pytest.fixture
def mock_wallet():
    import bittensor as bt
    wallet = bt.Wallet(name="test_attestation", hotkey="test_attest_hk")
    wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    return wallet


class TestAttestation:

    def test_create_verify_roundtrip(self, mock_wallet):
        result = TaskResult(
            plugin_name="weight_verification",
            plugin_version="1.0.0",
            status="pass",
            evidence={"cosine_similarity": 0.999, "match": True},
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        sig = create_attestation(result, mock_wallet)
        result.attestation = sig
        assert verify_attestation(result, mock_wallet.hotkey.ss58_address)

    def test_verify_rejects_tampered(self, mock_wallet):
        result = TaskResult(
            plugin_name="weight_verification",
            plugin_version="1.0.0",
            status="pass",
            evidence={"match": True},
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        sig = create_attestation(result, mock_wallet)
        result.attestation = sig
        # Tamper with evidence
        result.evidence["match"] = False
        assert not verify_attestation(result, mock_wallet.hotkey.ss58_address)

    def test_task_result_serialization(self):
        result = TaskResult(
            plugin_name="test",
            plugin_version="1.0.0",
            status="pass",
            evidence={"key": "value"},
            attestation="abc123",
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert result.plugin_name == "test"
        assert result.status == "pass"
        assert result.evidence["key"] == "value"
        assert result.attestation == "abc123"
