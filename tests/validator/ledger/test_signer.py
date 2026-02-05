"""Tests for ledger manifest signing and verification."""

import pytest
from datetime import datetime, timezone

from sparket.validator.ledger.models import LedgerManifest
from sparket.validator.ledger.signer import (
    compute_section_hash,
    sign_manifest,
    verify_manifest,
)


def _make_manifest(**overrides) -> LedgerManifest:
    defaults = dict(
        window_type="checkpoint",
        window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        checkpoint_epoch=1,
        content_hashes={"test": "abc123"},
        primary_hotkey="test_hotkey",
        created_at=datetime(2024, 1, 2, 1, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return LedgerManifest(**defaults)


@pytest.fixture
def mock_wallet():
    """Create a mock wallet with real keypair for signing."""
    import bittensor as bt
    wallet = bt.Wallet(name="test_ledger_signer", hotkey="test_ledger_hk")
    wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    return wallet


class TestSigning:

    def test_sign_verify_roundtrip(self, mock_wallet):
        manifest = _make_manifest(primary_hotkey=mock_wallet.hotkey.ss58_address)
        sig = sign_manifest(manifest, mock_wallet)
        manifest.signature = sig
        assert verify_manifest(manifest, mock_wallet.hotkey.ss58_address)

    def test_verify_rejects_tampered_manifest(self, mock_wallet):
        manifest = _make_manifest(primary_hotkey=mock_wallet.hotkey.ss58_address)
        sig = sign_manifest(manifest, mock_wallet)
        manifest.signature = sig
        # Tamper
        manifest.checkpoint_epoch = 999
        assert not verify_manifest(manifest, mock_wallet.hotkey.ss58_address)

    def test_verify_rejects_wrong_hotkey(self, mock_wallet):
        manifest = _make_manifest(primary_hotkey=mock_wallet.hotkey.ss58_address)
        sig = sign_manifest(manifest, mock_wallet)
        manifest.signature = sig
        # Verify with different hotkey
        assert not verify_manifest(manifest, "5GNJqTPyNqANBkUVMN1LPPrxXnFouWA2MRQg3gKrUYgw")

    def test_canonical_serialization_deterministic(self):
        """Same manifest produces same hash regardless of field order."""
        m1 = _make_manifest(content_hashes={"b": "2", "a": "1"})
        m2 = _make_manifest(content_hashes={"a": "1", "b": "2"})
        h1 = compute_section_hash(m1)
        h2 = compute_section_hash(m2)
        assert h1 == h2

    def test_sign_with_real_keypair(self, mock_wallet):
        manifest = _make_manifest(primary_hotkey=mock_wallet.hotkey.ss58_address)
        sig = sign_manifest(manifest, mock_wallet)
        assert isinstance(sig, str)
        assert len(sig) > 0
        # Can parse as hex
        bytes.fromhex(sig)


class TestSectionHash:

    def test_dict_hashing(self):
        h1 = compute_section_hash({"a": 1, "b": 2})
        h2 = compute_section_hash({"b": 2, "a": 1})
        assert h1 == h2  # Order independent

    def test_list_hashing(self):
        from sparket.validator.ledger.models import MinerRosterEntry
        items = [
            MinerRosterEntry(miner_id=1, uid=1, hotkey="a", active=True),
            MinerRosterEntry(miner_id=2, uid=2, hotkey="b", active=False),
        ]
        h1 = compute_section_hash(items)
        h2 = compute_section_hash(items)
        assert h1 == h2

    def test_different_data_different_hash(self):
        h1 = compute_section_hash({"x": 1})
        h2 = compute_section_hash({"x": 2})
        assert h1 != h2
