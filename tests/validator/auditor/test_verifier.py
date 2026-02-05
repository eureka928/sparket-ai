"""Tests for manifest verification."""

import pytest
from datetime import datetime, timezone

from sparket.validator.ledger.models import (
    LEDGER_SCHEMA_VERSION,
    AccumulatorEntry,
    CheckpointWindow,
    DeltaWindow,
    LedgerManifest,
    MinerRosterEntry,
    ScoringConfigSnapshot,
)
from sparket.validator.ledger.signer import compute_section_hash, sign_manifest
from sparket.validator.auditor.verifier import ManifestVerifier


@pytest.fixture
def mock_wallet():
    import bittensor as bt
    wallet = bt.Wallet(name="test_verifier", hotkey="test_verifier_hk")
    wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    return wallet


def _make_signed_checkpoint(wallet) -> CheckpointWindow:
    now = datetime.now(timezone.utc)
    roster = [MinerRosterEntry(miner_id=1, uid=1, hotkey="a", active=True)]
    accumulators = [AccumulatorEntry(miner_id=1, hotkey="a", uid=1)]
    config = ScoringConfigSnapshot(params={"test": True})

    content_hashes = {
        "roster": compute_section_hash(roster),
        "accumulators": compute_section_hash(accumulators),
        "scoring_config": compute_section_hash(config),
    }

    manifest = LedgerManifest(
        window_type="checkpoint",
        window_start=now, window_end=now,
        checkpoint_epoch=1,
        content_hashes=content_hashes,
        primary_hotkey=wallet.hotkey.ss58_address,
        created_at=now,
    )
    manifest.signature = sign_manifest(manifest, wallet)

    return CheckpointWindow(
        manifest=manifest, roster=roster,
        accumulators=accumulators, scoring_config=config,
    )


class TestManifestVerifier:

    def test_valid_window_passes(self, mock_wallet):
        cp = _make_signed_checkpoint(mock_wallet)
        verifier = ManifestVerifier(primary_hotkey=mock_wallet.hotkey.ss58_address)
        result = verifier.verify_checkpoint(cp)
        assert result.valid, f"Errors: {result.errors}"

    def test_tampered_content_fails(self, mock_wallet):
        cp = _make_signed_checkpoint(mock_wallet)
        # Tamper with data after signing
        cp.accumulators.append(AccumulatorEntry(miner_id=99, hotkey="x", uid=99))
        verifier = ManifestVerifier(primary_hotkey=mock_wallet.hotkey.ss58_address)
        result = verifier.verify_checkpoint(cp)
        assert not result.valid
        assert any("content hash mismatch" in e for e in result.errors)

    def test_invalid_signature_fails(self, mock_wallet):
        cp = _make_signed_checkpoint(mock_wallet)
        verifier = ManifestVerifier(primary_hotkey="5WRONG_HOTKEY_ADDRESS_HERE")
        result = verifier.verify_checkpoint(cp)
        assert not result.valid
        assert any("primary_hotkey mismatch" in e or "signature" in e for e in result.errors)

    def test_incompatible_schema_version(self, mock_wallet):
        cp = _make_signed_checkpoint(mock_wallet)
        cp.manifest.schema_version = 999
        verifier = ManifestVerifier(primary_hotkey=mock_wallet.hotkey.ss58_address)
        result = verifier.verify_checkpoint(cp)
        assert not result.valid
        assert any("schema_version" in e for e in result.errors)

    def test_wrong_window_type(self, mock_wallet):
        cp = _make_signed_checkpoint(mock_wallet)
        cp.manifest.window_type = "delta"  # Wrong type for a CheckpointWindow
        verifier = ManifestVerifier(primary_hotkey=mock_wallet.hotkey.ss58_address)
        result = verifier.verify_checkpoint(cp)
        assert not result.valid
        assert any("window_type" in e for e in result.errors)

    def test_missing_content_hash(self, mock_wallet):
        cp = _make_signed_checkpoint(mock_wallet)
        del cp.manifest.content_hashes["roster"]
        # Re-sign with missing hash
        cp.manifest.signature = sign_manifest(cp.manifest, mock_wallet)
        verifier = ManifestVerifier(primary_hotkey=mock_wallet.hotkey.ss58_address)
        result = verifier.verify_checkpoint(cp)
        assert not result.valid
        assert any("missing content hash" in e for e in result.errors)
