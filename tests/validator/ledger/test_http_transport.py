"""HTTP transport integration test.

Spins up a real LedgerHTTPServer on localhost, has an HTTPLedgerStore
client authenticate via challenge-response, and fetch checkpoint + delta
over the wire. Tests the full auth flow, serialization, and gzip
content handling end-to-end.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from sparket.validator.ledger.auth import AccessPolicy
from sparket.validator.ledger.models import (
    AccumulatorEntry,
    CheckpointWindow,
    ChainParamsSnapshot,
    DeltaWindow,
    LedgerManifest,
    MetricAccumulator,
    MinerRosterEntry,
    OutcomeEntry,
    ScoringConfigSnapshot,
    SettledSubmissionEntry,
)
from sparket.validator.ledger.signer import compute_section_hash, sign_manifest
from sparket.validator.ledger.store.filesystem import FilesystemStore
from sparket.validator.ledger.store.http_client import HTTPLedgerStore
from sparket.validator.ledger.store.http_server import LedgerHTTPServer


# ---------------------------------------------------------------------------
# Mock metagraph that recognises the auditor wallet
# ---------------------------------------------------------------------------

class _MockMetagraph:
    """Metagraph that grants vpermit + high stake to a specific hotkey."""

    def __init__(self, allowed_hotkey: str):
        self.hotkeys = [allowed_hotkey, "miner_hk", "low_stake_hk"]
        self.validator_permit = [True, False, True]
        self.S = [200_000, 500_000, 50_000]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def primary_wallet():
    import bittensor as bt
    w = bt.Wallet(name="test_http_primary", hotkey="test_http_primary_hk")
    w.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    return w


@pytest.fixture
def auditor_wallet():
    import bittensor as bt
    w = bt.Wallet(name="test_http_auditor", hotkey="test_http_auditor_hk")
    w.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    return w


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _build_checkpoint(wallet, epoch=1):
    now = datetime.now(timezone.utc)
    roster = [MinerRosterEntry(miner_id=1, uid=1, hotkey="m1", active=True)]
    accumulators = [AccumulatorEntry(
        miner_id=1, hotkey="m1", uid=1, n_submissions=50,
        brier=MetricAccumulator(ws=10.0, wt=50.0),
        fq=MetricAccumulator(ws=30.0, wt=50.0),
        cal_score=0.7, sharp_score=0.6,
    )]
    config = ScoringConfigSnapshot(params={"test": True})

    content_hashes = {
        "roster": compute_section_hash(roster),
        "accumulators": compute_section_hash(accumulators),
        "scoring_config": compute_section_hash(config),
    }

    manifest = LedgerManifest(
        window_type="checkpoint", window_start=now, window_end=now,
        checkpoint_epoch=epoch, content_hashes=content_hashes,
        primary_hotkey=wallet.hotkey.ss58_address, created_at=now,
    )
    manifest.signature = sign_manifest(manifest, wallet)

    return CheckpointWindow(
        manifest=manifest, roster=roster,
        accumulators=accumulators, scoring_config=config,
        chain_params=ChainParamsSnapshot(
            burn_rate=0.9, burn_uid=0, max_weight_limit=0.5,
            min_allowed_weights=1, n_neurons=10,
        ),
    )


def _build_delta(wallet, epoch=1):
    now = datetime.now(timezone.utc)
    submissions = [SettledSubmissionEntry(
        miner_id=1, market_id=100, side="home",
        imp_prob=0.65, brier=(0.65 - 1.0) ** 2, pss=0.1,
        settled_at=now,
    )]
    outcomes = [OutcomeEntry(
        market_id=100, event_id=1000, result="home",
        score_home=2, score_away=1, settled_at=now,
    )]

    content_hashes = {
        "settled_submissions": compute_section_hash(submissions),
        "settled_outcomes": compute_section_hash(outcomes),
    }

    manifest = LedgerManifest(
        window_type="delta",
        window_start=now - timedelta(hours=6), window_end=now,
        checkpoint_epoch=epoch, content_hashes=content_hashes,
        primary_hotkey=wallet.hotkey.ss58_address, created_at=now,
    )
    manifest.signature = sign_manifest(manifest, wallet)

    return DeltaWindow(
        manifest=manifest,
        settled_submissions=submissions,
        settled_outcomes=outcomes,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestHTTPTransport:
    """End-to-end HTTP transport tests with real server + client."""

    async def test_full_auth_and_fetch_flow(
        self, primary_wallet, auditor_wallet, tmp_dir,
    ):
        """
        1. Start HTTP server with AccessPolicy recognising auditor
        2. Client authenticates via challenge-response
        3. Client fetches checkpoint + delta over HTTP
        4. Verify data integrity survives the round-trip
        """
        # -- Setup: seed store with data --
        store = FilesystemStore(data_dir=tmp_dir, retention_days=7)
        cp = _build_checkpoint(primary_wallet)
        await store.put_checkpoint(cp)
        delta = _build_delta(primary_wallet)
        await store.put_delta(delta)

        # -- Setup: server with auth policy recognising auditor --
        metagraph = _MockMetagraph(allowed_hotkey=auditor_wallet.hotkey.ss58_address)
        policy = AccessPolicy(
            metagraph=metagraph,
            min_stake_threshold=100_000,
            token_ttl=300,
        )

        server = LedgerHTTPServer(
            store=store, access_policy=policy,
            host="127.0.0.1", port=18931,  # ephemeral-ish port
        )

        try:
            await server.start()
            # Give the server a moment to bind
            await asyncio.sleep(0.2)

            # -- Client: authenticate and fetch --
            client = HTTPLedgerStore(
                primary_url="http://127.0.0.1:18931",
                wallet=auditor_wallet,
                timeout=10.0,
                max_retries=1,
            )

            try:
                # Fetch checkpoint
                fetched_cp = await client.get_latest_checkpoint()
                assert fetched_cp is not None, "Checkpoint should be returned"
                assert fetched_cp.manifest.checkpoint_epoch == 1
                assert len(fetched_cp.accumulators) == 1
                assert fetched_cp.accumulators[0].miner_id == 1

                # List deltas
                delta_ids = await client.list_deltas(epoch=1)
                assert len(delta_ids) >= 1, "Should list at least 1 delta"

                # Fetch delta
                fetched_delta = await client.get_delta(delta_ids[0])
                assert fetched_delta is not None, "Delta should be returned"
                assert len(fetched_delta.settled_submissions) == 1
                assert fetched_delta.settled_submissions[0].imp_prob == 0.65

            finally:
                await client.close()

        finally:
            await server.stop()

    async def test_auth_rejects_miner_hotkey(
        self, primary_wallet, tmp_dir,
    ):
        """A wallet without vpermit should be rejected at the challenge stage."""
        import bittensor as bt

        miner_wallet = bt.Wallet(name="test_http_miner", hotkey="test_http_miner_hk")
        miner_wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)

        store = FilesystemStore(data_dir=tmp_dir)
        # Metagraph: miner hotkey has vpermit=False
        metagraph = _MockMetagraph(allowed_hotkey="some_other_validator")
        metagraph.hotkeys.append(miner_wallet.hotkey.ss58_address)
        metagraph.validator_permit.append(False)
        metagraph.S.append(999_999)

        policy = AccessPolicy(metagraph=metagraph, min_stake_threshold=100_000)
        server = LedgerHTTPServer(
            store=store, access_policy=policy,
            host="127.0.0.1", port=18932,
        )

        try:
            await server.start()
            await asyncio.sleep(0.2)

            client = HTTPLedgerStore(
                primary_url="http://127.0.0.1:18932",
                wallet=miner_wallet, timeout=5.0, max_retries=1,
            )

            try:
                with pytest.raises(ConnectionError, match="challenge failed"):
                    await client.get_latest_checkpoint()
            finally:
                await client.close()

        finally:
            await server.stop()

    async def test_auth_rejects_low_stake(
        self, primary_wallet, tmp_dir,
    ):
        """A validator with stake below threshold should be rejected."""
        import bittensor as bt

        low_wallet = bt.Wallet(name="test_http_low", hotkey="test_http_low_hk")
        low_wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)

        store = FilesystemStore(data_dir=tmp_dir)
        metagraph = _MockMetagraph(allowed_hotkey="some_other_validator")
        # Add low-stake validator
        metagraph.hotkeys.append(low_wallet.hotkey.ss58_address)
        metagraph.validator_permit.append(True)
        metagraph.S.append(50_000)  # Below 100K threshold

        policy = AccessPolicy(metagraph=metagraph, min_stake_threshold=100_000)
        server = LedgerHTTPServer(
            store=store, access_policy=policy,
            host="127.0.0.1", port=18933,
        )

        try:
            await server.start()
            await asyncio.sleep(0.2)

            client = HTTPLedgerStore(
                primary_url="http://127.0.0.1:18933",
                wallet=low_wallet, timeout=5.0, max_retries=1,
            )

            try:
                with pytest.raises(ConnectionError, match="challenge failed"):
                    await client.get_latest_checkpoint()
            finally:
                await client.close()

        finally:
            await server.stop()

    async def test_data_survives_serialization_roundtrip(
        self, primary_wallet, auditor_wallet, tmp_dir,
    ):
        """Verify numeric precision and structure survive JSON over HTTP."""
        store = FilesystemStore(data_dir=tmp_dir)

        # Build checkpoint with precise floats
        cp = _build_checkpoint(primary_wallet)
        await store.put_checkpoint(cp)

        metagraph = _MockMetagraph(allowed_hotkey=auditor_wallet.hotkey.ss58_address)
        policy = AccessPolicy(metagraph=metagraph, min_stake_threshold=100_000)
        server = LedgerHTTPServer(
            store=store, access_policy=policy,
            host="127.0.0.1", port=18934,
        )

        try:
            await server.start()
            await asyncio.sleep(0.2)

            client = HTTPLedgerStore(
                primary_url="http://127.0.0.1:18934",
                wallet=auditor_wallet, timeout=10.0,
            )

            try:
                fetched = await client.get_latest_checkpoint()
                assert fetched is not None

                # Check accumulator precision
                orig_acc = cp.accumulators[0]
                fetched_acc = fetched.accumulators[0]
                assert fetched_acc.brier.ws == orig_acc.brier.ws
                assert fetched_acc.brier.wt == orig_acc.brier.wt
                assert fetched_acc.cal_score == orig_acc.cal_score

                # Check manifest fields
                assert fetched.manifest.primary_hotkey == cp.manifest.primary_hotkey
                assert fetched.manifest.checkpoint_epoch == cp.manifest.checkpoint_epoch
                assert fetched.manifest.signature == cp.manifest.signature

            finally:
                await client.close()

        finally:
            await server.stop()
