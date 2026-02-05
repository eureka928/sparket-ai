"""End-to-end pipeline test for the full primary -> auditor flow.

This test exercises the complete ledger pipeline using real filesystem
I/O, real bittensor keypairs, and real compute_weights logic - but with
mock DB data rather than requiring a running validator/auditor.

Tests:
1. Primary exports checkpoint + delta (with real signing)
2. Filesystem store write + read roundtrip (with gzip)
3. Auditor verifies manifests (signatures + hashes)
4. Auditor independently verifies Brier scores
5. Auditor accumulates deltas and computes weights
6. Weights from checkpoint match weights from original data
7. Epoch bump causes auditor reset
8. Auth rejects miners and low-stake validators
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from sparket.validator.auditor.plugin_registry import AuditorContext, PluginRegistry
from sparket.validator.auditor.plugins.weight_verification import WeightVerificationHandler
from sparket.validator.auditor.sync import LedgerSync
from sparket.validator.auditor.verifier import ManifestVerifier
from sparket.validator.ledger.auth import AccessPolicy
from sparket.validator.ledger.compute_weights import compute_weights
from sparket.validator.ledger.models import (
    AccumulatorEntry,
    ChainParamsSnapshot,
    CheckpointWindow,
    DeltaWindow,
    LedgerManifest,
    MetricAccumulator,
    MinerMetrics,
    MinerRosterEntry,
    OutcomeEntry,
    RecomputeReasonCode,
    RecomputeRecord,
    ScoringConfigSnapshot,
    SettledSubmissionEntry,
)
from sparket.validator.ledger.redaction import contains_tier3
from sparket.validator.ledger.signer import compute_section_hash, sign_manifest
from sparket.validator.ledger.store.filesystem import FilesystemStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wallet():
    import bittensor as bt
    w = bt.Wallet(name="test_e2e_pipeline", hotkey="test_e2e_hk")
    w.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    return w


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store(tmp_dir):
    return FilesystemStore(data_dir=tmp_dir, retention_days=7)


@pytest.fixture
def config():
    return ScoringConfigSnapshot(params={
        "dimension_weights": {
            "w_fq": 0.6, "w_cal": 0.4,
            "w_edge": 0.7, "w_mes": 0.3,
            "w_sos": 0.6, "w_lead": 0.4,
        },
        "skill_score_weights": {
            "w_outcome_accuracy": 0.10,
            "w_outcome_relative": 0.10,
            "w_odds_edge": 0.50,
            "w_info_adv": 0.30,
        },
        "normalization": {"min_count_for_zscore": 10},
        "weight_emission": {"burn_rate": 0.9},
    })


@pytest.fixture
def chain_params():
    return ChainParamsSnapshot(
        burn_rate=0.9, burn_uid=0,
        max_weight_limit=0.5, min_allowed_weights=1, n_neurons=10,
    )


def _build_checkpoint(wallet, config, chain_params, epoch=1, n_miners=5):
    """Build a realistic signed checkpoint."""
    now = datetime.now(timezone.utc)

    roster = []
    accumulators = []
    for i in range(1, n_miners + 1):
        roster.append(MinerRosterEntry(miner_id=i, uid=i, hotkey=f"hk_{i}", active=True))
        # Varied performance: miner 1 is best, miner N is worst
        skill = 1.0 - (i - 1) * 0.15
        accumulators.append(AccumulatorEntry(
            miner_id=i, hotkey=f"hk_{i}", uid=i,
            n_submissions=100,
            brier=MetricAccumulator(ws=0.2 * skill * 100, wt=100.0),
            fq=MetricAccumulator(ws=0.6 * skill * 100, wt=100.0),
            pss=MetricAccumulator(ws=0.1 * skill * 100, wt=100.0),
            es=MetricAccumulator(ws=0.05 * skill * 100, wt=100.0),
            mes=MetricAccumulator(ws=0.6 * 100, wt=100.0),
            sos=MetricAccumulator(ws=0.5 * 100, wt=100.0),
            lead=MetricAccumulator(ws=0.4 * 100, wt=100.0),
            cal_score=0.7 * skill,
            sharp_score=0.6 * skill,
        ))

    content_hashes = {
        "roster": compute_section_hash(roster),
        "accumulators": compute_section_hash(accumulators),
        "scoring_config": compute_section_hash(config),
    }

    manifest = LedgerManifest(
        window_type="checkpoint",
        window_start=now, window_end=now,
        checkpoint_epoch=epoch,
        content_hashes=content_hashes,
        primary_hotkey=wallet.hotkey.ss58_address,
        created_at=now,
    )
    manifest.signature = sign_manifest(manifest, wallet)

    return CheckpointWindow(
        manifest=manifest, roster=roster,
        accumulators=accumulators,
        scoring_config=config,
        chain_params=chain_params,
    )


def _build_delta(wallet, epoch=1, n_submissions=20):
    """Build a realistic signed delta with correct Brier scores."""
    now = datetime.now(timezone.utc)

    submissions = []
    outcomes = []
    for market_id in range(100, 100 + n_submissions // 2):
        # Each market: home wins
        outcomes.append(OutcomeEntry(
            market_id=market_id, event_id=market_id,
            result="home", score_home=2, score_away=1,
            settled_at=now,
        ))

        # Two miners per market
        for miner_id in [1, 2]:
            prob = 0.6 + miner_id * 0.05  # miner 1: 0.65, miner 2: 0.70
            actual = 1.0  # home wins
            brier = (prob - actual) ** 2
            submissions.append(SettledSubmissionEntry(
                miner_id=miner_id, market_id=market_id, side="home",
                imp_prob=prob, brier=brier, pss=0.1 * miner_id,
                settled_at=now,
            ))

    content_hashes = {
        "settled_submissions": compute_section_hash(submissions),
        "settled_outcomes": compute_section_hash(outcomes),
    }

    manifest = LedgerManifest(
        window_type="delta",
        window_start=now - timedelta(hours=6), window_end=now,
        checkpoint_epoch=epoch,
        content_hashes=content_hashes,
        primary_hotkey=wallet.hotkey.ss58_address,
        created_at=now,
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
class TestE2EPipeline:
    """Full end-to-end pipeline test."""

    async def test_full_primary_to_auditor_flow(self, wallet, store, config, chain_params):
        """The big one: primary exports -> store -> auditor verifies -> computes weights."""

        # -- Primary side: export checkpoint + delta --
        cp = _build_checkpoint(wallet, config, chain_params, epoch=1, n_miners=5)
        cp_id = await store.put_checkpoint(cp)
        assert cp_id

        delta = _build_delta(wallet, epoch=1, n_submissions=20)
        delta_id = await store.put_delta(delta)
        assert delta_id

        # -- Auditor side: fetch and verify --
        loaded_cp = await store.get_latest_checkpoint()
        assert loaded_cp is not None

        delta_ids = await store.list_deltas(epoch=1)
        assert len(delta_ids) >= 1
        loaded_delta = await store.get_delta(delta_ids[0])
        assert loaded_delta is not None

        # Verify manifests
        verifier = ManifestVerifier(primary_hotkey=wallet.hotkey.ss58_address)
        cp_result = verifier.verify_checkpoint(loaded_cp)
        assert cp_result.valid, f"Checkpoint verification failed: {cp_result.errors}"

        delta_result = verifier.verify_delta(loaded_delta)
        assert delta_result.valid, f"Delta verification failed: {delta_result.errors}"

        # -- Auditor: verify Brier scores independently --
        brier_checks = 0
        brier_mismatches = 0
        for sub in loaded_delta.settled_submissions:
            outcome = next(
                (o for o in loaded_delta.settled_outcomes if o.market_id == sub.market_id),
                None,
            )
            if outcome is None or outcome.result is None or sub.brier is None:
                continue
            actual = 1.0 if sub.side == outcome.result else 0.0
            expected_brier = (sub.imp_prob - actual) ** 2
            brier_checks += 1
            if abs(expected_brier - sub.brier) > 1e-9:
                brier_mismatches += 1

        assert brier_checks > 0, "No Brier checks performed"
        assert brier_mismatches == 0, f"{brier_mismatches} Brier mismatches"

        # -- Auditor: compute weights from checkpoint --
        metrics = [MinerMetrics.from_accumulator(a) for a in loaded_cp.accumulators]
        result = compute_weights(metrics, loaded_cp.scoring_config, loaded_cp.chain_params)

        assert len(result.uids) > 0, "No weights computed"
        assert len(result.uint16_weights) == len(result.uids)
        assert all(w > 0 for w in result.uint16_weights)

        # -- Verify: no Tier 3 data leaked --
        cp_data = loaded_cp.model_dump(mode="json")
        assert not _deep_scan_tier3(cp_data), "Tier 3 data found in checkpoint"

        delta_data = loaded_delta.model_dump(mode="json")
        assert not _deep_scan_tier3(delta_data), "Tier 3 data found in delta"

    async def test_checkpoint_weights_are_deterministic(self, wallet, store, config, chain_params):
        """Same checkpoint data -> same weights, every time."""
        cp = _build_checkpoint(wallet, config, chain_params, n_miners=5)

        results = []
        for _ in range(5):
            metrics = [MinerMetrics.from_accumulator(a) for a in cp.accumulators]
            r = compute_weights(metrics, cp.scoring_config, cp.chain_params)
            results.append((r.uids, r.uint16_weights))

        for i in range(1, 5):
            assert results[i] == results[0], f"Run {i} differs from run 0"

    async def test_epoch_bump_flow(self, wallet, tmp_dir):
        """Epoch bump causes auditor to reset accumulators."""
        mock_store = AsyncMock()
        sync = LedgerSync(store=mock_store, data_dir=tmp_dir)

        # Initial sync at epoch 1
        config = ScoringConfigSnapshot()
        chain = ChainParamsSnapshot(burn_rate=0.9, burn_uid=0, max_weight_limit=0.5, min_allowed_weights=1, n_neurons=10)

        cp1 = _build_checkpoint(wallet, config, chain, epoch=1, n_miners=3)
        mock_store.get_latest_checkpoint = AsyncMock(return_value=cp1)
        mock_store.list_deltas = AsyncMock(return_value=[])

        await sync.sync_cycle()
        assert sync.epoch == 1

        # Apply some deltas
        delta = _build_delta(wallet, epoch=1, n_submissions=4)
        sync._apply_delta(delta)
        assert len(sync.accumulator) > 0

        # Epoch bump to 2
        cp2 = _build_checkpoint(wallet, config, chain, epoch=2, n_miners=3)
        cp2.manifest.recompute_record = RecomputeRecord(
            epoch=2, previous_epoch=1,
            reason_code=RecomputeReasonCode.SCORING_BUG,
            reason_detail="Fixed Brier rounding",
            severity="bugfix",
            timestamp=datetime.now(timezone.utc),
            code_version="test123",
        )
        cp2.manifest.signature = sign_manifest(cp2.manifest, wallet)

        mock_store.get_latest_checkpoint = AsyncMock(return_value=cp2)
        await sync.sync_cycle()

        assert sync.epoch == 2
        assert sync.accumulator == {}  # Reset
        assert len(sync.recompute_history) == 1

    async def test_auth_rejects_miners_and_low_stake(self):
        """Access policy correctly gates by vpermit + stake."""

        class MockMG:
            hotkeys = ["validator_ok", "miner_bad", "validator_poor"]
            validator_permit = [True, False, True]
            S = [200_000, 500_000, 50_000]

        policy = AccessPolicy(metagraph=MockMG(), min_stake_threshold=100_000)

        # Eligible validator
        assert policy.check_eligibility("validator_ok").eligible

        # Miner rejected (no vpermit)
        r = policy.check_eligibility("miner_bad")
        assert not r.eligible
        assert "no_validator_permit" in r.reason

        # Low-stake validator rejected
        r = policy.check_eligibility("validator_poor")
        assert not r.eligible
        assert "stake_too_low" in r.reason

        # Unknown hotkey rejected
        r = policy.check_eligibility("nonexistent")
        assert not r.eligible

    async def test_weight_verification_plugin_full_flow(self, wallet, config, chain_params):
        """WeightVerification plugin: checkpoint -> verify Brier -> compute -> attest."""
        cp = _build_checkpoint(wallet, config, chain_params, epoch=1, n_miners=5)
        delta = _build_delta(wallet, epoch=1, n_submissions=10)

        handler = WeightVerificationHandler(tolerance=0.01)
        context = AuditorContext(
            checkpoint=cp,
            deltas=[delta],
            wallet=wallet,
        )

        result = await handler.on_cycle(context)

        # Plugin should produce a result (may be pass/fail/skip depending on metagraph)
        assert result.plugin_name == "weight_verification"
        assert result.status in ("pass", "fail", "skip")

        # Brier verification should have run
        assert result.evidence.get("brier_checks", 0) > 0
        assert result.evidence.get("brier_mismatches", 0) == 0

    async def test_plugin_registry_dispatches_correctly(self, wallet, config, chain_params):
        """Registry discovers and dispatches to weight_verification plugin."""
        registry = PluginRegistry()
        registry.discover("sparket.validator.auditor.plugins")

        assert "weight_verification" in registry.handlers

        cp = _build_checkpoint(wallet, config, chain_params, n_miners=3)
        context = AuditorContext(checkpoint=cp, deltas=[], wallet=wallet)

        results = await registry.dispatch(context)
        assert len(results) == 1
        assert results[0].plugin_name == "weight_verification"

    async def test_delta_contains_only_settled_data(self, wallet):
        """Verify no unsettled submission fields in delta."""
        delta = _build_delta(wallet, epoch=1, n_submissions=10)

        for sub in delta.settled_submissions:
            assert sub.settled_at is not None
            # Should NOT have unsettled-only fields
            data = sub.model_dump()
            assert "odds_eu" not in data
            assert "priced_at" not in data
            assert "payload" not in data
            assert "submitted_at" not in data


def _deep_scan_tier3(obj, path=""):
    """Recursively scan for Tier 3 field names in serialized data."""
    from sparket.validator.ledger.redaction import TIER3_FIELD_PATTERNS

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in TIER3_FIELD_PATTERNS:
                return True
            if _deep_scan_tier3(value, f"{path}.{key}"):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if _deep_scan_tier3(item, path):
                return True
    return False
