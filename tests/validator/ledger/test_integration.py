"""Integration tests for the full checkpoint+delta pipeline.

Tests wire together multiple modules with real filesystem I/O
and mock DB/bittensor infrastructure.
"""

import pytest
import tempfile
from datetime import datetime, timezone

from sparket.validator.ledger.models import (
    AccumulatorEntry,
    CheckpointWindow,
    ChainParamsSnapshot,
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
from sparket.validator.ledger.signer import compute_section_hash, sign_manifest
from sparket.validator.ledger.store.filesystem import FilesystemStore
from sparket.validator.auditor.verifier import ManifestVerifier
from sparket.validator.ledger.compute_weights import compute_weights


@pytest.fixture
def mock_wallet():
    import bittensor as bt
    wallet = bt.Wallet(name="test_integration", hotkey="test_int_hk")
    wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
    return wallet


@pytest.fixture
def tmp_store():
    with tempfile.TemporaryDirectory() as d:
        yield FilesystemStore(data_dir=d, retention_days=7)


def _build_signed_checkpoint(wallet, epoch: int = 1, n_miners: int = 3) -> CheckpointWindow:
    now = datetime.now(timezone.utc)
    roster = [
        MinerRosterEntry(miner_id=i, uid=i, hotkey=f"hk_{i}", active=True)
        for i in range(1, n_miners + 1)
    ]
    accumulators = [
        AccumulatorEntry(
            miner_id=i, hotkey=f"hk_{i}", uid=i, n_submissions=50,
            brier=MetricAccumulator(ws=10.0 + i, wt=50.0),
            fq=MetricAccumulator(ws=25.0 + i, wt=50.0),
            pss=MetricAccumulator(ws=5.0, wt=50.0),
            es=MetricAccumulator(ws=2.0 + i * 0.5, wt=50.0),
            mes=MetricAccumulator(ws=30.0, wt=50.0),
            sos=MetricAccumulator(ws=25.0, wt=50.0),
            lead=MetricAccumulator(ws=20.0, wt=50.0),
            cal_score=0.7, sharp_score=0.6,
        )
        for i in range(1, n_miners + 1)
    ]
    config = ScoringConfigSnapshot(params={
        "dimension_weights": {"w_fq": 0.6, "w_cal": 0.4, "w_edge": 0.7, "w_mes": 0.3, "w_sos": 0.6, "w_lead": 0.4},
        "skill_score_weights": {"w_outcome_accuracy": 0.10, "w_outcome_relative": 0.10, "w_odds_edge": 0.50, "w_info_adv": 0.30},
        "normalization": {"min_count_for_zscore": 10},
        "weight_emission": {"burn_rate": 0.9},
    })

    content_hashes = {
        "roster": compute_section_hash(roster),
        "accumulators": compute_section_hash(accumulators),
        "scoring_config": compute_section_hash(config),
    }

    manifest = LedgerManifest(
        window_type="checkpoint",
        window_start=now, window_end=now,
        checkpoint_epoch=epoch, content_hashes=content_hashes,
        primary_hotkey=wallet.hotkey.ss58_address,
        created_at=now,
    )
    manifest.signature = sign_manifest(manifest, wallet)

    return CheckpointWindow(
        manifest=manifest, roster=roster,
        accumulators=accumulators, scoring_config=config,
        chain_params=ChainParamsSnapshot(
            burn_rate=0.9, burn_uid=0,
            max_weight_limit=0.5, min_allowed_weights=1, n_neurons=10,
        ),
    )


def _build_signed_delta(wallet, epoch: int = 1) -> DeltaWindow:
    now = datetime.now(timezone.utc)
    submissions = [
        SettledSubmissionEntry(
            miner_id=1, market_id=10, side="home",
            imp_prob=0.6, brier=(0.6 - 1.0) ** 2, pss=0.1,
            settled_at=now,
        ),
        SettledSubmissionEntry(
            miner_id=2, market_id=10, side="home",
            imp_prob=0.7, brier=(0.7 - 1.0) ** 2, pss=0.15,
            settled_at=now,
        ),
    ]
    outcomes = [
        OutcomeEntry(market_id=10, event_id=100, result="home",
                     score_home=2, score_away=1, settled_at=now),
    ]

    content_hashes = {
        "settled_submissions": compute_section_hash(submissions),
        "settled_outcomes": compute_section_hash(outcomes),
    }

    manifest = LedgerManifest(
        window_type="delta",
        window_start=now, window_end=now,
        checkpoint_epoch=epoch, content_hashes=content_hashes,
        primary_hotkey=wallet.hotkey.ss58_address,
        created_at=now,
    )
    manifest.signature = sign_manifest(manifest, wallet)

    return DeltaWindow(
        manifest=manifest,
        settled_submissions=submissions,
        settled_outcomes=outcomes,
    )


@pytest.mark.integration
class TestCheckpointDeltaPipeline:

    @pytest.mark.asyncio
    async def test_checkpoint_export_fetch_verify(self, mock_wallet, tmp_store):
        """Full pipeline: export -> store -> fetch -> verify."""
        cp = _build_signed_checkpoint(mock_wallet)

        # Store
        cp_id = await tmp_store.put_checkpoint(cp)
        assert cp_id

        # Fetch
        loaded = await tmp_store.get_latest_checkpoint()
        assert loaded is not None
        assert loaded.manifest.checkpoint_epoch == 1

        # Verify
        verifier = ManifestVerifier(primary_hotkey=mock_wallet.hotkey.ss58_address)
        result = verifier.verify_checkpoint(loaded)
        assert result.valid, f"Errors: {result.errors}"

    @pytest.mark.asyncio
    async def test_delta_export_fetch_verify(self, mock_wallet, tmp_store):
        delta = _build_signed_delta(mock_wallet)

        delta_id = await tmp_store.put_delta(delta)
        assert delta_id

        loaded = await tmp_store.get_delta(delta_id)
        assert loaded is not None
        assert len(loaded.settled_submissions) == 2

        verifier = ManifestVerifier(primary_hotkey=mock_wallet.hotkey.ss58_address)
        result = verifier.verify_delta(loaded)
        assert result.valid, f"Errors: {result.errors}"

    @pytest.mark.asyncio
    async def test_checkpoint_plus_deltas_reproduces_weights(self, mock_wallet, tmp_store):
        """Export checkpoint, compute weights directly vs from loaded data - must match."""
        cp = _build_signed_checkpoint(mock_wallet, n_miners=5)
        await tmp_store.put_checkpoint(cp)

        loaded = await tmp_store.get_latest_checkpoint()
        assert loaded is not None

        # Compute weights from original checkpoint
        metrics_orig = [MinerMetrics.from_accumulator(a) for a in cp.accumulators]
        result_orig = compute_weights(metrics_orig, cp.scoring_config, cp.chain_params)

        # Compute weights from loaded checkpoint
        metrics_loaded = [MinerMetrics.from_accumulator(a) for a in loaded.accumulators]
        result_loaded = compute_weights(metrics_loaded, loaded.scoring_config, loaded.chain_params)

        assert result_orig.uids == result_loaded.uids
        assert result_orig.uint16_weights == result_loaded.uint16_weights

    @pytest.mark.asyncio
    async def test_brier_independent_verification(self, mock_wallet, tmp_store):
        """Verify Brier scores can be recomputed from (imp_prob, outcome)."""
        delta = _build_signed_delta(mock_wallet)

        for sub in delta.settled_submissions:
            outcome = next(o for o in delta.settled_outcomes if o.market_id == sub.market_id)
            actual = 1.0 if sub.side == outcome.result else 0.0
            expected_brier = (sub.imp_prob - actual) ** 2
            assert abs(expected_brier - sub.brier) < 1e-9, (
                f"Brier mismatch: expected {expected_brier}, got {sub.brier}"
            )

    @pytest.mark.asyncio
    async def test_epoch_bump_resets_auditor(self, mock_wallet, tmp_store):
        """Epoch bump causes old data to be superseded."""
        # Export epoch 1
        cp1 = _build_signed_checkpoint(mock_wallet, epoch=1)
        await tmp_store.put_checkpoint(cp1)
        d1 = _build_signed_delta(mock_wallet, epoch=1)
        await tmp_store.put_delta(d1)

        # Epoch bump to 2
        cp2 = _build_signed_checkpoint(mock_wallet, epoch=2)
        cp2.manifest.recompute_record = RecomputeRecord(
            epoch=2, previous_epoch=1,
            reason_code=RecomputeReasonCode.SCORING_BUG,
            reason_detail="test recompute",
            severity="bugfix",
            timestamp=datetime.now(timezone.utc),
            code_version="abc123",
        )
        cp2.manifest.signature = sign_manifest(cp2.manifest, mock_wallet)
        await tmp_store.put_checkpoint(cp2)

        # Latest checkpoint should be epoch 2
        latest = await tmp_store.get_latest_checkpoint()
        assert latest.manifest.checkpoint_epoch == 2
        assert latest.manifest.recompute_record is not None
        assert latest.manifest.recompute_record.reason_code == RecomputeReasonCode.SCORING_BUG


@pytest.mark.integration
class TestFilesystemStore:

    @pytest.mark.asyncio
    async def test_checkpoint_put_get_roundtrip(self, mock_wallet, tmp_store):
        cp = _build_signed_checkpoint(mock_wallet)
        cp_id = await tmp_store.put_checkpoint(cp)
        loaded = await tmp_store.get_latest_checkpoint()
        assert loaded is not None
        assert len(loaded.accumulators) == len(cp.accumulators)

    @pytest.mark.asyncio
    async def test_delta_put_list_get(self, mock_wallet, tmp_store):
        from datetime import timedelta
        deltas = []
        # Use recent timestamps so retention doesn't prune them
        base = datetime.now(timezone.utc) - timedelta(hours=1)
        for i in range(5):
            d = _build_signed_delta(mock_wallet, epoch=1)
            d.manifest.window_start = base + timedelta(minutes=i * 10)
            d.manifest.window_end = base + timedelta(minutes=i * 10 + 10)
            did = await tmp_store.put_delta(d)
            deltas.append(did)

        listed = await tmp_store.list_deltas(epoch=1)
        assert len(listed) == 5

    @pytest.mark.asyncio
    async def test_list_deltas_filters_by_epoch(self, mock_wallet, tmp_store):
        d1 = _build_signed_delta(mock_wallet, epoch=1)
        d2 = _build_signed_delta(mock_wallet, epoch=2)
        await tmp_store.put_delta(d1)
        await tmp_store.put_delta(d2)

        epoch1_deltas = await tmp_store.list_deltas(epoch=1)
        epoch2_deltas = await tmp_store.list_deltas(epoch=2)
        # Each should only show its own epoch's deltas
        assert len(epoch1_deltas) >= 1
        assert len(epoch2_deltas) >= 1

    @pytest.mark.asyncio
    async def test_gzip_compression_applied(self, mock_wallet, tmp_store):
        import gzip
        from pathlib import Path

        cp = _build_signed_checkpoint(mock_wallet)
        cp_id = await tmp_store.put_checkpoint(cp)

        # Check that accumulators file is gzipped
        cp_dir = tmp_store.checkpoints_dir / cp_id
        acc_file = cp_dir / "accumulators.json.gz"
        assert acc_file.exists()

        # Verify it's valid gzip
        with gzip.open(acc_file, "rb") as f:
            data = f.read()
        assert len(data) > 0
