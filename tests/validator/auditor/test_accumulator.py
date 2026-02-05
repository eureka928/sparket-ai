"""Tests for auditor accumulator state management and epoch handling."""

import json
import pytest
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

from sparket.validator.auditor.sync import EpochChangeResult, LedgerSync
from sparket.validator.ledger.models import (
    AccumulatorEntry,
    CheckpointWindow,
    DeltaWindow,
    LedgerManifest,
    MetricAccumulator,
    MinerRosterEntry,
    OutcomeEntry,
    RecomputeReasonCode,
    RecomputeRecord,
    ScoringConfigSnapshot,
    SettledSubmissionEntry,
)


def _make_checkpoint(epoch: int = 1, recompute: RecomputeRecord | None = None) -> CheckpointWindow:
    now = datetime.now(timezone.utc)
    return CheckpointWindow(
        manifest=LedgerManifest(
            window_type="checkpoint",
            window_start=now, window_end=now,
            checkpoint_epoch=epoch, content_hashes={},
            primary_hotkey="primary_hk", created_at=now,
            recompute_record=recompute,
        ),
        roster=[MinerRosterEntry(miner_id=1, uid=1, hotkey="a", active=True)],
        accumulators=[AccumulatorEntry(
            miner_id=1, hotkey="a", uid=1,
            brier=MetricAccumulator(ws=10.0, wt=50.0),
        )],
        scoring_config=ScoringConfigSnapshot(),
    )


def _make_delta(epoch: int = 1) -> DeltaWindow:
    now = datetime.now(timezone.utc)
    return DeltaWindow(
        manifest=LedgerManifest(
            window_type="delta",
            window_start=now, window_end=now,
            checkpoint_epoch=epoch, content_hashes={},
            primary_hotkey="primary_hk", created_at=now,
        ),
        settled_submissions=[
            SettledSubmissionEntry(
                miner_id=1, market_id=10, side="home",
                imp_prob=0.6, brier=0.16, settled_at=now,
            ),
        ],
        settled_outcomes=[
            OutcomeEntry(market_id=10, event_id=100, result="home",
                         score_home=2, score_away=1, settled_at=now),
        ],
    )


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.get_latest_checkpoint = AsyncMock(return_value=None)
    store.list_deltas = AsyncMock(return_value=[])
    store.get_delta = AsyncMock(return_value=None)
    return store


class TestAccumulatorState:

    def test_apply_delta_updates_accumulator(self, tmp_dir, mock_store):
        sync = LedgerSync(store=mock_store, data_dir=tmp_dir)
        sync.epoch = 1

        delta = _make_delta(epoch=1)
        sync._apply_delta(delta)

        assert "1" in sync.accumulator
        acc = sync.accumulator["1"]
        assert acc["brier_ws"] == 0.16
        assert acc["brier_wt"] == 1.0
        assert acc["count"] == 1

    def test_accumulator_accumulates(self, tmp_dir, mock_store):
        sync = LedgerSync(store=mock_store, data_dir=tmp_dir)
        sync.epoch = 1

        sync._apply_delta(_make_delta())
        sync._apply_delta(_make_delta())

        acc = sync.accumulator["1"]
        assert abs(acc["brier_ws"] - 0.32) < 1e-9
        assert acc["count"] == 2

    def test_accumulator_from_checkpoint(self, tmp_dir, mock_store):
        from sparket.validator.ledger.models import MinerMetrics

        sync = LedgerSync(store=mock_store, data_dir=tmp_dir)
        cp = _make_checkpoint(epoch=1)

        metrics = [MinerMetrics.from_accumulator(acc) for acc in cp.accumulators]
        assert len(metrics) == 1
        assert metrics[0].uid == 1
        assert abs(metrics[0].brier_mean - 0.2) < 1e-9  # 10/50

    def test_epoch_bump_resets_state(self, tmp_dir, mock_store):
        sync = LedgerSync(store=mock_store, data_dir=tmp_dir)
        sync.epoch = 1
        sync.accumulator = {"1": {"brier_ws": 5.0, "brier_wt": 25.0, "count": 25}}
        sync.last_delta_id = "d_old"

        cp = _make_checkpoint(epoch=2, recompute=RecomputeRecord(
            epoch=2, previous_epoch=1,
            reason_code=RecomputeReasonCode.SCORING_BUG,
            reason_detail="test recompute",
            severity="bugfix",
            timestamp=datetime.now(timezone.utc),
            code_version="abc",
        ))

        result = sync._handle_epoch_change(cp)
        assert result.status == "accepted"
        assert sync.epoch == 2
        assert sync.accumulator == {}
        assert sync.last_delta_id == ""

    def test_epoch_mismatch_delta_rejected(self, tmp_dir, mock_store):
        sync = LedgerSync(store=mock_store, data_dir=tmp_dir)
        sync.epoch = 6

        delta = _make_delta(epoch=5)
        # In sync_cycle, deltas with wrong epoch are skipped
        assert delta.manifest.checkpoint_epoch != sync.epoch

    def test_epoch_rate_limit_pauses_weights(self, tmp_dir, mock_store):
        sync = LedgerSync(
            store=mock_store, data_dir=tmp_dir,
            max_epoch_bumps_per_day=1,
        )
        sync.epoch = 1

        # First bump - accepted
        cp1 = _make_checkpoint(epoch=2, recompute=RecomputeRecord(
            epoch=2, previous_epoch=1,
            reason_code=RecomputeReasonCode.SCORING_BUG,
            reason_detail="first fix", severity="bugfix",
            timestamp=datetime.now(timezone.utc), code_version="abc",
        ))
        r1 = sync._handle_epoch_change(cp1)
        assert r1.status == "accepted"

        # Second bump within 24h - paused
        cp2 = _make_checkpoint(epoch=3, recompute=RecomputeRecord(
            epoch=3, previous_epoch=2,
            reason_code=RecomputeReasonCode.SCORING_BUG,
            reason_detail="second fix", severity="bugfix",
            timestamp=datetime.now(timezone.utc), code_version="abc",
        ))
        r2 = sync._handle_epoch_change(cp2)
        assert r2.status == "paused"
        assert "RATE_EXCEEDED" in r2.reason

    def test_epoch_rate_limit_configurable(self, tmp_dir, mock_store):
        sync = LedgerSync(
            store=mock_store, data_dir=tmp_dir,
            max_epoch_bumps_per_day=5,
            max_epoch_bumps_per_week=10,
        )
        sync.epoch = 1

        for i in range(5):
            cp = _make_checkpoint(epoch=i + 2, recompute=RecomputeRecord(
                epoch=i + 2, previous_epoch=i + 1,
                reason_code=RecomputeReasonCode.MANUAL_CORRECTION,
                reason_detail=f"bump {i}", severity="correction",
                timestamp=datetime.now(timezone.utc), code_version="abc",
            ))
            result = sync._handle_epoch_change(cp)
            assert result.status == "accepted", f"Bump {i} should be accepted"

    def test_state_persistence(self, tmp_dir, mock_store):
        sync = LedgerSync(store=mock_store, data_dir=tmp_dir)
        sync.epoch = 5
        sync.last_delta_id = "d_test"
        sync.accumulator = {"1": {"brier_ws": 1.0, "brier_wt": 5.0, "count": 5}}
        sync._save_state()

        # Load in new instance
        sync2 = LedgerSync(store=mock_store, data_dir=tmp_dir)
        assert sync2.epoch == 5
        assert sync2.last_delta_id == "d_test"
        assert sync2.accumulator["1"]["count"] == 5

    def test_recompute_history_persisted(self, tmp_dir, mock_store):
        sync = LedgerSync(store=mock_store, data_dir=tmp_dir)
        sync.epoch = 1

        cp = _make_checkpoint(epoch=2, recompute=RecomputeRecord(
            epoch=2, previous_epoch=1,
            reason_code=RecomputeReasonCode.DB_CORRUPTION,
            reason_detail="backup restored",
            severity="recovery",
            timestamp=datetime.now(timezone.utc), code_version="abc",
        ))
        sync._handle_epoch_change(cp)

        # Check history persisted
        sync2 = LedgerSync(store=mock_store, data_dir=tmp_dir)
        assert len(sync2.recompute_history) == 1
        assert sync2.recompute_history[0].reason_code == "DB_CORRUPTION"
