from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from sparket.validator.scoring.ground_truth.snapshot_pipeline import SnapshotPipeline, SnapshotResult
from sparket.validator.config.scoring_params import ScoringParams


@pytest.mark.asyncio
async def test_snapshot_retention_enforced():
    db = MagicMock()
    db.write = AsyncMock()
    params = ScoringParams()
    params.ground_truth.max_snapshots_per_market = 1
    pipeline = SnapshotPipeline(db=db, logger=MagicMock(), params=params)

    now = datetime.now(timezone.utc)
    snaps = [
        SnapshotResult(
            market_id=1,
            side="home",
            snapshot_ts=now,
            prob_consensus=Decimal("0.5"),
            odds_consensus=Decimal("2.0"),
            contributing_books=2,
            std_dev=None,
            bias_version=1,
            is_closing=False,
        )
    ]

    await pipeline._persist_snapshots(snaps)

    assert db.write.call_count == 2
