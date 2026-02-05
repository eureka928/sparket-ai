"""LedgerStore protocol - pluggable transport interface.

Implementations: FilesystemStore (v1), HTTPLedgerStore (auditor client),
future S3Store, etc.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from sparket.validator.ledger.models import CheckpointWindow, DeltaWindow


@runtime_checkable
class LedgerStore(Protocol):
    """Abstract interface for reading/writing ledger data."""

    async def put_checkpoint(self, cp: CheckpointWindow) -> str:
        """Write a checkpoint. Returns the checkpoint ID."""
        ...

    async def put_delta(self, delta: DeltaWindow) -> str:
        """Write a delta. Returns the delta ID."""
        ...

    async def get_latest_checkpoint(self) -> CheckpointWindow | None:
        """Fetch the most recent checkpoint."""
        ...

    async def list_deltas(
        self, epoch: int, since: datetime | None = None,
    ) -> list[str]:
        """List delta IDs for an epoch, optionally filtered by time."""
        ...

    async def get_delta(self, delta_id: str) -> DeltaWindow | None:
        """Fetch a specific delta by ID."""
        ...


__all__ = ["LedgerStore"]
