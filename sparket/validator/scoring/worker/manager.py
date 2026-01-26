"""Worker process lifecycle manager.

Manages spawning, monitoring, and graceful shutdown of scoring worker processes.
The manager runs in the main validator process and communicates with workers
via database tables (heartbeat, job state).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text

from sparket.validator.config.scoring_params import get_scoring_params

logger = logging.getLogger(__name__)


@dataclass
class WorkerSlot:
    """Tracking info for a worker process slot."""
    worker_id: str
    process: Optional[mp.Process] = None
    restart_count: int = 0


class ScoringWorkerManager:
    """Manages lifecycle of scoring worker process(es).

    Responsibilities:
    - Spawn worker process on validator startup
    - Monitor worker health via heartbeat table
    - Restart worker if it dies unexpectedly
    - Graceful shutdown on SIGTERM/SIGINT
    """

    # Default configuration
    HEARTBEAT_TIMEOUT_SEC = 60
    RESTART_DELAY_SEC = 5
    MAX_RESTART_ATTEMPTS = 10

    def __init__(self, config: Any, database: Any, *, worker_count: int = 1):
        """Initialize the worker manager.

        Args:
            config: Validator configuration
            database: Database manager for heartbeat monitoring
        """
        self.config = config
        self.database = database
        self.worker_count = max(int(worker_count), 1)
        self.workers: list[WorkerSlot] = []
        self.heartbeat_timeout_sec = self.HEARTBEAT_TIMEOUT_SEC
        self.restart_delay_sec = self.RESTART_DELAY_SEC
        self.max_restart_attempts = self.MAX_RESTART_ATTEMPTS
        self._shutdown_requested = False

        try:
            params = get_scoring_params()
            self.heartbeat_timeout_sec = params.worker.heartbeat_timeout_sec
            self.restart_delay_sec = params.worker.restart_delay_sec
            self.max_restart_attempts = params.worker.max_restart_attempts
        except Exception:
            pass

        for idx in range(self.worker_count):
            self.workers.append(WorkerSlot(worker_id=self._new_worker_id(idx)))

    def start(self) -> None:
        """Start the scoring worker process."""
        if self._shutdown_requested:
            logger.warning("Cannot start worker: shutdown requested")
            return
        for slot in self.workers:
            if slot.process is not None and slot.process.is_alive():
                continue
            self._start_worker(slot)

    def _start_worker(self, slot: WorkerSlot) -> None:
        """Start a worker process for a specific slot."""
        # Import here to avoid circular imports
        from .runner import worker_main

        slot.process = mp.Process(
            target=worker_main,
            args=(self.config, slot.worker_id),
            name=f"scoring-worker-{slot.worker_id}",
            daemon=False,  # Not daemon so it can clean up
        )
        slot.process.start()
        logger.info(
            f"Started scoring worker {slot.worker_id} (PID: {slot.process.pid})"
        )

    def _new_worker_id(self, index: int) -> str:
        """Generate a new unique worker ID."""
        return f"worker_{os.getpid()}_{index}_{int(time.time())}"

    def monitor(self) -> None:
        """Check worker health and restart if needed.

        Should be called periodically from the main validator loop.
        """
        if self._shutdown_requested:
            return
        for idx, slot in enumerate(self.workers):
            if slot.process is None:
                continue

            if slot.process.is_alive():
                continue

            exit_code = slot.process.exitcode
            logger.warning(f"Worker {slot.worker_id} died with exit code {exit_code}")

            if slot.restart_count < self.max_restart_attempts:
                slot.restart_count += 1
                logger.info(
                    "Restarting worker "
                    f"{slot.worker_id} (attempt {slot.restart_count}/{self.max_restart_attempts})"
                )
                time.sleep(self.restart_delay_sec)

                slot.worker_id = self._new_worker_id(idx)
                slot.process = None
                self._start_worker(slot)
            else:
                logger.error(
                    "Max restart attempts "
                    f"({self.max_restart_attempts}) reached for {slot.worker_id}. "
                    "Worker will not be restarted."
                )

    async def check_heartbeat(self) -> bool:
        """Check if worker heartbeat is fresh.

        Returns:
            True if heartbeat is fresh, False if stale or missing
        """
        try:
            worker_ids = [slot.worker_id for slot in self.workers if slot.process is not None]
            if not worker_ids:
                return False

            rows = await self.database.read(
                text(
                    """
                    SELECT worker_id, last_heartbeat
                    FROM scoring_worker_heartbeat
                    WHERE worker_id = ANY(:worker_ids)
                    """
                ),
                params={"worker_ids": worker_ids},
                mappings=True,
            )

            now = datetime.now(timezone.utc)
            heartbeat_by_id = {row["worker_id"]: row["last_heartbeat"] for row in rows}
            fresh_count = 0

            for slot in self.workers:
                if slot.worker_id not in heartbeat_by_id:
                    continue
                last_heartbeat = heartbeat_by_id[slot.worker_id]
                if last_heartbeat is None:
                    continue
                age = now - last_heartbeat
                if age.total_seconds() < self.heartbeat_timeout_sec:
                    fresh_count += 1
                    if slot.restart_count > 0:
                        slot.restart_count = 0
                        logger.info(
                            f"Worker heartbeat confirmed for {slot.worker_id}, reset restart counter"
                        )

            return fresh_count > 0

        except Exception as e:
            logger.warning(f"Failed to check heartbeat: {e}")
            return False

    def shutdown(self) -> None:
        """Gracefully stop the worker process."""
        self._shutdown_requested = True
        for slot in self.workers:
            if slot.process is None:
                continue

            if not slot.process.is_alive():
                logger.info(f"Worker {slot.worker_id} already stopped")
                continue

            logger.info(
                f"Requesting worker shutdown {slot.worker_id} (PID: {slot.process.pid})"
            )

            # Send SIGTERM for graceful shutdown
            try:
                slot.process.terminate()
            except Exception as e:
                logger.warning(f"Failed to terminate worker {slot.worker_id}: {e}")

            # Wait for graceful shutdown
            slot.process.join(timeout=30)

            if slot.process.is_alive():
                logger.warning(f"Worker {slot.worker_id} did not stop gracefully, killing...")
                try:
                    slot.process.kill()
                    slot.process.join(timeout=5)
                except Exception as e:
                    logger.error(f"Failed to kill worker {slot.worker_id}: {e}")

        logger.info("Worker shutdown complete")

    @property
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return any(slot.process is not None and slot.process.is_alive() for slot in self.workers)


__all__ = ["ScoringWorkerManager"]
