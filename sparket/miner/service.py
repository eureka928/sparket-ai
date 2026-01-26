from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from typing import Any, Optional

import bittensor as bt

from sparket.miner.client import ValidatorClient
from sparket.miner.utils.payloads import (
    build_submit_odds_payload,
    build_submit_outcome_payload,
)
from sparket.miner.utils.ratelimit import TokenBucket


class MinerService:
    def __init__(self, *, miner: Any) -> None:
        self.miner = miner
        self.config = miner.app_config.miner
        self.running = False
        self._tasks: list[asyncio.Task] = []
        self._client = ValidatorClient(
            wallet=miner.wallet,
            metagraph=miner.metagraph,
            get_validator_endpoint=lambda: getattr(miner, "validator_endpoint", None),
        )
        self._global_bucket = TokenBucket(self.config.rate.global_per_minute)
        self._per_market_buckets: dict[int, TokenBucket] = {}

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        loop = asyncio.new_event_loop()
        self._loop = loop

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run())

        import threading

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        try:
            for t in self._tasks:
                t.cancel()
        except Exception:
            pass

    async def _run(self) -> None:
        try:
            self._tasks = [
                asyncio.create_task(self._run_odds_pipeline()),
                asyncio.create_task(self._run_outcomes_pipeline()),
            ]
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    def _token(self) -> Optional[str]:
        endpoint = getattr(self.miner, "validator_endpoint", None) or {}
        return endpoint.get("token") if isinstance(endpoint, dict) else None

    async def _run_odds_pipeline(self) -> None:
        interval = max(1, int(self.config.cadence.odds_seconds))
        kinds = ["moneyline"]
        markets = self.config.markets or []
        miner_id = int(self.config.id)
        while self.running:
            started_ms = int(time.time() * 1000)
            for idx, market_id in enumerate(markets):
                if not self._global_bucket.allow():
                    continue
                bucket = self._per_market_buckets.setdefault(
                    int(market_id), TokenBucket(self.config.rate.per_market_per_minute)
                )
                if not bucket.allow():
                    continue
                try:
                    now = datetime.now(timezone.utc)
                    payload = build_submit_odds_payload(
                        miner_id=miner_id,
                        miner_hotkey=self.miner.wallet.hotkey.ss58_address,
                        market_id=int(market_id),
                        kind=kinds[0],
                        token=self._token(),
                        now=now,
                    )
                    ok = await self._attempt_with_retries(
                        lambda: self._client.submit_odds(payload)
                    )
                    bt.logging.info(
                        {
                            "component": "miner_service",
                            "op": "submit_odds",
                            "status": "ok" if ok else "error",
                            "market_id": int(market_id),
                            "attempts": self.config.retry.max_attempts,
                            "dedup_bucket_seconds": self.config.idempotency.bucket_seconds,
                        }
                    )
                except Exception as e:
                    bt.logging.warning({"miner_submit_odds_error": str(e)})
            elapsed = int(time.time() * 1000) - started_ms
            await asyncio.sleep(max(0.0, interval - (elapsed / 1000.0)))

    async def _run_outcomes_pipeline(self) -> None:
        interval = max(1, int(self.config.cadence.outcomes_seconds))
        events = self.config.events or []
        while self.running:
            started_ms = int(time.time() * 1000)
            for event_id in events:
                if not self._global_bucket.allow():
                    continue
                try:
                    now = datetime.now(timezone.utc)
                    payload = build_submit_outcome_payload(
                        event_id=event_id,
                        miner_hotkey=self.miner.wallet.hotkey.ss58_address,
                        token=self._token(),
                        now=now,
                    )
                    ok = await self._attempt_with_retries(
                        lambda: self._client.submit_outcome(payload)
                    )
                    bt.logging.info(
                        {
                            "component": "miner_service",
                            "op": "submit_outcome",
                            "status": "ok" if ok else "error",
                            "event_id": event_id,
                            "attempts": self.config.retry.max_attempts,
                            "dedup_bucket_seconds": self.config.idempotency.bucket_seconds,
                        }
                    )
                except Exception as e:
                    bt.logging.warning({"miner_submit_outcome_error": str(e)})
            elapsed = int(time.time() * 1000) - started_ms
            await asyncio.sleep(max(0.0, interval - (elapsed / 1000.0)))

    async def _attempt_with_retries(self, fn):
        attempts = max(1, int(self.config.retry.max_attempts))
        backoff = float(self.config.retry.initial_backoff_ms) / 1000.0
        max_backoff = float(self.config.retry.max_backoff_ms) / 1000.0
        for i in range(attempts):
            ok = await fn()
            if ok:
                return True
            if i == attempts - 1:
                break
            jitter = backoff * (0.9 + 0.2 * random.random())
            await asyncio.sleep(min(max_backoff, jitter))
            backoff = min(max_backoff, backoff * 2.0)
        return False


