# Custom Miner Service Example

This example shows how to replace the base miner with your own service
while reusing the validator client and payload formats.

## Step 1: disable the base miner
```
export SPARKET_BASE_MINER__ENABLED=false
```

## Step 2: create a custom service
Create `sparket/miner/custom_service.py`:
```python
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sparket.miner.client import ValidatorClient


class CustomMinerService:
    def __init__(self, miner) -> None:
        self.miner = miner
        self._client = ValidatorClient(
            wallet=miner.wallet,
            metagraph=miner.metagraph,
            get_validator_endpoint=lambda: getattr(miner, "validator_endpoint", None),
        )
        self.running = False

    def _token(self) -> str | None:
        endpoint = getattr(self.miner, "validator_endpoint", None) or {}
        return endpoint.get("token") if isinstance(endpoint, dict) else None

    async def submit_odds(self, market_id: int, probs: dict[str, float]) -> None:
        now = datetime.now(timezone.utc)
        payload = {
            "miner_id": int(self.miner.app_config.miner.id),
            "miner_hotkey": self.miner.wallet.hotkey.ss58_address,
            "submissions": [
                {
                    "market_id": int(market_id),
                    "kind": "MONEYLINE",
                    "priced_at": now,
                    "prices": [
                        {
                            "side": "home",
                            "odds_eu": round(1.0 / probs["home"], 2),
                            "imp_prob": probs["home"],
                        },
                        {
                            "side": "away",
                            "odds_eu": round(1.0 / probs["away"], 2),
                            "imp_prob": probs["away"],
                        },
                    ],
                }
            ],
            "token": self._token(),
        }
        await self._client.submit_odds(payload)

    async def start(self) -> None:
        self.running = True
        while self.running:
            for market_id in self.miner.app_config.miner.markets:
                # Replace this with your model output
                probs = {"home": 0.52, "away": 0.48}
                await self.submit_odds(market_id, probs)
            await asyncio.sleep(60)
```

## Step 3: start the service
In `sparket/entrypoints/miner.py`, after creating the miner instance:
```python
from sparket.miner.custom_service import CustomMinerService

custom_service = CustomMinerService(miner)
asyncio.run(custom_service.start())
```

## Notes
- Use `sparket/miner/utils/payloads.py` as a reference for payload structure.
- For outcomes, send a payload via `ValidatorClient.submit_outcome`.
- Add rate limiting and retries similar to `sparket/miner/service.py`.
