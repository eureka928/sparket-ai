from __future__ import annotations

import hmac
import os
import secrets
import time
from typing import Any, Dict, Optional

import bittensor as bt


class ValidatorComms:
    """
    Manages validator-side announcement details and a rotating token for miner pushes.
    - Builds the advertised endpoint (proxy_url if set, else host/port from axon)
    - Maintains an HMAC token rotated every N steps
    - Verifies presented tokens from miners
    """

    def __init__(self, *, proxy_url: Optional[str], require_token: bool, step_rotation: int = 10) -> None:
        self.proxy_url = proxy_url
        self.require_token = require_token
        self.step_rotation = max(1, int(step_rotation))
        # Secret key for HMAC; generated at runtime. Can be overridden via env for testing.
        env_key = os.getenv("SPARKET_VALIDATOR_PUSH_SECRET")
        self._secret = env_key.encode("utf-8") if env_key else secrets.token_bytes(32)
        self._last_epoch_step: Optional[int] = None
        self._cached_token: Optional[str] = None

    def current_token(self, *, step: int) -> str:
        epoch = step // self.step_rotation
        # Cache per epoch
        if self._last_epoch_step != epoch or not self._cached_token:
            msg = str(epoch).encode("utf-8")
            self._cached_token = hmac.new(self._secret, msg, digestmod="sha256").hexdigest()
            self._last_epoch_step = epoch
        return self._cached_token

    def verify_token(self, *, token: Optional[str], step: int) -> bool:
        if not self.require_token:
            return True
        if not token:
            return False
        # Accept current or previous epoch to allow minor clock/step drift
        for epoch in (step // self.step_rotation, max(0, step // self.step_rotation - 1)):
            expect = hmac.new(self._secret, str(epoch).encode("utf-8"), digestmod="sha256").hexdigest()
            if hmac.compare_digest(expect, token):
                return True
        return False

    def advertised_endpoint(self, *, axon: Any) -> Dict[str, Any]:
        if self.proxy_url:
            return {"url": self.proxy_url}
        # Prefer external_ip/external_port for broadcasting (what miners can reach)
        host = getattr(axon, "external_ip", None) or getattr(axon, "ip", None) or "127.0.0.1"
        port = getattr(axon, "external_port", None) or getattr(axon, "port", None) or 0
        return {"host": host, "port": port}


