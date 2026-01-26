"""Provider registry used by application code.

This module centralizes provider code-to-ID mappings so that feature modules
like `sparket.providers.sportsdataio` don't hardcode database IDs. Seed these
rows in your reference DB at boot and keep the mapping in sync.
"""

from __future__ import annotations

from typing import Dict, Optional


# Canonical provider records. In production, prefer seeding your DB and
# reading provider IDs dynamically. This mapping is a stable fallback.
_PROVIDERS: Dict[str, int] = {
    "SDIO": 1,  # SportsDataIO
}


def get_provider_id(code: str) -> Optional[int]:
    """Return provider_id for a provider code, if known."""
    return _PROVIDERS.get(code)


__all__ = ["get_provider_id"]


