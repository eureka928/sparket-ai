"""SportsDataIO provider constants.

Avoid hardcoding provider IDs in feature modules. Prefer resolving IDs via
`sparket.reference.providers.get_provider_id(PROVIDER_CODE)`. The ID defined
here is a default for local development and tests.
"""

PROVIDER_CODE = "SDIO"
PROVIDER_ID = 1  # default; override with DB-seeded ID in production

__all__ = ["PROVIDER_CODE", "PROVIDER_ID"]


