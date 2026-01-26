from __future__ import annotations

import os
from typing import Any


def build_database_url(
    *,
    user: str,
    password: str | None,
    host: str,
    port: str,
    name: str,
) -> str:
    auth = f"{user}:{password}" if password else f"{user}"
    return f"postgresql+asyncpg://{auth}@{host}:{port}/{name}"


def ensure_env_database_url() -> dict[str, Any]:
    """Ensure database URL is set in environment variables."""
    def _env2(k1: str, k2: str | None = None) -> str | None:
        v = os.getenv(k1)
        if v is None and k2 is not None:
            v = os.getenv(k2)
        return v

    current_port = _env2("SPARKET_DATABASE__PORT", "SPARKET_DATABASE_PORT") or "5432"
    existing_url = os.getenv("SPARKET_DATABASE__URL") or os.getenv("DATABASE_URL")
    url_port = None
    if existing_url:
        try:
            from urllib.parse import urlsplit
            parts = urlsplit(existing_url)
            url_port = str(parts.port) if parts.port else None
        except Exception:
            url_port = None

    if not existing_url or (url_port and url_port != current_port):
        user = _env2("SPARKET_DATABASE__USER", "SPARKET_DATABASE_USER")
        pwd = _env2("SPARKET_DATABASE__PASSWORD", "SPARKET_DATABASE_PASSWORD") or ""
        host = _env2("SPARKET_DATABASE__HOST", "SPARKET_DATABASE_HOST") or "127.0.0.1"
        port = current_port
        name = _env2("SPARKET_DATABASE__NAME", "SPARKET_DATABASE_NAME")
        if user and name:
            url = build_database_url(
                user=user,
                password=pwd,
                host=host,
                port=port,
                name=name,
            )
            os.environ["SPARKET_DATABASE__URL"] = url
            os.environ["DATABASE_URL"] = url
            return {
                "composed": True,
                "reason": "missing_url" if not existing_url else "port_changed",
                "port": port,
                "old_port": url_port,
            }

    return {"composed": False, "url_already_set": bool(existing_url)}


def ensure_config_database_url(core_db: Any) -> dict[str, Any]:
    """Ensure database URL is set on config object."""
    if core_db is None:
        return {"composed": False, "reason": "missing_config"}

    if getattr(core_db, "url", None):
        return {"composed": False, "url_already_set": True}

    host = getattr(core_db, "host", None) or "127.0.0.1"
    port = str(getattr(core_db, "port", None) or 5432)
    user = getattr(core_db, "user", None)
    pwd = getattr(core_db, "password", None) or ""
    name = getattr(core_db, "name", None)
    if user and name:
        url = build_database_url(
            user=user,
            password=pwd,
            host=host,
            port=port,
            name=name,
        )
        setattr(core_db, "url", url)
        return {"composed": True}

    return {"composed": False, "reason": "missing_fields"}


__all__ = [
    "build_database_url",
    "ensure_env_database_url",
    "ensure_config_database_url",
]
