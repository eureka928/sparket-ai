"""Utility helpers shared across the database initialization package."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from urllib.parse import urlsplit, urlunsplit

__all__ = [
    "_sanitize_url_for_log",
    "_command_exists",
    "_run",
    "_is_root",
]


def _sanitize_url_for_log(url: str) -> str:
    """Sanitize URL for logging by redacting the password."""
    try:
        parts = urlsplit(url)
        if parts.password:
            netloc = f"{parts.username}:***@{parts.hostname}"
            if parts.port:
                netloc += f":{parts.port}"
        else:
            netloc = parts.netloc
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return "***"


def _command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(cmd: str) -> int:
    return subprocess.call(shlex.split(cmd))


def _is_root() -> bool:
    try:
        return os.geteuid() == 0
    except AttributeError:
        return False


