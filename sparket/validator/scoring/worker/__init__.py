"""Scoring worker process module.

This module manages the dedicated worker process(es) for batch scoring jobs.
The worker runs independently from the main validator process to avoid
blocking API/synapse handling.
"""

from __future__ import annotations

__all__: list[str] = []
