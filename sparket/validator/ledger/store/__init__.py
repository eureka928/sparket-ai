"""Pluggable transport layer for ledger checkpoints and deltas."""

from .interface import LedgerStore

__all__ = ["LedgerStore"]
