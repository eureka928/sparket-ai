"""
Database bootstrap for the Sparket miner.

This package is responsible for Alembic migrations and runtime helpers
used by the lightweight SQLite store that tracks validator endpoints
and any other miner-specific state.
"""
from .init import initialize
from .dbm import DBM

__all__ = ["initialize", "DBM"]

