"""Auditor validator runtime.

Lightweight validator process that fetches scoring ledger data from the
primary, independently verifies outcome-based scores, recomputes weights
deterministically, and sets them on chain.

No SportsDataIO subscription, no database, no full scoring pipeline.
"""
