"""Allowlist-based field redaction for license-safe ledger exports.

Every exported field must be explicitly listed. Unknown fields are dropped.
This is the hard boundary between Tier 2 (validator-gated) and Tier 3
(primary-only) data.

Tier 1 - Public: on-chain data, settled outcomes, scoring config
Tier 2 - Validator-Gated: accumulator state, settled submission scores, roster
Tier 3 - Primary-Only: SDIO data, closing lines, unsettled submissions, CLV/CLE
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class DataTier(str, Enum):
    """Data sensitivity classification."""

    PUBLIC = "public"
    VALIDATOR_GATED = "validator_gated"
    PRIMARY_ONLY = "primary_only"


# ---------------------------------------------------------------------------
# Allowlists (Tier 2 - safe to export)
# ---------------------------------------------------------------------------

SAFE_ACCUMULATOR_FIELDS: frozenset[str] = frozenset({
    "miner_id",
    "hotkey",
    "uid",
    "n_submissions",
    "n_outcomes",
    # Accumulator pairs
    "brier",
    "fq",
    "pss",
    "es",
    "mes",
    "sos",
    "lead",
    # Derived means
    "brier_mean",
    "fq_raw",
    "pss_mean",
    "es_adj",
    "mes_mean",
    "sos_score",
    "lead_score",
    "cal_score",
    "sharp_score",
})

SAFE_ROLLING_SCORE_FIELDS: frozenset[str] = frozenset({
    "miner_id",
    "miner_hotkey",
    "uid",
    "n_submissions",
    "n_eff",
    "es_mean",
    "es_std",
    "es_adj",
    "mes_mean",
    "sos_mean",
    "pss_mean",
    "fq_raw",
    "brier_mean",
    "lead_ratio",
    "fq_score",
    "cal_score",
    "sharp_score",
    "edge_score",
    "mes_score",
    "sos_score",
    "lead_score",
    "forecast_dim",
    "econ_dim",
    "info_dim",
    "skill_score",
    "score_version",
    "as_of",
    "window_days",
    # Accumulator pairs (new columns)
    "brier_ws",
    "brier_wt",
    "fq_ws",
    "fq_wt",
    "pss_ws",
    "pss_wt",
    "es_ws",
    "es_wt",
    "mes_ws",
    "mes_wt",
    "sos_ws",
    "sos_wt",
    "lead_ws",
    "lead_wt",
})

SAFE_OUTCOME_FIELDS: frozenset[str] = frozenset({
    "market_id",
    "event_id",
    "result",
    "score_home",
    "score_away",
    "settled_at",
})

SAFE_MINER_FIELDS: frozenset[str] = frozenset({
    "miner_id",
    "uid",
    "hotkey",
    "active",
})

SAFE_SETTLED_SUBMISSION_FIELDS: frozenset[str] = frozenset({
    "miner_id",
    "market_id",
    "side",
    "imp_prob",
    "brier",
    "pss",
    "settled_at",
})

# ---------------------------------------------------------------------------
# Tier 3 fields (NEVER export - explicit denylist for safety checks)
# ---------------------------------------------------------------------------

TIER3_FIELD_PATTERNS: frozenset[str] = frozenset({
    # SportsDataIO raw data
    "provider_quote",
    "odds_eu_close",
    "imp_prob_close",
    "imp_prob_norm_close",
    "ts_close",
    "raw",
    # Ground truth
    "ground_truth_snapshot",
    "ground_truth_closing",
    "sportsbook_bias",
    "close_odds_eu",
    "close_imp_prob",
    "close_imp_prob_norm",
    # Per-submission CLV/CLE
    "clv_odds",
    "clv_prob",
    "cle",
    "minutes_to_close",
    "snapshot_prob",
    "snapshot_odds",
    # External references
    "ext_ref",
    # Unsettled submission fields
    "odds_eu",
    "priced_at",
    "payload",
    "submitted_at",
})


def redact(row: dict[str, Any], allowlist: frozenset[str]) -> dict[str, Any]:
    """Filter a row to only include allowlisted fields.

    Args:
        row: Source data dictionary.
        allowlist: Set of field names permitted in the output.

    Returns:
        New dictionary containing only allowlisted keys with non-None values.
    """
    return {k: v for k, v in row.items() if k in allowlist and v is not None}


def contains_tier3(row: dict[str, Any]) -> bool:
    """Check whether a row contains any Tier 3 (primary-only) fields.

    Used as a safety check in tests and export validation.
    """
    return bool(set(row.keys()) & TIER3_FIELD_PATTERNS)


def classify_field(field_name: str) -> DataTier:
    """Classify a field name into its data tier."""
    if field_name in TIER3_FIELD_PATTERNS:
        return DataTier.PRIMARY_ONLY
    # Check all Tier 2 allowlists
    tier2_fields = (
        SAFE_ROLLING_SCORE_FIELDS
        | SAFE_OUTCOME_FIELDS
        | SAFE_MINER_FIELDS
        | SAFE_SETTLED_SUBMISSION_FIELDS
        | SAFE_ACCUMULATOR_FIELDS
    )
    if field_name in tier2_fields:
        return DataTier.VALIDATOR_GATED
    return DataTier.PUBLIC


__all__ = [
    "DataTier",
    "SAFE_ACCUMULATOR_FIELDS",
    "SAFE_MINER_FIELDS",
    "SAFE_OUTCOME_FIELDS",
    "SAFE_ROLLING_SCORE_FIELDS",
    "SAFE_SETTLED_SUBMISSION_FIELDS",
    "TIER3_FIELD_PATTERNS",
    "classify_field",
    "contains_tier3",
    "redact",
]
