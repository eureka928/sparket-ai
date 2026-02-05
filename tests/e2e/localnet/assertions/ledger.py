"""Reusable assertion functions for ledger E2E tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from sparket.validator.ledger.models import CheckpointWindow, DeltaWindow
from sparket.validator.ledger.redaction import TIER3_FIELD_PATTERNS, contains_tier3
from sparket.validator.ledger.signer import compute_section_hash, verify_manifest


@dataclass
class LedgerAssertionResult:
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.failed == 0


def assert_checkpoint_valid(cp: CheckpointWindow) -> LedgerAssertionResult:
    """Structural validation of a CheckpointWindow."""
    result = LedgerAssertionResult()

    if cp.manifest.window_type != "checkpoint":
        result.failed += 1
        result.details.append(f"wrong window_type: {cp.manifest.window_type}")
    else:
        result.passed += 1

    if not cp.manifest.primary_hotkey:
        result.failed += 1
        result.details.append("missing primary_hotkey")
    else:
        result.passed += 1

    if cp.manifest.checkpoint_epoch < 1:
        result.failed += 1
        result.details.append(f"invalid epoch: {cp.manifest.checkpoint_epoch}")
    else:
        result.passed += 1

    return result


def assert_delta_valid(delta: DeltaWindow) -> LedgerAssertionResult:
    """Structural validation of a DeltaWindow."""
    result = LedgerAssertionResult()

    if delta.manifest.window_type != "delta":
        result.failed += 1
        result.details.append(f"wrong window_type: {delta.manifest.window_type}")
    else:
        result.passed += 1

    return result


def assert_manifest_signature_valid(
    window: CheckpointWindow | DeltaWindow, primary_hotkey: str,
) -> bool:
    """Verify the manifest signature."""
    return verify_manifest(window.manifest, primary_hotkey)


def assert_no_tier3_data(window: CheckpointWindow | DeltaWindow) -> bool:
    """Exhaustive scan for Tier 3 fields in exported data."""
    data = window.model_dump(mode="json")
    return not _scan_for_tier3(data)


def _scan_for_tier3(obj, path="") -> bool:
    """Recursively scan for Tier 3 field names."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in TIER3_FIELD_PATTERNS:
                return True
            if _scan_for_tier3(value, f"{path}.{key}"):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if _scan_for_tier3(item, path):
                return True
    return False


def assert_content_hashes_match(window: CheckpointWindow | DeltaWindow) -> bool:
    """Recompute content hashes and compare to manifest."""
    if isinstance(window, CheckpointWindow):
        sections = {
            "roster": window.roster,
            "accumulators": window.accumulators,
            "scoring_config": window.scoring_config,
        }
    else:
        sections = {
            "settled_submissions": window.settled_submissions,
            "settled_outcomes": window.settled_outcomes,
        }

    for name, data in sections.items():
        expected = window.manifest.content_hashes.get(name)
        if expected is None:
            return False
        actual = compute_section_hash(data)
        if actual != expected:
            return False

    return True


def assert_brier_scores_independently_correct(delta: DeltaWindow) -> tuple[int, int]:
    """Recompute each Brier score from (imp_prob, outcome).

    Returns (checks, mismatches).
    """
    checks = 0
    mismatches = 0

    for sub in delta.settled_submissions:
        if sub.brier is None:
            continue
        outcome = next(
            (o for o in delta.settled_outcomes if o.market_id == sub.market_id),
            None,
        )
        if outcome is None or outcome.result is None:
            continue

        actual = 1.0 if sub.side == outcome.result else 0.0
        expected_brier = (sub.imp_prob - actual) ** 2
        checks += 1
        if abs(expected_brier - sub.brier) > 1e-6:
            mismatches += 1

    return checks, mismatches


def assert_delta_only_settled(delta: DeltaWindow) -> bool:
    """Verify delta contains no unsettled market data."""
    for sub in delta.settled_submissions:
        if sub.settled_at is None:
            return False
    return True


def run_all_ledger_assertions(
    window: CheckpointWindow | DeltaWindow,
    primary_hotkey: str,
) -> LedgerAssertionResult:
    """Run all ledger assertions and return a combined result."""
    result = LedgerAssertionResult()

    # Structural
    if isinstance(window, CheckpointWindow):
        struct = assert_checkpoint_valid(window)
    else:
        struct = assert_delta_valid(window)
    result.passed += struct.passed
    result.failed += struct.failed
    result.details.extend(struct.details)

    # Signature
    if assert_manifest_signature_valid(window, primary_hotkey):
        result.passed += 1
    else:
        result.failed += 1
        result.details.append("signature verification failed")

    # No Tier 3 data
    if assert_no_tier3_data(window):
        result.passed += 1
    else:
        result.failed += 1
        result.details.append("Tier 3 data found in export")

    # Content hashes
    if assert_content_hashes_match(window):
        result.passed += 1
    else:
        result.failed += 1
        result.details.append("content hash mismatch")

    # Brier verification (delta only)
    if isinstance(window, DeltaWindow):
        checks, mismatches = assert_brier_scores_independently_correct(window)
        if mismatches == 0:
            result.passed += 1
        else:
            result.failed += 1
            result.details.append(f"Brier mismatches: {mismatches}/{checks}")

        if assert_delta_only_settled(window):
            result.passed += 1
        else:
            result.failed += 1
            result.details.append("unsettled submissions in delta")

    return result


__all__ = [
    "LedgerAssertionResult",
    "assert_brier_scores_independently_correct",
    "assert_checkpoint_valid",
    "assert_content_hashes_match",
    "assert_delta_only_settled",
    "assert_delta_valid",
    "assert_manifest_signature_valid",
    "assert_no_tier3_data",
    "run_all_ledger_assertions",
]
