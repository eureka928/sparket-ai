"""Manifest verification for checkpoint and delta windows.

Checks signature, content hashes, schema version, and window alignment
before an auditor trusts ledger data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sparket.validator.ledger.models import (
    LEDGER_SCHEMA_VERSION,
    CheckpointWindow,
    DeltaWindow,
)
from sparket.validator.ledger.signer import compute_section_hash, verify_manifest


@dataclass
class VerificationResult:
    """Outcome of manifest verification."""

    valid: bool
    errors: list[str]

    def __bool__(self) -> bool:
        return self.valid


class ManifestVerifier:
    """Verifies ledger window integrity before use."""

    def __init__(self, primary_hotkey: str):
        self.primary_hotkey = primary_hotkey

    def verify_checkpoint(self, cp: CheckpointWindow) -> VerificationResult:
        """Verify a checkpoint's manifest, signature, and content hashes."""
        return self._verify(cp, sections={
            "roster": cp.roster,
            "accumulators": cp.accumulators,
            "scoring_config": cp.scoring_config,
        })

    def verify_delta(self, delta: DeltaWindow) -> VerificationResult:
        """Verify a delta's manifest, signature, and content hashes."""
        return self._verify(delta, sections={
            "settled_submissions": delta.settled_submissions,
            "settled_outcomes": delta.settled_outcomes,
        })

    def _verify(
        self,
        window: CheckpointWindow | DeltaWindow,
        sections: dict,
    ) -> VerificationResult:
        errors: list[str] = []
        manifest = window.manifest

        # Schema version
        if manifest.schema_version != LEDGER_SCHEMA_VERSION:
            errors.append(
                f"schema_version mismatch: got {manifest.schema_version}, "
                f"expected {LEDGER_SCHEMA_VERSION}"
            )

        # Primary hotkey
        if manifest.primary_hotkey != self.primary_hotkey:
            errors.append(
                f"primary_hotkey mismatch: got {manifest.primary_hotkey}, "
                f"expected {self.primary_hotkey}"
            )

        # Signature
        if not verify_manifest(manifest, self.primary_hotkey):
            errors.append("signature verification failed")

        # Content hashes
        for section_name, section_data in sections.items():
            expected = manifest.content_hashes.get(section_name)
            if expected is None:
                errors.append(f"missing content hash for section: {section_name}")
                continue

            actual = compute_section_hash(section_data)
            if actual != expected:
                errors.append(
                    f"content hash mismatch for {section_name}: "
                    f"expected {expected[:16]}..., got {actual[:16]}..."
                )

        # Window type
        expected_type = "checkpoint" if isinstance(window, CheckpointWindow) else "delta"
        if manifest.window_type != expected_type:
            errors.append(
                f"window_type mismatch: got {manifest.window_type}, expected {expected_type}"
            )

        return VerificationResult(valid=len(errors) == 0, errors=errors)


__all__ = ["ManifestVerifier", "VerificationResult"]
