"""Signed attestation helpers for auditor plugin results.

Attestations are chain-verifiable proofs that an auditor performed
specific verification work and reached a conclusion.
"""

from __future__ import annotations

from typing import Any

from sparket.validator.scoring.determinism import compute_hash

from .plugin_registry import TaskResult


def create_attestation(result: TaskResult, wallet: Any) -> str:
    """Create a signed attestation for a TaskResult.

    Signs: hash(plugin_name, status, evidence_hash)

    Args:
        result: The plugin result to attest.
        wallet: Bittensor wallet with hotkey.

    Returns:
        Hex-encoded signature.
    """
    payload = {
        "plugin_name": result.plugin_name,
        "plugin_version": result.plugin_version,
        "status": result.status,
        "evidence_hash": compute_hash(result.evidence),
        "completed_at": result.completed_at.isoformat() if result.completed_at else "",
    }
    payload_hash = compute_hash(payload)
    signature = wallet.hotkey.sign(payload_hash.encode())
    return signature.hex() if isinstance(signature, bytes) else str(signature)


def verify_attestation(result: TaskResult, hotkey_ss58: str) -> bool:
    """Verify a TaskResult attestation against a hotkey.

    Args:
        result: The attested TaskResult.
        hotkey_ss58: SS58 address of the expected signer.

    Returns:
        True if the attestation is valid.
    """
    import bittensor as bt

    if not result.attestation:
        return False

    payload = {
        "plugin_name": result.plugin_name,
        "plugin_version": result.plugin_version,
        "status": result.status,
        "evidence_hash": compute_hash(result.evidence),
        "completed_at": result.completed_at.isoformat() if result.completed_at else "",
    }
    payload_hash = compute_hash(payload)

    try:
        sig_bytes = bytes.fromhex(result.attestation)
        keypair = bt.Keypair(ss58_address=hotkey_ss58)
        return keypair.verify(payload_hash.encode(), sig_bytes)
    except Exception:
        return False


__all__ = ["create_attestation", "verify_attestation"]
