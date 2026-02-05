"""Manifest signing and verification using bittensor keypairs.

The primary signs each manifest with its hotkey. Auditors verify the
signature against the known primary hotkey before trusting the data.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sparket.validator.scoring.determinism import compute_hash

if TYPE_CHECKING:
    from .models import LedgerManifest


def _manifest_signing_payload(manifest: LedgerManifest) -> str:
    """Build the canonical string to sign.

    Excludes the signature field itself to avoid circular dependency.
    """
    data = manifest.model_dump(mode="json")
    data.pop("signature", None)
    return compute_hash(data)


def sign_manifest(manifest: LedgerManifest, wallet: Any) -> str:
    """Sign a manifest with the validator's hotkey.

    Args:
        manifest: The manifest to sign (signature field will be ignored).
        wallet: Bittensor wallet with hotkey access.

    Returns:
        Hex-encoded signature string.
    """
    payload_hash = _manifest_signing_payload(manifest)
    signature = wallet.hotkey.sign(payload_hash.encode())
    return signature.hex() if isinstance(signature, bytes) else str(signature)


def verify_manifest(manifest: LedgerManifest, hotkey_ss58: str) -> bool:
    """Verify a manifest signature against a hotkey.

    Args:
        manifest: The manifest to verify (must have signature set).
        hotkey_ss58: SS58 address of the expected signer.

    Returns:
        True if signature is valid for the given hotkey.
    """
    import bittensor as bt

    if not manifest.signature:
        return False

    payload_hash = _manifest_signing_payload(manifest)
    try:
        sig_bytes = bytes.fromhex(manifest.signature)
    except ValueError:
        return False

    try:
        keypair = bt.Keypair(ss58_address=hotkey_ss58)
        return keypair.verify(payload_hash.encode(), sig_bytes)
    except Exception:
        return False


def compute_section_hash(data: Any) -> str:
    """Compute SHA256 hash of a data section for manifest content_hashes.

    Works with Pydantic models, dicts, and lists.
    """
    if hasattr(data, "model_dump"):
        as_dict = data.model_dump(mode="json")
    elif isinstance(data, list):
        as_dict = {"items": [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in data
        ]}
    elif isinstance(data, dict):
        as_dict = data
    else:
        as_dict = {"value": data}

    return compute_hash(as_dict)


__all__ = [
    "compute_section_hash",
    "sign_manifest",
    "verify_manifest",
]
