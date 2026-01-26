from __future__ import annotations

import importlib
from typing import Any

import bittensor as bt
from sqlalchemy import text


def check_python_requirements() -> None:
    """Log availability and versions of critical runtime dependencies."""
    requirements = [
        ("sqlalchemy", None),
        ("aiosqlite", None),
        ("bittensor", None),
    ]
    for module_name, _ in requirements:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None) or "unknown"
            bt.logging.info(
                {
                    "miner_startup": {
                        "step": "dependency_check",
                        "module": module_name,
                        "status": "ok",
                        "version": version,
                    }
                }
            )
        except Exception as exc:
            bt.logging.warning(
                {
                    "miner_startup": {
                        "step": "dependency_check",
                        "module": module_name,
                        "status": "error",
                        "error": str(exc),
                    }
                }
            )


async def ping_database(dbm: Any) -> bool:
    """Execute a simple read to confirm the miner database is reachable."""
    try:
        await dbm.read(text("select 1"))
        bt.logging.info({"miner_startup": {"step": "database_ping", "status": "ok"}})
        return True
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        bt.logging.error(
            {
                "miner_startup": {
                    "step": "database_ping",
                    "status": "error",
                    "error": str(exc),
                }
            }
        )
        return False


def summarize_miner_state(miner: Any) -> None:
    """Log high-level information about the miner's bittensor runtime state."""
    try:
        wallet = getattr(miner, "wallet", None)
        hotkey = getattr(wallet, "hotkey", None)
        coldkeypub = getattr(wallet, "coldkeypub", None)
        config_obj = getattr(miner, "config", None)
        config_subtensor = getattr(config_obj, "subtensor", None) if config_obj else None
        config_axon = getattr(config_obj, "axon", None) if config_obj else None
        runtime_subtensor = getattr(miner, "subtensor", None)
        metagraph = getattr(miner, "metagraph", None)

        def _is_loopback(addr: str | None) -> bool:
            if not addr:
                return False
            lower = addr.lower()
            return "127.0.0.1" in lower or "localhost" in lower

        summary: dict[str, Any] = {
            "hotkey": getattr(hotkey, "ss58_address", None),
            "coldkeypub": getattr(coldkeypub, "ss58_address", None),
            "netuid": getattr(config_obj, "netuid", None),
            "config_endpoint": getattr(config_subtensor, "chain_endpoint", None) if config_subtensor else None,
            "config_network": getattr(config_subtensor, "network", None) if config_subtensor else None,
            "axon_config": {
                "ip": getattr(config_axon, "ip", None) if config_axon else None,
                "port": getattr(config_axon, "port", None) if config_axon else None,
                "external_ip": getattr(config_axon, "external_ip", None) if config_axon else None,
                "external_port": getattr(config_axon, "external_port", None) if config_axon else None,
            },
            "runtime_endpoint": getattr(runtime_subtensor, "chain_endpoint", None) if runtime_subtensor else None,
            "runtime_network": getattr(runtime_subtensor, "network", None) if runtime_subtensor else None,
            "axon_off": getattr(getattr(config_obj, "neuron", None), "axon_off", False) if config_obj else False,
            "metagraph_n": getattr(metagraph, "n", None),
        }

        try:
            axons = list(getattr(metagraph, "axons", []) or [])
            summary["metagraph_axons_counts"] = {
                "total": len(axons),
                "external": sum(0 if _is_loopback(getattr(axon, "ip", None)) else 1 for axon in axons),
            }
            if axons:
                sample: list[dict[str, Any]] = []
                for uid, axon in enumerate(axons):
                    sample.append(
                        {
                            "uid": uid,
                            "hotkey": getattr(axon, "hotkey", None),
                            "ip": getattr(axon, "ip", None),
                            "port": getattr(axon, "port", None),
                            "loopback": _is_loopback(getattr(axon, "ip", None)),
                        }
                    )
                    if len(sample) >= 10:
                        break
                summary["metagraph_axons_sample"] = sample
        except Exception:  # pragma: no cover - diagnostics only
            pass

        axon = getattr(miner, "axon", None)
        if axon is not None:
            summary["axon"] = {
                "ip": getattr(axon, "ip", None),
                "port": getattr(axon, "port", None),
                "external_ip": getattr(axon, "external_ip", None),
                "external_port": getattr(axon, "external_port", None),
            }

        dendrite = getattr(miner, "dendrite", None)
        if dendrite is not None:
            summary["dendrite"] = {
                "object": repr(dendrite),
                "wallet_hotkey": getattr(hotkey, "ss58_address", None),
            }

        bt.logging.info({"miner_state": summary})
    except Exception:  # pragma: no cover - diagnostics only
        pass

