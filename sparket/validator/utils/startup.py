from __future__ import annotations

from typing import Any

import importlib
import bittensor as bt
from sqlalchemy import text


def check_python_requirements() -> None:
    """Verify critical runtime dependencies are importable.

    Keep lightweight: just import and optionally log versions; do not raise unless missing.
    """
    required = [
        ("sqlalchemy", None),
        ("asyncpg", None),
        ("bittensor", None),
    ]
    for mod, _ in required:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", None)
            bt.logging.info({"requirement_ok": {"module": mod, "version": ver or "unknown"}})
        except Exception as e:
            bt.logging.warning({"requirement_missing": {"module": mod, "error": str(e)}})


async def ping_database(dbm: Any) -> bool:
    try:
        rows = await dbm.read(text("select 1"))
        bt.logging.info({"database_ping": "ok"})
        return True
    except Exception as e:
        bt.logging.error({"database_ping_error": str(e)})
        return False


def summarize_bittensor_state(neuron: Any) -> None:
    try:
        wallet = getattr(neuron, "wallet", None)
        hotkey = getattr(wallet, "hotkey", None)
        coldkeypub = getattr(wallet, "coldkeypub", None)
        config_obj = getattr(neuron, "config", None)
        config_subtensor = getattr(config_obj, "subtensor", None) if config_obj is not None else None
        config_axon = getattr(config_obj, "axon", None) if config_obj is not None else None
        runtime_subtensor = getattr(neuron, "subtensor", None)
        metagraph = getattr(neuron, "metagraph", None)
        def _is_loopback(addr: str | None) -> bool:
            if not addr:
                return False
            lower = addr.lower()
            return "127.0.0.1" in lower or "localhost" in lower
        summary = {
            "hotkey": getattr(hotkey, "ss58_address", None),
            "coldkeypub": getattr(coldkeypub, "ss58_address", None),
            "netuid": getattr(config_obj, "netuid", None),
            "config_endpoint": getattr(config_subtensor, "chain_endpoint", None) if config_subtensor else None,
            "config_network": getattr(config_subtensor, "network", None) if config_subtensor else None,
            "config_axon": {
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
                sample: list[dict[str, object]] = []
                for uid, axon in enumerate(axons):
                    sample.append({
                        "uid": uid,
                        "hotkey": getattr(axon, "hotkey", None),
                        "ip": getattr(axon, "ip", None),
                        "port": getattr(axon, "port", None),
                        "loopback": _is_loopback(getattr(axon, "ip", None)),
                    })
                    if len(sample) >= 10:
                        break
                summary["metagraph_axons_sample"] = sample
        except Exception:
            pass
        axon = getattr(neuron, "axon", None)
        if axon is not None:
            summary["axon"] = {
                "ip": getattr(axon, "ip", None),
                "port": getattr(axon, "port", None),
                "external_ip": getattr(axon, "external_ip", None),
                "external_port": getattr(axon, "external_port", None),
            }
        dendrite = getattr(neuron, "dendrite", None)
        if dendrite is not None:
            summary["dendrite"] = {
                "object": repr(dendrite),
                "wallet_hotkey": getattr(hotkey, "ss58_address", None),
            }
        bt.logging.info({"bt_state": summary})
    except Exception:
        pass


