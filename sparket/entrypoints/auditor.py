"""Auditor validator entrypoint.

Lightweight process that fetches ledger data from the primary validator,
verifies scoring integrity, and sets weights on chain.

No SportsDataIO subscription, no database, no axon serving.
"""

import asyncio
import os
import signal
import sys

import bittensor as bt
from dotenv import load_dotenv


def main() -> None:
    # Load .env if not in test mode
    if os.environ.get("SPARKET_TEST_MODE") != "true":
        load_dotenv()

    bt.logging.info({"auditor": "starting"})

    # Parse args
    import argparse
    parser = argparse.ArgumentParser(description="Sparket Auditor Validator")
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    parser.add_argument("--netuid", type=int, default=57)
    parser.add_argument("--auditor.primary_hotkey", type=str, required=False)
    parser.add_argument("--auditor.primary_url", type=str, required=False)
    parser.add_argument("--auditor.poll_interval", type=int, default=900)
    parser.add_argument("--auditor.weight_tolerance", type=float, default=0.001)
    parser.add_argument("--auditor.data_dir", type=str, default="sparket/data/auditor")

    config = bt.config(parser)

    # Override from env vars
    primary_hotkey = os.environ.get(
        "SPARKET_AUDITOR__PRIMARY_HOTKEY",
        getattr(config, "auditor", {}).get("primary_hotkey", ""),
    )
    primary_url = os.environ.get(
        "SPARKET_AUDITOR__PRIMARY_URL",
        getattr(config, "auditor", {}).get("primary_url", ""),
    )
    poll_interval = int(os.environ.get(
        "SPARKET_AUDITOR__POLL_INTERVAL_SECONDS",
        getattr(config, "auditor", {}).get("poll_interval", 900),
    ))
    weight_tolerance = float(os.environ.get(
        "SPARKET_AUDITOR__WEIGHT_TOLERANCE",
        getattr(config, "auditor", {}).get("weight_tolerance", 0.001),
    ))
    data_dir = os.environ.get(
        "SPARKET_AUDITOR__DATA_DIR",
        getattr(config, "auditor", {}).get("data_dir", "sparket/data/auditor"),
    )

    if not primary_hotkey:
        bt.logging.error("SPARKET_AUDITOR__PRIMARY_HOTKEY is required")
        sys.exit(1)
    if not primary_url:
        bt.logging.error("SPARKET_AUDITOR__PRIMARY_URL is required")
        sys.exit(1)

    # Initialize bittensor components
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    netuid = config.netuid or 57
    metagraph = subtensor.metagraph(netuid=netuid)

    bt.logging.info({
        "auditor_config": {
            "primary_hotkey": primary_hotkey,
            "primary_url": primary_url,
            "poll_interval": poll_interval,
            "weight_tolerance": weight_tolerance,
            "hotkey": wallet.hotkey.ss58_address,
            "netuid": netuid,
        }
    })

    # Build auditor components
    from sparket.validator.ledger.store.http_client import HTTPLedgerStore
    from sparket.validator.auditor.sync import LedgerSync
    from sparket.validator.auditor.verifier import ManifestVerifier
    from sparket.validator.auditor.plugin_registry import PluginRegistry
    from sparket.validator.auditor.runtime import AuditorRuntime
    from sparket.validator.auditor.plugins.weight_verification import WeightVerificationHandler

    store = HTTPLedgerStore(primary_url=primary_url, wallet=wallet)
    sync = LedgerSync(store=store, data_dir=data_dir)
    verifier = ManifestVerifier(primary_hotkey=primary_hotkey)

    registry = PluginRegistry()
    registry.register(WeightVerificationHandler(tolerance=weight_tolerance))

    runtime = AuditorRuntime(
        wallet=wallet,
        subtensor=subtensor,
        metagraph=metagraph,
        sync=sync,
        verifier=verifier,
        registry=registry,
        config={
            "netuid": netuid,
            "auditor_poll_interval_seconds": poll_interval,
            "auditor_weight_tolerance": weight_tolerance,
        },
    )

    # Graceful shutdown
    loop = asyncio.new_event_loop()

    def _signal_handler(sig, frame):
        bt.logging.info({"auditor": "shutdown_signal_received"})
        runtime.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        loop.run_until_complete(runtime.run())
    except KeyboardInterrupt:
        bt.logging.info({"auditor": "keyboard_interrupt"})
    finally:
        loop.run_until_complete(store.close())
        loop.close()
        bt.logging.info({"auditor": "stopped"})


if __name__ == "__main__":
    main()
