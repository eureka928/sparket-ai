"""Bootstrap an initial ledger checkpoint for e2e testing.

One-shot script that exports a checkpoint from the e2e database to the
filesystem store so the auditor has data to fetch immediately.

Usage:
    DATABASE_URL="postgresql+asyncpg://sparket:sparket@127.0.0.1:5435/sparket_e2e" \
      uv run python scripts/dev/bootstrap_checkpoint.py
"""

import asyncio
import os
import sys

# Ensure project root on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


async def main() -> None:
    import bittensor as bt

    bt.logging.info({"bootstrap": "starting"})

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    # Set up database connection via DBM
    os.environ.setdefault("SPARKET_DATABASE__URL", db_url)
    os.environ.setdefault("SPARKET_TEST_MODE", "true")

    from sparket.validator.config.config import Config as ValidatorAppConfig
    from sparket.validator.database.init import initialize as init_db
    from sparket.validator.database.dbm import DBM
    from sparket.validator.ledger.exporter import LedgerExporter
    from sparket.validator.ledger.store.filesystem import FilesystemStore

    app_config = ValidatorAppConfig()
    init_db(app_config)
    dbm = DBM.get_manager(app_config)

    # Use the local-validator wallet for signing
    wallet = bt.Wallet(name="local-validator", hotkey="default")
    netuid = int(os.environ.get("SPARKET_CHAIN__NETUID", "2"))
    data_dir = os.environ.get(
        "SPARKET_LEDGER__DATA_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "sparket", "data", "ledger"),
    )

    exporter = LedgerExporter(database=dbm, wallet=wallet, netuid=netuid)
    store = FilesystemStore(data_dir=data_dir)

    bt.logging.info({"bootstrap": "exporting_checkpoint", "netuid": netuid, "data_dir": data_dir})

    checkpoint = await exporter.export_checkpoint()
    cp_id = await store.put_checkpoint(checkpoint)

    bt.logging.info({
        "bootstrap": "checkpoint_exported",
        "cp_id": cp_id,
        "epoch": checkpoint.manifest.checkpoint_epoch,
        "miners": len(checkpoint.accumulators),
        "roster": len(checkpoint.roster),
    })

    # Also export an empty delta so the auditor has a complete state
    from datetime import datetime, timedelta, timezone

    since = datetime.now(timezone.utc) - timedelta(hours=12)
    delta = await exporter.export_delta(since=since)
    if delta.settled_submissions:
        delta_id = await store.put_delta(delta)
        bt.logging.info({
            "bootstrap": "delta_exported",
            "delta_id": delta_id,
            "submissions": len(delta.settled_submissions),
        })
    else:
        bt.logging.info({"bootstrap": "no_settled_submissions_for_delta"})

    await dbm.engine.dispose()
    bt.logging.info({"bootstrap": "done"})


if __name__ == "__main__":
    asyncio.run(main())
