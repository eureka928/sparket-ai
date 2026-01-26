import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

import bittensor as bt

# Add project root to path
sys.path.append(os.getcwd())

from sparket.validator.config.config import Config
from sparket.validator.database.dbm import DBM
from sparket.validator.database.init import initialize as init_db
from sparket.validator.database.init.seed import _seed_reference_minimal
from sparket.validator.services import SportsDataIngestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single SportsDataIO ingestion pass.")
    parser.add_argument(
        "--league",
        dest="leagues",
        action="append",
        help="Optional league codes to ingest (e.g., nfl, nba, epl). Repeat for multiple leagues.",
    )
    parser.add_argument(
        "--now",
        dest="run_now",
        help="Optional ISO8601 timestamp (UTC) to use instead of current time.",
    )
    return parser.parse_args()


async def async_main(config: Config, args: argparse.Namespace):
    bt.logging.info("Creating DBM...")
    dbm = DBM.get_manager(config)
    ingestor: SportsDataIngestor | None = None

    try:
        # 4. Seed Data
        bt.logging.info("Seeding Reference Data...")
        await _seed_reference_minimal(config)
        ingestor = SportsDataIngestor(database=dbm)

        if args.leagues:
            allowed = {code.lower() for code in args.leagues}
            filtered = {code: state for code, state in ingestor.leagues.items() if code.value.lower() in allowed}
            missing = allowed.difference({code.value.lower() for code in filtered})
            if missing:
                bt.logging.warning({"sdio_ingestor_test": {"ignored_leagues": sorted(missing)}})
            ingestor.leagues = filtered

        run_now = None
        if args.run_now:
            try:
                parsed = datetime.fromisoformat(args.run_now)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                run_now = parsed.astimezone(timezone.utc)
            except Exception as exc:
                bt.logging.warning({"sdio_ingestor_test": {"invalid_now": args.run_now, "error": str(exc)}})

        # 5. Run ingestion once to validate wiring
        bt.logging.info("Running SportsDataIO ingestor once...")
        await ingestor.run_once(now=run_now)
        bt.logging.info(
            {
                "sdio_ingestor_test": {
                    "tracked_events_total": len(ingestor.tracked_events),
                    "leagues": list(ingestor.leagues.keys()),
                }
            }
        )

    except Exception as e:
        bt.logging.error(f"An error occurred: {e}")
        import traceback

        bt.logging.error(traceback.format_exc())
        raise
    finally:
        if ingestor is not None:
            try:
                await ingestor.close()
            except Exception:
                pass
        if dbm:
            try:
                await dbm.engine.dispose()
            except Exception:
                pass


def main():
    bt.logging.set_debug(True)
    args = parse_args()
    bt.logging.info({"sdio_ingestor_test": {"args": vars(args)}})

    bt.logging.info("--- Starting Manual Ingest Test ---")

    # 1. Load Config
    bt.logging.info("Loading Config...")
    config = Config()

    # 2. Initialize DB (Sync)
    bt.logging.info("Initializing Database (Sync)...")
    init_db(config)

    # 3. Run Async Logic
    bt.logging.info("Running Async Logic...")
    asyncio.run(async_main(config, args))
    bt.logging.info("--- Test Completed ---")


if __name__ == "__main__":
    main()
