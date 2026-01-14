"""
Refresh only the `players` table from FPL bootstrap-static.

Why: full `load_data.py` also fetches per-player history (slow).
This script only updates Player fields like position/team/price.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.services.fpl_api import FPLAPIService
from app.services.etl_service import ETLService
from app.database import SessionLocal
from app.models import Player

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("=" * 60)
    logger.info("REFRESH PLAYERS (BOOTSTRAP -> DB)")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")

    fpl_api = FPLAPIService()
    etl = ETLService()
    try:
        bootstrap = await fpl_api.get_bootstrap_data()
        players = fpl_api.extract_players_from_bootstrap(bootstrap)
        logger.info(f"Fetched {len(players)} players from bootstrap-static")

        result = await etl.bulk_upsert_players(players)
        logger.info(f"Bulk upsert result: {result}")

        # Quick verification
        db = SessionLocal()
        try:
            positions = db.query(Player.position).distinct().all()
            logger.info(f"Distinct positions in DB: {[p[0] for p in positions]}")
        finally:
            db.close()

    finally:
        await fpl_api.close()
        await etl.close()


if __name__ == "__main__":
    asyncio.run(main())

