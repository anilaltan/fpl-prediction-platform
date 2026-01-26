"""
Database Population Script
FPL API'den veri çekip veritabanına kaydeder.

Akış:
1. Takımları çekip kaydet
2. Oyuncuları çekip kaydet
3. Mevcut gameweek istatistiklerini çekip kaydet
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.fpl.service import FPLAPIService
from app.services.etl_service import ETLService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def populate_database():
    """
    Ana veritabanı doldurma fonksiyonu.
    Service modüllerini kullanarak FPL API'den veri çekip DB'ye kaydeder.
    """
    fpl_service = None
    etl_service = None

    try:
        # Initialize services
        logger.info("Initializing services...")
        fpl_service = FPLAPIService(rate_limit_delay=0.1)
        etl_service = ETLService()

        # ==================== STEP 1: Fetch and Save Teams ====================
        logger.info("=" * 60)
        logger.info("STEP 1: Fetching teams from FPL API...")
        logger.info("=" * 60)

        bootstrap_data = await fpl_service.get_bootstrap_data(use_cache=False)
        teams_data = fpl_service.extract_teams_from_bootstrap(bootstrap_data)

        logger.info(f"Found {len(teams_data)} teams")
        logger.info("Saving teams to database...")

        teams_result = await etl_service.bulk_upsert_teams(teams_data)
        logger.info(f"Teams saved: {teams_result}")

        # ==================== STEP 2: Fetch and Save Players ====================
        logger.info("=" * 60)
        logger.info("STEP 2: Fetching players from FPL API...")
        logger.info("=" * 60)

        players_data = fpl_service.extract_players_from_bootstrap(bootstrap_data)

        logger.info(f"Found {len(players_data)} players")
        logger.info("Saving players to database...")

        players_result = await etl_service.bulk_upsert_players(players_data)
        logger.info(f"Players saved: {players_result}")

        # ==================== STEP 3: Fetch and Save Fixtures ====================
        logger.info("=" * 60)
        logger.info("STEP 3: Fetching fixtures from FPL API...")
        logger.info("=" * 60)

        # Fetch all fixtures (not filtered by gameweek to get all fixtures)
        fixtures_data = await fpl_service.get_fixtures(gameweek=None, future_only=False)
        
        logger.info(f"Found {len(fixtures_data)} fixtures from FPL API")
        
        # Filter out fixtures without gameweek (event) - these can't be saved
        fixtures_with_gameweek = [
            f for f in fixtures_data 
            if f.get("event") is not None and f.get("team_h") is not None and f.get("team_a") is not None
        ]
        skipped_count = len(fixtures_data) - len(fixtures_with_gameweek)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} fixtures without gameweek or team information")
        
        logger.info(f"Processing {len(fixtures_with_gameweek)} valid fixtures")
        logger.info("Saving fixtures to database...")

        # Extract fixtures with difficulty ratings for better data quality
        fixtures_with_difficulty = fpl_service.extract_fixtures_with_difficulty(
            fixtures_with_gameweek, teams_data
        )

        fixtures_result = await etl_service.bulk_upsert_fixtures(
            fixtures_with_difficulty, season="2025-26"
        )
        logger.info(f"Fixtures saved: {fixtures_result}")

        # ==================== STEP 4: Fetch and Save Current Gameweek Stats ====================
        logger.info("=" * 60)
        logger.info("STEP 4: Fetching current gameweek statistics...")
        logger.info("=" * 60)

        current_gameweek = await fpl_service.get_current_gameweek()
        if not current_gameweek:
            logger.warning("Could not determine current gameweek. Skipping stats...")
            return

        logger.info(f"Current gameweek: {current_gameweek}")
        logger.info("Fetching player statistics for current gameweek...")
        logger.info("This may take several minutes due to rate limiting (801 players)...")

        # Use bulk_save_gameweek_stats from FPLAPIService
        # This method handles fetching player data and saving stats
        stats_result = await fpl_service.bulk_save_gameweek_stats(
            gameweek=current_gameweek,
            season="2025-26",
            max_players=None,  # Process all players
        )

        logger.info(f"Gameweek stats saved: {stats_result}")

        # ==================== Summary ====================
        logger.info("=" * 60)
        logger.info("DATABASE POPULATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Teams: {teams_result.get('inserted', 0)} inserted, {teams_result.get('errors', 0)} errors")
        logger.info(f"Players: {players_result.get('inserted', 0)} inserted, {players_result.get('errors', 0)} errors")
        logger.info(f"Fixtures: {fixtures_result.get('inserted', 0)} inserted, {fixtures_result.get('errors', 0)} errors")
        logger.info(f"Gameweek Stats: {stats_result.get('saved', 0)} saved, {stats_result.get('errors', 0)} errors")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during database population: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        if fpl_service:
            await fpl_service.close()
        if etl_service:
            await etl_service.close()
        logger.info("Services closed.")


if __name__ == "__main__":
    asyncio.run(populate_database())
