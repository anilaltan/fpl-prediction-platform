"""
Data Loading Script for FPL Prediction Platform
Loads player and gameweek statistics data into PostgreSQL for backtesting.
"""
import asyncio
import sys
import os
import logging
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.fpl_api import FPLAPIService
from app.services.etl_service import ETLService
from app.database import SessionLocal, Base, engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def load_data(
    season: str = "2025-26",
    gameweek: int = None,
    max_players: int = None
):
    """
    Load FPL data into PostgreSQL.
    
    Args:
        season: Season string (default: "2025-26")
        gameweek: Optional specific gameweek to load (None = all)
        max_players: Optional limit on number of players (for testing)
    """
    logger.info("=" * 60)
    logger.info("FPL DATA LOADING SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Season: {season}")
    logger.info(f"Gameweek: {gameweek or 'All'}")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("")
    
    # Create database tables if needed
    logger.info("Creating database tables if needed...")
    Base.metadata.create_all(bind=engine)
    logger.info("✓ Database tables ready")
    logger.info("")
    
    # Initialize services
    fpl_api = FPLAPIService()
    etl_service = ETLService()
    
    try:
        # Run ETL sync
        logger.info("Starting ETL sync...")
        result = await etl_service.sync_from_fpl_api(
            fpl_api,
            gameweek=gameweek,
            season=season
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("ETL SYNC COMPLETED")
        logger.info("=" * 60)
        
        # Extract results from nested structure
        players_result = result.get('players', {})
        stats_result = result.get('gameweek_stats', {})
        
        players_saved = players_result.get('saved', 0) if isinstance(players_result, dict) else 0
        stats_saved = stats_result.get('saved', 0) if isinstance(stats_result, dict) else 0
        total_players = result.get('total_players', 0)
        total_stats = result.get('total_stats_records', 0)
        
        logger.info(f"Total players processed: {total_players}")
        logger.info(f"Players saved/updated: {players_saved}")
        logger.info(f"Total stats records: {total_stats}")
        logger.info(f"Stats records saved/updated: {stats_saved}")
        
        # Verify in database
        from app.models import Player, PlayerGameweekStats
        db = SessionLocal()
        try:
            db_players = db.query(Player).count()
            db_stats = db.query(PlayerGameweekStats).count()
            logger.info("")
            logger.info("Database Verification:")
            logger.info(f"  Players in DB: {db_players}")
            logger.info(f"  Gameweek Stats in DB: {db_stats}")
            
            if db_stats > 0:
                seasons = db.query(PlayerGameweekStats.season).distinct().all()
                gameweeks = db.query(PlayerGameweekStats.gameweek).distinct().order_by(PlayerGameweekStats.gameweek).all()
                logger.info(f"  Seasons: {[s[0] for s in seasons]}")
                logger.info(f"  Gameweeks: {len(gameweeks)} (first: {gameweeks[0][0] if gameweeks else 'N/A'}, last: {gameweeks[-1][0] if gameweeks else 'N/A'})")
        finally:
            db.close()
        
        logger.info("")
        logger.info(f"Completed at: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during data loading: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup
        await fpl_api.close()
        await etl_service.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load FPL data into PostgreSQL')
    parser.add_argument('--season', type=str, default='2025-26', help='Season string (default: 2025-26)')
    parser.add_argument('--gameweek', type=int, default=None, help='Specific gameweek to load (default: all)')
    parser.add_argument('--max-players', type=int, default=None, help='Maximum players to process (for testing)')
    
    args = parser.parse_args()
    
    # Run async function
    result = asyncio.run(load_data(
        season=args.season,
        gameweek=args.gameweek,
        max_players=args.max_players
    ))
    
    # Exit with error code if errors occurred
    if result.get('errors', 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)