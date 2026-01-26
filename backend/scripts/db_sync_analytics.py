"""
Database Analytics Synchronization Script
Runs MarketIntelligenceService based on existing predictions and syncs
entity_mappings using EntityResolutionService for FBref/Understat sync.

Flow:
1. Find gameweeks with existing predictions
2. Run MarketIntelligenceService for each gameweek
3. Populate entity_mappings table if empty (from master map)
4. Fetch and display entity mappings for FBref/Understat sync
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import distinct, and_

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models import Prediction, EntityMapping
from app.services.market_intelligence import MarketIntelligenceService
from app.services.entity_resolution import EntityResolutionService
from app.services.fpl import FPLAPIService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_gameweeks_with_predictions(
    db: Session, season: str = "2025-26"
) -> List[int]:
    """
    Get list of gameweeks that have predictions in the database.

    Args:
        db: Database session
        season: Season identifier (default: "2025-26")

    Returns:
        List of gameweek numbers (sorted)
    """
    gameweeks = (
        db.query(distinct(Prediction.gameweek))
        .filter(Prediction.season == season)
        .order_by(Prediction.gameweek)
        .all()
    )
    return [gw[0] for gw in gameweeks]


def run_market_intelligence(
    db: Session, gameweek: int, season: str = "2025-26"
) -> Dict:
    """
    Run MarketIntelligenceService for a specific gameweek.

    Args:
        db: Database session
        gameweek: Gameweek number
        season: Season identifier (default: "2025-26")

    Returns:
        Dictionary with results from MarketIntelligenceService
    """
    logger.info(f"Running MarketIntelligenceService for GW{gameweek}, {season}")

    market_intelligence_service = MarketIntelligenceService()

    result = market_intelligence_service.calculate_and_persist_market_intelligence(
        db=db,
        gameweek=gameweek,
        season=season,
        use_fpl_api_ownership=True,
        overvalued_ownership_threshold=30.0,
        differential_ownership_threshold=10.0,
    )

    return result


async def populate_entity_mappings(
    db: Session,
) -> Dict:
    """
    Populate entity_mappings table by resolving all FPL players.
    This syncs the master map data to the database.

    Args:
        db: Database session

    Returns:
        Dictionary with resolution report
    """
    logger.info("Populating entity_mappings table from master map...")

    fpl_service = None
    entity_resolution_service = None

    try:
        # Initialize services
        fpl_service = FPLAPIService(rate_limit_delay=0.1)
        entity_resolution_service = EntityResolutionService()

        # Load master map
        logger.info("Loading master ID map...")
        await entity_resolution_service.load_master_map()

        # Fetch all FPL players
        logger.info("Fetching FPL players from bootstrap data...")
        bootstrap = await fpl_service.get_bootstrap_data()
        fpl_players = fpl_service.extract_players_from_bootstrap(bootstrap)

        logger.info(f"Resolving {len(fpl_players)} FPL players...")

        # Resolve all players and store mappings
        report = entity_resolution_service.resolve_all_players(
            db=db, fpl_players=fpl_players, store_mappings=True
        )

        logger.info(
            f"Entity mappings populated: {report['mappings_stored']} stored, "
            f"{report['matched_count']} matched, {report['unmatched_count']} unmatched"
        )

        return report

    finally:
        if fpl_service:
            await fpl_service.close()


def fetch_entity_mappings(
    db: Session,
    limit: Optional[int] = None,
    manually_verified_only: bool = False,
) -> List[EntityMapping]:
    """
    Fetch entity mappings using EntityResolutionService.

    Args:
        db: Database session
        limit: Maximum number of mappings to return (None for all)
        manually_verified_only: If True, only return manually verified mappings

    Returns:
        List of EntityMapping objects
    """
    logger.info("Fetching entity mappings using EntityResolutionService")

    entity_resolution_service = EntityResolutionService()

    mappings = entity_resolution_service.get_all_mappings(
        db=db, limit=limit, manually_verified_only=manually_verified_only
    )

    return mappings


def display_entity_mappings_summary(mappings: List[EntityMapping]) -> None:
    """
    Display summary statistics of entity mappings.

    Args:
        mappings: List of EntityMapping objects
    """
    if not mappings:
        logger.info("No entity mappings found")
        return

    total = len(mappings)
    with_fbref = sum(1 for m in mappings if m.fbref_name)
    with_understat = sum(1 for m in mappings if m.understat_name)
    with_both = sum(
        1
        for m in mappings
        if m.fbref_name and m.understat_name
    )
    manually_verified = sum(1 for m in mappings if m.manually_verified)
    high_confidence = sum(
        1
        for m in mappings
        if m.confidence_score and float(m.confidence_score) >= 0.85
    )

    logger.info("=" * 60)
    logger.info("ENTITY MAPPINGS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total mappings: {total}")
    logger.info(f"With FBref name: {with_fbref} ({with_fbref/total*100:.1f}%)")
    logger.info(
        f"With Understat name: {with_understat} ({with_understat/total*100:.1f}%)"
    )
    logger.info(f"With both FBref and Understat: {with_both} ({with_both/total*100:.1f}%)")
    logger.info(
        f"Manually verified: {manually_verified} ({manually_verified/total*100:.1f}%)"
    )
    logger.info(
        f"High confidence (>=0.85): {high_confidence} ({high_confidence/total*100:.1f}%)"
    )
    logger.info("=" * 60)


async def sync_analytics(
    season: str = "2025-26",
    gameweek: Optional[int] = None,
    fetch_entity_mappings_flag: bool = True,
    entity_mappings_limit: Optional[int] = None,
    force_populate_entity_mappings: bool = False,
) -> Dict:
    """
    Main function to sync analytics data.

    Args:
        season: Season identifier (default: "2025-26")
        gameweek: Specific gameweek to process (None for all gameweeks with predictions)
        fetch_entity_mappings_flag: Whether to fetch entity mappings (default: True)
        entity_mappings_limit: Limit for entity mappings query (None for all)

    Returns:
        Dictionary with summary of operations
    """
    db = None
    market_intelligence_service = None

    try:
        db = SessionLocal()

        # ==================== STEP 1: Market Intelligence ====================
        logger.info("=" * 60)
        logger.info("STEP 1: Running MarketIntelligenceService")
        logger.info("=" * 60)

        if gameweek:
            # Process specific gameweek
            gameweeks_to_process = [gameweek]
            logger.info(f"Processing specific gameweek: {gameweek}")
        else:
            # Find all gameweeks with predictions
            gameweeks_to_process = get_gameweeks_with_predictions(db, season=season)
            logger.info(
                f"Found {len(gameweeks_to_process)} gameweeks with predictions: {gameweeks_to_process}"
            )

        if not gameweeks_to_process:
            logger.warning(
                f"No gameweeks with predictions found for season {season}. "
                "Skipping MarketIntelligenceService."
            )
        else:
            market_intelligence_service = MarketIntelligenceService()
            market_intelligence_results = []

            for gw in gameweeks_to_process:
                try:
                    result = run_market_intelligence(db, gameweek=gw, season=season)
                    market_intelligence_results.append(
                        {"gameweek": gw, "result": result}
                    )
                    logger.info(
                        f"✓ Market intelligence completed for GW{gw}: "
                        f"{result.get('total_players', 0)} players, "
                        f"{result.get('inserted', 0)} inserted, "
                        f"{result.get('updated', 0)} updated"
                    )
                except Exception as e:
                    logger.error(
                        f"✗ Error processing market intelligence for GW{gw}: {str(e)}",
                        exc_info=True,
                    )
                    market_intelligence_results.append(
                        {"gameweek": gw, "error": str(e)}
                    )

            logger.info("=" * 60)
            logger.info("MARKET INTELLIGENCE SUMMARY")
            logger.info("=" * 60)
            for result in market_intelligence_results:
                if "error" in result:
                    logger.warning(
                        f"GW{result['gameweek']}: ERROR - {result['error']}"
                    )
                else:
                    r = result["result"]
                    logger.info(
                        f"GW{result['gameweek']}: {r.get('total_players', 0)} players, "
                        f"{r.get('inserted', 0)} inserted, {r.get('updated', 0)} updated"
                    )
            logger.info("=" * 60)

        # ==================== STEP 2: Entity Mappings ====================
        if fetch_entity_mappings_flag:
            logger.info("=" * 60)
            logger.info("STEP 2: Entity Mappings Sync")
            logger.info("=" * 60)

            try:
                # Check if entity_mappings table is empty
                existing_count = db.query(EntityMapping).count()
                logger.info(f"Current entity mappings in database: {existing_count}")

                # If table is empty or force flag is set, populate it from master map
                if existing_count == 0 or force_populate_entity_mappings:
                    if force_populate_entity_mappings and existing_count > 0:
                        logger.info(
                            f"Force populate flag set. Repopulating entity mappings (current: {existing_count})..."
                        )
                    else:
                        logger.warning(
                            "Entity mappings table is empty. Populating from master map..."
                        )
                    populate_report = await populate_entity_mappings(db=db)
                    logger.info(
                        f"✓ Populated {populate_report.get('mappings_stored', 0)} entity mappings"
                    )
                else:
                    logger.info(
                        f"Entity mappings table already has {existing_count} entries. Skipping population."
                    )

                # Fetch and display entity mappings
                mappings = fetch_entity_mappings(
                    db=db, limit=entity_mappings_limit, manually_verified_only=False
                )
                display_entity_mappings_summary(mappings)

                # Log sample mappings for verification
                if mappings:
                    logger.info("\nSample entity mappings (first 5):")
                    for i, mapping in enumerate(mappings[:5], 1):
                        logger.info(
                            f"  {i}. FPL ID: {mapping.fpl_id}, "
                            f"Canonical: {mapping.canonical_name}, "
                            f"FBref: {mapping.fbref_name or 'N/A'}, "
                            f"Understat: {mapping.understat_name or 'N/A'}, "
                            f"Confidence: {mapping.confidence_score or 'N/A'}"
                        )

            except Exception as e:
                logger.error(
                    f"Error syncing entity mappings: {str(e)}", exc_info=True
                )

        # ==================== Summary ====================
        logger.info("=" * 60)
        logger.info("ANALYTICS SYNC COMPLETE")
        logger.info("=" * 60)

        return {
            "success": True,
            "season": season,
            "gameweeks_processed": len(gameweeks_to_process),
            "market_intelligence_results": market_intelligence_results
            if gameweeks_to_process
            else [],
            "entity_mappings_fetched": fetch_entity_mappings_flag,
        }

    except Exception as e:
        logger.error(f"Error during analytics sync: {str(e)}", exc_info=True)
        raise
    finally:
        if db:
            db.close()
        if market_intelligence_service:
            # Cleanup FPL API service if needed
            try:
                if (
                    hasattr(market_intelligence_service, "_fpl_api")
                    and market_intelligence_service._fpl_api
                ):
                    await market_intelligence_service._fpl_api.close()
            except Exception:
                pass
        logger.info("Database session closed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync analytics data: Market Intelligence and Entity Mappings"
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2025-26",
        help="Season identifier (default: 2025-26)",
    )
    parser.add_argument(
        "--gameweek",
        type=int,
        default=None,
        help="Specific gameweek to process (default: all gameweeks with predictions)",
    )
    parser.add_argument(
        "--skip-entity-mappings",
        action="store_true",
        help="Skip fetching/syncing entity mappings",
    )
    parser.add_argument(
        "--force-populate-entity-mappings",
        action="store_true",
        help="Force populate entity mappings even if table is not empty",
    )
    parser.add_argument(
        "--entity-mappings-limit",
        type=int,
        default=None,
        help="Limit number of entity mappings to fetch (default: all)",
    )

    args = parser.parse_args()

    asyncio.run(
        sync_analytics(
            season=args.season,
            gameweek=args.gameweek,
            fetch_entity_mappings_flag=not args.skip_entity_mappings,
            entity_mappings_limit=args.entity_mappings_limit,
            force_populate_entity_mappings=args.force_populate_entity_mappings,
        )
    )
