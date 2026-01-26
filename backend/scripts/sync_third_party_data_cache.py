"""
Third-Party Data Cache Sync Script
Pre-fetches and caches Understat/FBref data for all players to enable fast ML predictions.

This script:
1. Fetches all FPL players
2. Enriches each player with Understat/FBref data
3. Stores enriched data in third_party_data_cache table
4. Uses entity resolution for accurate player matching

Run this as a background job (e.g., daily) to keep cache fresh.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models import Player, ThirdPartyDataCache
from app.services.fpl import FPLAPIService
from app.services.entity_resolution import EntityResolutionService
from app.services.third_party_data import ThirdPartyDataService
from sqlalchemy.dialects.postgresql import insert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def sync_third_party_cache(
    season: str = "2025-26",
    max_players: Optional[int] = None,
    force_refresh: bool = False,
    update_existing: bool = True,
) -> Dict:
    """
    Sync third-party data cache for all players.

    Args:
        season: Season identifier (default: "2025-26")
        max_players: Maximum number of players to process (None for all)
        force_refresh: If True, refresh even if cache exists
        update_existing: If True, update existing cache entries

    Returns:
        Dictionary with sync results
    """
    db = None
    fpl_service = None
    third_party_service = None
    entity_resolution_service = None

    try:
        db = SessionLocal()

        # Initialize services
        logger.info("Initializing services...")
        fpl_service = FPLAPIService(rate_limit_delay=0.1)
        entity_resolution_service = EntityResolutionService()
        third_party_service = ThirdPartyDataService(
            entity_resolution_service=entity_resolution_service, db_session=db
        )

        # Load master map for entity resolution
        logger.info("Loading master ID map for entity resolution...")
        await entity_resolution_service.load_master_map()

        # Fetch all FPL players
        logger.info("Fetching FPL players...")
        bootstrap = await fpl_service.get_bootstrap_data()
        fpl_players = fpl_service.extract_players_from_bootstrap(bootstrap)

        if max_players:
            fpl_players = fpl_players[:max_players]
            logger.info(f"Processing {len(fpl_players)} players (limited from {len(fpl_players)})")
        else:
            logger.info(f"Processing {len(fpl_players)} players")

        # Get existing cache entries
        existing_cache = {}
        if not force_refresh:
            existing_entries = db.query(ThirdPartyDataCache).filter(
                ThirdPartyDataCache.season == season
            ).all()
            existing_cache = {entry.player_id: entry for entry in existing_entries}
            logger.info(f"Found {len(existing_cache)} existing cache entries")

        # Process players
        success_count = 0
        error_count = 0
        skipped_count = 0
        updated_count = 0
        created_count = 0

        import gc
        for idx, fpl_player in enumerate(fpl_players, 1):
            player_id = fpl_player.get("id")
            player_name = fpl_player.get("web_name", "") or fpl_player.get("name", "")

            if not player_id:
                logger.warning(f"Skipping player without ID: {fpl_player}")
                skipped_count += 1
                continue

            # Check if cache exists and should be skipped
            if not force_refresh and player_id in existing_cache:
                if not update_existing:
                    logger.debug(f"Skipping {player_name} (ID: {player_id}) - cache exists")
                    skipped_count += 1
                    continue

            try:
                # Enrich player with third-party data
                enriched = await third_party_service.enrich_player_data(
                    fpl_player, season=season.split("-")[0]  # Convert "2025-26" to "2025"
                )

                # Extract third-party data
                cache_data = {
                    "player_id": player_id,
                    "season": season,
                    # Understat metrics
                    "understat_xg": _safe_decimal(enriched.get("understat_xg")),
                    "understat_xa": _safe_decimal(enriched.get("understat_xa")),
                    "understat_npxg": _safe_decimal(enriched.get("understat_npxg")),
                    "understat_xg_per_90": _safe_decimal(enriched.get("understat_xg_per_90")),
                    "understat_xa_per_90": _safe_decimal(enriched.get("understat_xa_per_90")),
                    "understat_npxg_per_90": _safe_decimal(enriched.get("understat_npxg_per_90")),
                    # FBref metrics
                    "fbref_blocks": _safe_int(enriched.get("fbref_blocks")),
                    "fbref_blocks_per_90": _safe_decimal(enriched.get("fbref_blocks_per_90")),
                    "fbref_interventions": _safe_int(enriched.get("fbref_interventions")),
                    "fbref_interventions_per_90": _safe_decimal(enriched.get("fbref_interventions_per_90")),
                    "fbref_tackles": _safe_int(enriched.get("fbref_tackles")),
                    "fbref_interceptions": _safe_int(enriched.get("fbref_interceptions")),
                    "fbref_passes": _safe_int(enriched.get("fbref_passes")),
                    "fbref_passes_per_90": _safe_decimal(enriched.get("fbref_passes_per_90")),
                    # Metadata
                    "data_source": _determine_data_source(enriched),
                    "confidence_score": _safe_decimal(enriched.get("confidence", 0.0)),
                    "last_updated": datetime.utcnow(),
                }

                # Upsert cache entry
                stmt = insert(ThirdPartyDataCache).values(cache_data)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_third_party_cache_player_season",
                    set_={
                        **{k: v for k, v in cache_data.items() if k not in ["player_id", "season"]},
                        "last_updated": datetime.utcnow(),
                    },
                )
                db.execute(stmt)
                db.commit()

                if player_id in existing_cache:
                    updated_count += 1
                else:
                    created_count += 1

                success_count += 1

                # Log progress
                if idx % 10 == 0:
                    logger.info(
                        f"Progress: {idx}/{len(fpl_players)} - "
                        f"Success: {success_count}, Errors: {error_count}, "
                        f"Created: {created_count}, Updated: {updated_count}"
                    )
                    gc.collect()

            except Exception as e:
                error_count += 1
                logger.error(
                    f"Error enriching player {player_name} (ID: {player_id}): {str(e)}",
                    exc_info=True,
                )
                db.rollback()

        # Summary
        logger.info("=" * 60)
        logger.info("THIRD-PARTY DATA CACHE SYNC COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total players processed: {len(fpl_players)}")
        logger.info(f"Success: {success_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Skipped: {skipped_count}")
        logger.info(f"Created: {created_count}")
        logger.info(f"Updated: {updated_count}")
        logger.info("=" * 60)

        return {
            "success": True,
            "season": season,
            "total_players": len(fpl_players),
            "success_count": success_count,
            "error_count": error_count,
            "skipped_count": skipped_count,
            "created_count": created_count,
            "updated_count": updated_count,
        }

    except Exception as e:
        logger.error(f"Error during third-party cache sync: {str(e)}", exc_info=True)
        raise
    finally:
        if db:
            db.close()
        if fpl_service:
            await fpl_service.close()
        if entity_resolution_service:
            await entity_resolution_service.close()
        logger.info("Services closed.")


def _safe_decimal(value) -> Optional[Decimal]:
    """Safely convert value to Decimal."""
    if value is None:
        return None
    try:
        return Decimal(str(float(value)))
    except (ValueError, TypeError):
        return None


def _safe_int(value) -> Optional[int]:
    """Safely convert value to int."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _determine_data_source(enriched: Dict) -> str:
    """Determine data source from enriched data."""
    has_understat = any(
        enriched.get(k) is not None
        for k in ["understat_xg", "understat_xa", "understat_npxg"]
    )
    has_fbref = any(
        enriched.get(k) is not None
        for k in ["fbref_blocks", "fbref_interventions", "fbref_tackles"]
    )

    if has_understat and has_fbref:
        return "both"
    elif has_understat:
        return "understat"
    elif has_fbref:
        return "fbref"
    else:
        return "none"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync third-party data cache (Understat/FBref)"
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2025-26",
        help="Season identifier (default: 2025-26)",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Maximum number of players to process (default: all)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh even if cache exists",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip players with existing cache entries",
    )

    args = parser.parse_args()

    asyncio.run(
        sync_third_party_cache(
            season=args.season,
            max_players=args.max_players,
            force_refresh=args.force_refresh,
            update_existing=not args.skip_existing,
        )
    )
