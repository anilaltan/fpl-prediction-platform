"""
Batch Prediction Update Script
Calculates ML predictions for all players and stores them in Prediction table.
This script runs in background to pre-calculate predictions, making API responses ultra-fast.
"""
import asyncio
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.database import SessionLocal
from app.models import Player, PlayerGameweekStats, Prediction
from app.services.ml_engine import PLEngine
from app.services.fpl import FPLAPIService

logger = logging.getLogger(__name__)


async def update_predictions_for_gameweek(
    db: Session, gameweek: int, season: str = "2025-26"
) -> dict:
    """
    Calculate and store ML predictions for all players for a specific gameweek.

    Args:
        db: Database session
        gameweek: Gameweek number
        season: Season string (default: "2025-26")

    Returns:
        Dictionary with update statistics
    """
    try:
        logger.info(
            f"[BATCH PREDICTION] Starting prediction update for GW{gameweek} (season: {season})"
        )

        # Initialize ML engine
        ml_engine = PLEngine()
        ml_engine._ensure_models_loaded()

        if (
            not ml_engine.xmins_model.is_trained
            or not ml_engine.attack_model.xg_trained
        ):
            logger.warning(
                "[BATCH PREDICTION] ML models not trained, skipping prediction update"
            )
            return {
                "status": "skipped",
                "reason": "ML models not trained",
                "gameweek": gameweek,
                "updated_count": 0,
            }

        # Initialize FPL API service for ownership data
        fpl_api = FPLAPIService()

        # Fetch all players
        players = db.query(Player).all()
        logger.info(f"[BATCH PREDICTION] Processing {len(players)} players")

        # Fetch FPL bootstrap data once for ownership and availability
        ownership_map = {}
        availability_map = {}  # Maps fpl_id to {chance_of_playing_next_round, status}
        try:
            bootstrap = await fpl_api.get_bootstrap_data()
            elements = bootstrap.get("elements", [])
            for element in elements:
                fpl_id = element.get("id")
                if fpl_id:
                    ownership_map[fpl_id] = float(
                        element.get("selected_by_percent", 0.0)
                    )
                    # Store availability info: chance_of_playing_next_round and status
                    chance = element.get("chance_of_playing_next_round")
                    status = element.get("status", "a")
                    availability_map[fpl_id] = {
                        "chance_of_playing_next_round": chance,
                        "status": status,
                    }
        except Exception as e:
            logger.warning(
                f"[BATCH PREDICTION] Failed to fetch ownership/availability data: {str(e)}"
            )

        # Batch load all stats
        all_stats = (
            db.query(PlayerGameweekStats)
            .filter(PlayerGameweekStats.season == season)
            .all()
        )

        # Group stats by fpl_id
        stats_by_player = {}
        for stat in all_stats:
            if stat.fpl_id not in stats_by_player:
                stats_by_player[stat.fpl_id] = []
            stats_by_player[stat.fpl_id].append(stat)

        # Sort each player's stats by gameweek descending
        for fpl_id in stats_by_player:
            stats_by_player[fpl_id].sort(key=lambda x: x.gameweek or 0, reverse=True)

        # Process each player with individual transaction management
        # Each player gets its own transaction - commit after every player (success or failure)
        # This ensures complete isolation: one player's failure cannot affect others
        updated_count = 0
        error_count = 0

        for idx, player in enumerate(players, 1):
            # Start fresh transaction for each player
            try:
                # Get player's historical stats
                player_stats = stats_by_player.get(
                    player.id, []
                )  # Player.id is the FPL ID
                latest_stat = player_stats[0] if player_stats else None

                # Build player_data for ML engine
                # Convert Decimal price to float
                player_price = float(player.price) if player.price else 5.0
                player_data = {
                    "fpl_id": player.id,  # Player.id is the FPL ID
                    "position": player.position or "MID",
                    "price": player_price,
                    "status": getattr(player, "status", "a") or "a",
                    "minutes": 0,
                    "xg_per_90": 0.0,
                    "xa_per_90": 0.0,
                    "recent_minutes": [45],
                    "recent_xg": [0.1],
                    "recent_xa": [0.05],
                    "form": 0.0,
                    "ict_index": 0.0,
                }

                if latest_stat:
                    player_minutes = latest_stat.minutes or 0
                    player_form = float(latest_stat.total_points or 0) / 10.0

                    mins_divisor = max(player_minutes, 1)
                    player_data.update(
                        {
                            "minutes": player_minutes,
                            "xg_per_90": float(latest_stat.xg or 0)
                            * (90.0 / mins_divisor),
                            "xa_per_90": float(latest_stat.xa or 0)
                            * (90.0 / mins_divisor),
                            "form": player_form,
                            "ict_index": float(latest_stat.ict_index or 0),
                        }
                    )

                    # Get recent stats (up to 5 gameweeks before current)
                    recent_stats = [
                        s for s in player_stats if s.gameweek and s.gameweek < gameweek
                    ][:5]
                    if recent_stats:
                        player_data["recent_minutes"] = [
                            s.minutes or 0 for s in recent_stats
                        ]
                        player_data["recent_xg"] = [
                            float(s.xg or 0) for s in recent_stats
                        ]
                        player_data["recent_xa"] = [
                            float(s.xa or 0) for s in recent_stats
                        ]

                # Get prediction from ML engine
                prediction = ml_engine.predict(
                    player_data=player_data, fixture_data=None
                )

                # Extract values - ensure all are floats, never None
                xp = float(prediction.get("expected_points", 0.0) or 0.0)
                xg = float(prediction.get("xg", 0.0) or 0.0)
                xa = float(prediction.get("xa", 0.0) or 0.0)
                xmins = float(prediction.get("xmins", 45.0) or 45.0)
                xcs = float(prediction.get("xcs", 0.0) or 0.0)
                defcon_score = float(prediction.get("defcon_points", 0.0) or 0.0)
                # CRITICAL: confidence_score must NEVER be None - ensure it's always a valid float
                confidence_score_raw = prediction.get("confidence_score")
                if confidence_score_raw is None:
                    confidence_score = 0.7  # Default confidence
                else:
                    confidence_score = float(confidence_score_raw)
                # Clamp to valid range [0.0, 1.0]
                confidence_score = max(0.0, min(1.0, confidence_score))

                # ======================================================================
                # HARD FILTER: Check player availability (injury/suspension)
                # ======================================================================
                player_availability = availability_map.get(
                    player.id, {}
                )  # Player.id is the FPL ID
                chance_of_playing = player_availability.get(
                    "chance_of_playing_next_round"
                )
                player_status = player_availability.get("status", "a")

                # IMPORTANT: None means 100% available (not injured)
                # Only filter if chance is explicitly 0, 25, or 50 (or status != 'a' with non-None chance)
                should_filter = False
                filter_reason = ""

                if chance_of_playing is not None:
                    # Explicitly low chance values indicate injury/suspension
                    if chance_of_playing in [0, 25, 50]:
                        should_filter = True
                        filter_reason = f"Chance: {chance_of_playing}%"
                elif player_status != "a":
                    # Status is not 'a' (available) and chance is None - treat as unavailable
                    # But only if status is explicitly not 'a'
                    if player_status in [
                        "d",
                        "i",
                        "n",
                        "s",
                        "u",
                    ]:  # doubtful, injured, not available, suspended, unavailable
                        should_filter = True
                        filter_reason = f"Status: {player_status}"

                if should_filter:
                    # Set xp and xmins to 0 for injured/suspended players
                    xp = 0.0
                    xmins = 0.0
                    player_name = getattr(player, "name", f"ID:{player.id}")
                    logger.info(
                        f"[BATCH PREDICTION] Player {player_name} (ID:{player.id}) set to 0 xP due to injury/suspension ({filter_reason})"
                    )

                # Check if prediction already exists
                existing = (
                    db.query(Prediction)
                    .filter(
                        and_(
                            Prediction.fpl_id == player.id,  # Player.id is the FPL ID
                            Prediction.gameweek == gameweek,
                            Prediction.season == season,
                        )
                    )
                    .first()
                )

                if existing:
                    # Update existing prediction
                    existing.xp = xp
                    existing.xg = xg
                    existing.xa = xa
                    existing.xmins = xmins
                    existing.xcs = xcs
                    existing.defcon_score = defcon_score
                    existing.confidence_score = confidence_score
                    existing.player_id = player.id
                    existing.updated_at = datetime.now()
                else:
                    # Create new prediction
                    new_prediction = Prediction(
                        fpl_id=player.id,  # Player.id is the FPL ID
                        gameweek=gameweek,
                        season=season,
                        xp=xp,
                        xg=xg,
                        xa=xa,
                        xmins=xmins,
                        xcs=xcs,
                        defcon_score=defcon_score,
                        confidence_score=confidence_score,
                        player_id=player.id,
                        model_version="plengine_latest",
                    )
                    db.add(new_prediction)

                # Commit transaction for this player immediately
                # This ensures that if the next player fails, this one is already saved
                db.commit()
                updated_count += 1

                # Log progress every 100 players
                if updated_count % 100 == 0:
                    logger.info(
                        f"[BATCH PREDICTION] Progress: {updated_count}/{len(players)} players processed"
                    )

            except Exception as e:
                # CRITICAL: Rollback this player's transaction and clear session state
                try:
                    db.rollback()
                    db.expire_all()  # Clear all cached objects from session
                except Exception as rollback_error:
                    # If rollback fails, log but continue - we'll try to start fresh next iteration
                    logger.warning(
                        f"[BATCH PREDICTION] Rollback failed for player {player.id}: {str(rollback_error)}"
                    )
                    # Try to expire anyway to clear session
                    try:
                        db.expire_all()
                    except Exception:
                        pass

                error_count += 1
                player_name = getattr(player, "name", f"ID:{player.id}")
                logger.error(
                    f"[BATCH PREDICTION] Error processing player {player_name} (ID:{player.id}): {str(e)}",
                    exc_info=True,  # Include full traceback for debugging
                )
                # Continue to next player - don't let one error stop the entire process
                # The rollback above ensures the next player starts with a clean transaction
                continue

        # No final commit needed - we commit after each player
        # This is just a safety check to ensure session is clean
        try:
            # Check if there are any pending changes (shouldn't be, but just in case)
            if db.dirty or db.new or db.deleted:
                logger.warning(
                    "[BATCH PREDICTION] Found pending changes at end, committing..."
                )
                db.commit()
        except Exception as e:
            logger.warning(f"[BATCH PREDICTION] Final safety commit warning: {str(e)}")
            try:
                db.rollback()
            except Exception:
                pass

        logger.info(
            f"[BATCH PREDICTION] Completed: {updated_count} predictions updated, "
            f"{error_count} errors for GW{gameweek}"
        )

        return {
            "status": "success",
            "gameweek": gameweek,
            "season": season,
            "updated_count": updated_count,
            "error_count": error_count,
            "total_players": len(players),
        }

    except Exception as e:
        db.rollback()
        logger.error(f"[BATCH PREDICTION] Failed to update predictions: {str(e)}")
        raise


async def update_predictions_for_current_gameweek():
    """
    Update predictions for the current gameweek.
    Fetches current gameweek from FPL API and updates predictions.
    """
    try:
        db = SessionLocal()
        try:
            fpl_api = FPLAPIService()
            current_gw = await fpl_api.get_current_gameweek()

            if not current_gw:
                logger.warning(
                    "[BATCH PREDICTION] Could not determine current gameweek"
                )
                return

            result = await update_predictions_for_gameweek(db, current_gw)
            logger.info(f"[BATCH PREDICTION] Update result: {result}")
        finally:
            db.close()
    except Exception as e:
        logger.error(
            f"[BATCH PREDICTION] Error in update_predictions_for_current_gameweek: {str(e)}"
        )


if __name__ == "__main__":
    # CLI usage: python update_predictions.py [gameweek]
    import sys

    db = SessionLocal()
    try:
        if len(sys.argv) > 1:
            gameweek = int(sys.argv[1])
            asyncio.run(update_predictions_for_gameweek(db, gameweek))
        else:
            asyncio.run(update_predictions_for_current_gameweek())
    finally:
        db.close()
