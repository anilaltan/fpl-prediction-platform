"""
Batch Prediction Update Script
Calculates ML predictions for all players and stores them in Prediction table.
This script runs in background to pre-calculate predictions, making API responses ultra-fast.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

# Add backend directory to path for imports (works in Docker where /app is the backend root)
current_file = Path(__file__).resolve()
# Try /app first (Docker container), then fallback to relative path
if os.path.exists("/app"):
    backend_dir = Path("/app")
else:
    # Go up from app/scripts/ to backend/
    backend_dir = current_file.parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database import SessionLocal
from app.models import Player, PlayerGameweekStats, Prediction, ThirdPartyDataCache
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

        # Construct model path explicitly - use backend/models
        # In Docker: /app is backend mount, so /app/models = backend/models
        # In local: use backend/models relative to script location
        model_filename = f"plengine_model_gw{gameweek}_{season.replace('-', '_')}.pkl"
        model_path = None
        
        # Try possible model directories (priority: backend/models)
        possible_dirs = []
        
        # 1. Check backend/models relative to current file location (works in both Docker and local)
        current_file_dir = Path(__file__).resolve().parent
        # From app/scripts/update_predictions.py -> app/scripts -> app -> backend
        backend_models = current_file_dir.parent.parent.parent / "models"
        if backend_models.exists():
            possible_dirs.append(str(backend_models))
        
        # 2. Docker: /app/models (where /app is backend mount)
        if os.path.exists("/app"):
            possible_dirs.append("/app/models")
        
        # 3. Fallback: models/ in current directory
        possible_dirs.append("models")
        
        # Try to find model file in possible directories
        for models_dir in possible_dirs:
            candidate_path = os.path.join(models_dir, model_filename)
            if os.path.exists(candidate_path):
                model_path = candidate_path
                # Normalize path: if it's /app/models, show as backend/models for clarity
                display_path = model_path.replace("/app/models", "backend/models").replace("\\", "/")
                logger.info(f"[BATCH PREDICTION] Found model at: {display_path} (actual: {model_path})")
                break
        
        # If not found, report error
        if not model_path:
            checked_paths = [os.path.join(d, model_filename) for d in possible_dirs]
            logger.warning(
                f"[BATCH PREDICTION] Model file not found. Checked paths: {checked_paths}. "
                "Please train models first using: "
                "docker compose exec backend python3 scripts/train_ml_models.py"
            )
            return {
                "status": "skipped",
                "reason": f"Model file not found. Checked: {checked_paths}",
                "gameweek": gameweek,
                "updated_count": 0,
            }
        
        logger.info(f"[BATCH PREDICTION] Using model file: {model_path}")
        
        # Initialize ML engine with explicit model path
        ml_engine = PLEngine(model_path=model_path)
        
        # Load models asynchronously (we're in an async function)
        logger.info(f"[BATCH PREDICTION] Loading models from: {model_path}")
        logger.info(f"[BATCH PREDICTION] Model file exists: {os.path.exists(model_path)}")
        logger.info(f"[BATCH PREDICTION] Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'} bytes")
        
        # Log initial state before loading
        logger.info(
            f"[BATCH PREDICTION] Before load - xMins: is_trained={ml_engine.xmins_strategy.is_trained}, "
            f"model={'exists' if ml_engine.xmins_strategy.model is not None else 'None'}, "
            f"is_loaded={ml_engine.xmins_strategy.is_loaded}"
        )
        
        try:
            await ml_engine.async_load_models(model_path)
            
            # Log model loading status immediately after load
            logger.info(
                f"[BATCH PREDICTION] After load - xMins: is_trained={ml_engine.xmins_strategy.is_trained}, "
                f"model={'exists' if ml_engine.xmins_strategy.model is not None else 'None'}, "
                f"is_loaded={ml_engine.xmins_strategy.is_loaded}, "
                f"model_type={type(ml_engine.xmins_strategy.model).__name__ if ml_engine.xmins_strategy.model is not None else 'None'}"
            )
            logger.info(
                f"[BATCH PREDICTION] After load - Attack: xg_trained={ml_engine.attack_strategy.xg_trained}, "
                f"xa_trained={ml_engine.attack_strategy.xa_trained}, "
                f"xg_model={'exists' if ml_engine.attack_strategy.xg_model is not None else 'None'}, "
                f"xa_model={'exists' if ml_engine.attack_strategy.xa_model is not None else 'None'}, "
                f"is_loaded={ml_engine.attack_strategy.is_loaded}"
            )
        except Exception as e:
            logger.error(
                f"[BATCH PREDICTION] Failed to load models: {str(e)}. "
                "Please train models first using: "
                "docker compose exec backend python3 scripts/train_ml_models.py",
                exc_info=True
            )
            # Try explicit reload as fallback
            try:
                logger.info("[BATCH PREDICTION] Attempting explicit reload...")
                await ml_engine.async_load_models(model_path)
                logger.info("[BATCH PREDICTION] Explicit reload succeeded")
            except Exception as retry_error:
                logger.error(
                    f"[BATCH PREDICTION] Explicit reload also failed: {str(retry_error)}",
                    exc_info=True
                )
                return {
                    "status": "error",
                    "reason": f"Failed to load models: {str(e)}. Retry also failed: {str(retry_error)}",
                    "gameweek": gameweek,
                    "updated_count": 0,
                }

        # Verify models are loaded and trained (redundant check after async_load_models validation)
        # Check both is_trained flag and model object existence
        if not ml_engine.xmins_strategy.is_loaded or not ml_engine.xmins_strategy.is_trained or ml_engine.xmins_strategy.model is None:
            logger.error(
                "[BATCH PREDICTION] xMins model not loaded or not trained. "
                f"is_trained={ml_engine.xmins_strategy.is_trained}, "
                f"model={'exists' if ml_engine.xmins_strategy.model is not None else 'None'}, "
                f"is_loaded={ml_engine.xmins_strategy.is_loaded}, "
                f"model_path={ml_engine.model_path}. "
                "Please train models first using: "
                "docker compose exec backend python3 scripts/train_ml_models.py"
            )
            return {
                "status": "skipped",
                "reason": "xMins model not loaded or not trained",
                "gameweek": gameweek,
                "updated_count": 0,
            }
        
        if not ml_engine.attack_strategy.is_loaded or not ml_engine.attack_strategy.xg_trained or ml_engine.attack_strategy.xg_model is None:
            logger.warning(
                "[BATCH PREDICTION] Attack model not loaded or not trained. "
                f"is_loaded={ml_engine.attack_strategy.is_loaded}, "
                f"xg_trained={ml_engine.attack_strategy.xg_trained}, "
                f"xg_model={'exists' if ml_engine.attack_strategy.xg_model is not None else 'None'}. "
                "Please train models first using: "
                "docker compose exec backend python3 scripts/train_ml_models.py"
            )
            return {
                "status": "skipped",
                "reason": "Attack model not loaded or not trained",
                "gameweek": gameweek,
                "updated_count": 0,
            }
        
        logger.info("[BATCH PREDICTION] Models loaded and verified successfully")

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

        # Load third-party data cache
        logger.info("[BATCH PREDICTION] Loading third-party data cache...")
        third_party_cache = {}
        try:
            cache_entries = (
                db.query(ThirdPartyDataCache)
                .filter(ThirdPartyDataCache.season == season)
                .all()
            )
            for entry in cache_entries:
                third_party_cache[entry.player_id] = {
                    # Understat metrics
                    "understat_xg": float(entry.understat_xg) if entry.understat_xg else None,
                    "understat_xa": float(entry.understat_xa) if entry.understat_xa else None,
                    "understat_npxg": float(entry.understat_npxg) if entry.understat_npxg else None,
                    "understat_xg_per_90": float(entry.understat_xg_per_90) if entry.understat_xg_per_90 else None,
                    "understat_xa_per_90": float(entry.understat_xa_per_90) if entry.understat_xa_per_90 else None,
                    "understat_npxg_per_90": float(entry.understat_npxg_per_90) if entry.understat_npxg_per_90 else None,
                    # FBref metrics
                    "fbref_blocks": entry.fbref_blocks if entry.fbref_blocks else None,
                    "fbref_blocks_per_90": float(entry.fbref_blocks_per_90) if entry.fbref_blocks_per_90 else None,
                    "fbref_interventions": entry.fbref_interventions if entry.fbref_interventions else None,
                    "fbref_interventions_per_90": float(entry.fbref_interventions_per_90) if entry.fbref_interventions_per_90 else None,
                    "fbref_tackles": entry.fbref_tackles if entry.fbref_tackles else None,
                    "fbref_interceptions": entry.fbref_interceptions if entry.fbref_interceptions else None,
                    "fbref_passes": entry.fbref_passes if entry.fbref_passes else None,
                    "fbref_passes_per_90": float(entry.fbref_passes_per_90) if entry.fbref_passes_per_90 else None,
                }
            logger.info(f"[BATCH PREDICTION] Loaded third-party cache for {len(third_party_cache)} players")
        except Exception as e:
            logger.warning(f"[BATCH PREDICTION] Failed to load third-party cache: {str(e)}")
            third_party_cache = {}

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
                    "goals_per_90": 0.0,
                    "assists_per_90": 0.0,
                    "recent_minutes": [0.0] * 5,
                    "recent_xg": [0.0] * 5,
                    "recent_xa": [0.0] * 5,
                    "recent_cs": [0.0] * 5,
                    "form": 0.0,
                    "ict_index": 0.0,
                    # DefCon defaults
                    "blocks_per_90": 0.0,
                    "interventions_per_90": 0.0,
                    "passes_per_90": 0.0,
                    "defcon_floor_points": 0.0,
                    "avg_blocks": 0.0,
                    "avg_interventions": 0.0,
                    "avg_passes": 0.0,
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
                            "goals_per_90": float(latest_stat.goals or 0)
                            * (90.0 / mins_divisor),
                            "assists_per_90": float(latest_stat.assists or 0)
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
                        player_data["recent_cs"] = [
                            float(s.clean_sheets or 0) for s in recent_stats
                        ]
                        
                        # Calculate DefCon features from historical stats
                        # Get all historical stats for DefCon calculation
                        historical_stats = [
                            s for s in player_stats 
                            if s.gameweek and s.gameweek < gameweek
                        ]
                        if historical_stats:
                            total_minutes = sum(s.minutes or 0 for s in historical_stats)
                            games_played = len([s for s in historical_stats if (s.minutes or 0) > 0])
                            if games_played > 0 and total_minutes > 0:
                                total_blocks = sum(s.blocks or 0 for s in historical_stats)
                                total_interventions = sum(s.interventions or 0 for s in historical_stats)
                                total_passes = sum(s.passes or 0 for s in historical_stats)
                                
                                avg_minutes_per_game = total_minutes / games_played
                                blocks_per_90 = (total_blocks / games_played) * (90.0 / max(avg_minutes_per_game, 1.0))
                                interventions_per_90 = (total_interventions / games_played) * (90.0 / max(avg_minutes_per_game, 1.0))
                                passes_per_90 = (total_passes / games_played) * (90.0 / max(avg_minutes_per_game, 1.0))
                                
                                avg_blocks = total_blocks / games_played
                                avg_interventions = total_interventions / games_played
                                avg_passes = total_passes / games_played
                                
                                # DefCon floor points
                                defcon_floor_points = avg_blocks * 1.0 + avg_interventions * 1.0 + (avg_passes / 10.0) * 0.1
                                
                                player_data["blocks_per_90"] = float(blocks_per_90)
                                player_data["interventions_per_90"] = float(interventions_per_90)
                                player_data["passes_per_90"] = float(passes_per_90)
                                player_data["defcon_floor_points"] = float(defcon_floor_points)
                                player_data["avg_blocks"] = float(avg_blocks)
                                player_data["avg_interventions"] = float(avg_interventions)
                                player_data["avg_passes"] = float(avg_passes)
                            else:
                                # Default values if no games played
                                player_data["blocks_per_90"] = 0.0
                                player_data["interventions_per_90"] = 0.0
                                player_data["passes_per_90"] = 0.0
                                player_data["defcon_floor_points"] = 0.0
                                player_data["avg_blocks"] = 0.0
                                player_data["avg_interventions"] = 0.0
                                player_data["avg_passes"] = 0.0
                        else:
                            # Default values if no historical stats
                            player_data["blocks_per_90"] = 0.0
                            player_data["interventions_per_90"] = 0.0
                            player_data["passes_per_90"] = 0.0
                            player_data["defcon_floor_points"] = 0.0
                            player_data["avg_blocks"] = 0.0
                            player_data["avg_interventions"] = 0.0
                            player_data["avg_passes"] = 0.0
                    else:
                        # Default values if no recent stats - ensure all arrays have 5 elements
                        player_data["recent_minutes"] = [0.0] * 5
                        player_data["recent_xg"] = [0.0] * 5
                        player_data["recent_xa"] = [0.0] * 5
                        player_data["recent_cs"] = [0.0] * 5
                        player_data["blocks_per_90"] = 0.0
                        player_data["interventions_per_90"] = 0.0
                        player_data["passes_per_90"] = 0.0
                        player_data["defcon_floor_points"] = 0.0
                        player_data["avg_blocks"] = 0.0
                        player_data["avg_interventions"] = 0.0
                        player_data["avg_passes"] = 0.0

                # Get cached third-party data for this player
                cached_third_party = third_party_cache.get(player.id)
                # Filter out None values to avoid passing None to feature engine
                if cached_third_party:
                    cached_third_party = {
                        k: v for k, v in cached_third_party.items() if v is not None
                    }
                    # Only pass if we have actual data
                    if not cached_third_party:
                        cached_third_party = None
                
                # Get prediction from ML engine
                # Pass third_party_data if available (will be used by feature engineering)
                prediction = ml_engine.predict(
                    player_data=player_data,
                    fixture_data=None,
                    third_party_data=cached_third_party,
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
