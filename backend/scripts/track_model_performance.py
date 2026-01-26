"""
Model Performance Tracking Script
Calculates and stores model performance metrics by comparing predictions with actual results.

This script:
1. Finds gameweeks with both predictions and actual stats
2. Compares Prediction.xp with PlayerGameweekStats.total_points
3. Calculates MAE, RMSE, and accuracy per gameweek
4. Saves results to model_performance table

Usage:
    docker compose exec backend python3 scripts/track_model_performance.py [season] [model_version]
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sqlalchemy import and_, distinct

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models import Prediction, PlayerGameweekStats, ModelPerformance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_gameweeks_with_data(
    db, season: str = "2025-26"
) -> List[int]:
    """
    Get list of gameweeks that have both predictions and actual stats.

    Args:
        db: Database session
        season: Season identifier (default: "2025-26")

    Returns:
        List of gameweek numbers (sorted)
    """
    # Get gameweeks with predictions
    pred_gameweeks = (
        db.query(distinct(Prediction.gameweek))
        .filter(Prediction.season == season)
        .all()
    )
    pred_gw_set = {gw[0] for gw in pred_gameweeks}

    # Get gameweeks with actual stats
    stats_gameweeks = (
        db.query(distinct(PlayerGameweekStats.gameweek))
        .filter(PlayerGameweekStats.season == season)
        .all()
    )
    stats_gw_set = {gw[0] for gw in stats_gameweeks}

    # Return intersection (gameweeks with both)
    common_gameweeks = sorted(pred_gw_set & stats_gw_set)
    return common_gameweeks


def calculate_performance_metrics(
    db, gameweek: int, season: str = "2025-26"
) -> Optional[Dict]:
    """
    Calculate performance metrics for a specific gameweek.

    Args:
        db: Database session
        gameweek: Gameweek number
        season: Season identifier (default: "2025-26")

    Returns:
        Dictionary with metrics (mae, rmse, accuracy) or None if no data
    """
    # Get predictions for this gameweek
    predictions = (
        db.query(Prediction)
        .filter(
            and_(
                Prediction.gameweek == gameweek,
                Prediction.season == season,
            )
        )
        .all()
    )

    # Get actual stats for this gameweek
    stats = (
        db.query(PlayerGameweekStats)
        .filter(
            and_(
                PlayerGameweekStats.gameweek == gameweek,
                PlayerGameweekStats.season == season,
            )
        )
        .all()
    )

    # Create lookup maps
    pred_map = {p.fpl_id: p.xp for p in predictions}
    actual_map = {s.fpl_id: s.total_points for s in stats}

    # Find matching players (have both prediction and actual)
    matching_players = set(pred_map.keys()) & set(actual_map.keys())

    if len(matching_players) == 0:
        logger.warning(f"No matching players for GW{gameweek}")
        return None

    # Calculate errors
    errors = [
        abs(pred_map[fpl_id] - actual_map[fpl_id])
        for fpl_id in matching_players
    ]

    if len(errors) == 0:
        return None

    # Calculate metrics
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean([e ** 2 for e in errors])))

    # Accuracy: percentage of predictions within 1 point
    accurate_predictions = sum(1 for e in errors if e <= 1.0)
    accuracy = float(accurate_predictions / len(errors))

    return {
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
        "n_predictions": len(matching_players),
    }


def track_model_performance(
    season: str = "2025-26", model_version: str = "5.0.0"
) -> Dict:
    """
    Track model performance for all available gameweeks.

    Args:
        season: Season identifier (default: "2025-26")
        model_version: Model version string (default: "5.0.0")

    Returns:
        Dictionary with summary statistics
    """
    db = SessionLocal()
    try:
        logger.info("=" * 60)
        logger.info("MODEL PERFORMANCE TRACKING")
        logger.info("=" * 60)
        logger.info(f"Season: {season}")
        logger.info(f"Model Version: {model_version}")

        # Get gameweeks with both predictions and actual stats
        gameweeks = get_gameweeks_with_data(db, season)
        logger.info(f"Found {len(gameweeks)} gameweeks with both predictions and actual stats")

        if len(gameweeks) == 0:
            logger.warning("No gameweeks with both predictions and actual stats found!")
            logger.warning("Make sure you have:")
            logger.warning("  1. Predictions in the predictions table")
            logger.warning("  2. Actual stats in the player_gameweek_stats table")
            return {"status": "no_data", "processed": 0}

        processed = 0
        updated = 0
        created = 0

        for gameweek in gameweeks:
            logger.info(f"Processing GW{gameweek}...")

            # Calculate metrics
            metrics = calculate_performance_metrics(db, gameweek, season)
            if metrics is None:
                logger.warning(f"  Skipping GW{gameweek} - no matching data")
                continue

            # Check if record already exists
            existing = (
                db.query(ModelPerformance)
                .filter(
                    and_(
                        ModelPerformance.model_version == model_version,
                        ModelPerformance.gameweek == gameweek,
                    )
                )
                .first()
            )

            if existing:
                # Update existing record
                existing.mae = metrics["mae"]
                existing.rmse = metrics["rmse"]
                existing.accuracy = metrics["accuracy"]
                updated += 1
                logger.info(
                    f"  Updated: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
                    f"Accuracy={metrics['accuracy']*100:.1f}% ({metrics['n_predictions']} players)"
                )
            else:
                # Create new record
                perf = ModelPerformance(
                    model_version=model_version,
                    gameweek=gameweek,
                    mae=metrics["mae"],
                    rmse=metrics["rmse"],
                    accuracy=metrics["accuracy"],
                )
                db.add(perf)
                created += 1
                logger.info(
                    f"  Created: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
                    f"Accuracy={metrics['accuracy']*100:.1f}% ({metrics['n_predictions']} players)"
                )

            processed += 1

        # Commit all changes
        db.commit()

        logger.info("=" * 60)
        logger.info("MODEL PERFORMANCE TRACKING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed: {processed} gameweeks")
        logger.info(f"Created: {created} records")
        logger.info(f"Updated: {updated} records")

        return {
            "status": "success",
            "processed": processed,
            "created": created,
            "updated": updated,
        }

    except Exception as e:
        logger.error(f"Error tracking model performance: {str(e)}", exc_info=True)
        db.rollback()
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Track model performance metrics"
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2025-26",
        help="Season identifier (default: 2025-26)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="5.0.0",
        help="Model version string (default: 5.0.0)",
    )

    args = parser.parse_args()

    result = track_model_performance(
        season=args.season, model_version=args.model_version
    )

    if result.get("status") == "error":
        sys.exit(1)
