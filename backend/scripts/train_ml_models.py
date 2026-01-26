"""
ML Model Training Script
Trains all component models (xMins, Attack, Defense) using historical data from database.

Usage:
    docker compose exec backend python3 scripts/train_ml_models.py [gameweek] [season]
    
    - gameweek: Optional. Train using data up to this gameweek (default: current gameweek)
    - season: Optional. Season string (default: "2025-26")
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models import PlayerGameweekStats, Player
from app.services.ml_engine import PLEngine
from app.services.fpl import FPLAPIService
from app.services.component_feature_engineering import ComponentFeatureEngineering
from sqlalchemy import and_

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_training_data(
    db, gameweeks: List[int], season: str = "2025-26"
) -> pd.DataFrame:
    """
    Load training data from database for specified gameweeks.

    Args:
        db: Database session
        gameweeks: List of gameweeks to load
        season: Season string

    Returns:
        DataFrame with training data
    """
    logger.info(f"Loading training data: season='{season}', gameweeks={gameweeks}")

    # Join with Player table to get position and price
    query = (
        db.query(PlayerGameweekStats)
        .join(Player, PlayerGameweekStats.fpl_id == Player.id)
        .filter(
            and_(
                PlayerGameweekStats.season == season,
                PlayerGameweekStats.gameweek.in_(gameweeks),
            )
        )
    )

    data = pd.read_sql(query.statement, db.bind)
    logger.info(f"Loaded {len(data)} rows from database")

    if data.empty:
        logger.warning("No training data found!")
        return pd.DataFrame()

    # Merge with Player table to get position and price
    player_ids = data["fpl_id"].unique().tolist()
    players_query = db.query(Player).filter(Player.id.in_(player_ids))
    players_df = pd.read_sql(players_query.statement, db.bind)

    # Merge player data
    data = data.merge(
        players_df[["id", "position", "price"]],
        left_on="fpl_id",
        right_on="id",
        how="left",
    )

    # Fill missing values
    data["position"] = data["position"].fillna("MID")
    data["price"] = data["price"].fillna(5.0)

    logger.info(f"Training data prepared: {len(data)} rows")
    return data


def prepare_xmins_features(
    training_data: pd.DataFrame, feature_engine: ComponentFeatureEngineering
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels for xMins model.

    Args:
        training_data: Training DataFrame
        feature_engine: ComponentFeatureEngineering instance

    Returns:
        Tuple of (features, labels)
    """
    logger.info("Preparing xMins features...")

    features_df, labels = feature_engine.prepare_xmins_features(training_data)

    if features_df.empty:
        logger.warning("No xMins features prepared!")
        return np.array([]), np.array([])

    # Extract feature columns (exclude fpl_id, gameweek, position)
    feature_columns = [
        col
        for col in features_df.columns
        if col not in ["fpl_id", "gameweek", "position"]
    ]

    X = features_df[feature_columns].values.astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    logger.info(f"xMins features: {X.shape}, labels: {y.shape}")
    return X, y


def prepare_attack_features(
    training_data: pd.DataFrame, feature_engine: ComponentFeatureEngineering
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare features and labels for Attack model (xG and xA).

    Args:
        training_data: Training DataFrame
        feature_engine: ComponentFeatureEngineering instance

    Returns:
        Tuple of (features, xg_labels, xa_labels)
    """
    logger.info("Preparing Attack features...")

    features_df, xg_labels, xa_labels = feature_engine.prepare_attack_features(
        training_data
    )

    if features_df.empty:
        logger.warning("No Attack features prepared!")
        return np.array([]), np.array([]), np.array([])

    # Extract feature columns
    feature_columns = [
        col
        for col in features_df.columns
        if col not in ["fpl_id", "gameweek", "position"]
    ]

    X = features_df[feature_columns].values.astype(np.float32)
    y_xg = np.array(xg_labels, dtype=np.float32)
    y_xa = np.array(xa_labels, dtype=np.float32)

    logger.info(f"Attack features: {X.shape}, xG labels: {y_xg.shape}, xA labels: {y_xa.shape}")
    return X, y_xg, y_xa


async def train_models(
    gameweek: Optional[int] = None, season: str = "2025-26"
) -> dict:
    """
    Train all ML models using historical data.

    Args:
        gameweek: Optional gameweek to train up to (default: current gameweek)
        season: Season string

    Returns:
        Dictionary with training results
    """
    db = SessionLocal()
    fpl_service = None

    try:
        # Initialize services
        logger.info("Initializing services...")
        fpl_service = FPLAPIService()
        ml_engine = PLEngine()
        feature_engine = ComponentFeatureEngineering(db)

        # Get current gameweek if not specified
        if gameweek is None:
            current_gw = await fpl_service.get_current_gameweek()
            if not current_gw:
                logger.error("Could not determine current gameweek")
                return {"status": "error", "message": "Could not determine current gameweek"}
            gameweek = current_gw

        logger.info(f"Training models up to gameweek {gameweek} (season: {season})")

        # Determine training gameweeks (all gameweeks up to specified gameweek)
        # Need at least 5 gameweeks for training
        min_gameweeks = 5
        training_gameweeks = list(range(1, gameweek + 1))

        if len(training_gameweeks) < min_gameweeks:
            logger.warning(
                f"Not enough gameweeks for training (need {min_gameweeks}, have {len(training_gameweeks)})"
            )
            return {
                "status": "error",
                "message": f"Not enough gameweeks for training (need {min_gameweeks}, have {len(training_gameweeks)})",
            }

        # Load training data
        training_data = load_training_data(db, training_gameweeks, season)

        if training_data.empty:
            logger.error("No training data available!")
            return {"status": "error", "message": "No training data available"}

        # Prepare features
        logger.info("=" * 60)
        logger.info("STEP 1: Preparing xMins features...")
        logger.info("=" * 60)
        xmins_features, xmins_labels = prepare_xmins_features(training_data, feature_engine)

        logger.info("=" * 60)
        logger.info("STEP 2: Preparing Attack features...")
        logger.info("=" * 60)
        attack_features, attack_xg_labels, attack_xa_labels = prepare_attack_features(
            training_data, feature_engine
        )

        # Train models
        logger.info("=" * 60)
        logger.info("STEP 3: Training models...")
        logger.info("=" * 60)

        await ml_engine.async_train(
            training_data=training_data,
            xmins_features=xmins_features,
            xmins_labels=xmins_labels,
            attack_features=attack_features,
            attack_xg_labels=attack_xg_labels,
            attack_xa_labels=attack_xa_labels,
        )

        # Save models
        logger.info("=" * 60)
        logger.info("STEP 4: Saving models...")
        logger.info("=" * 60)

        # Create models directory if it doesn't exist
        import os
        models_dir = "/app/models" if os.path.exists("/app") else "models"
        os.makedirs(models_dir, exist_ok=True)

        model_filename = f"plengine_model_gw{gameweek}_{season.replace('-', '_')}.pkl"
        model_path = os.path.join(models_dir, model_filename)

        await ml_engine.async_save_models(model_path)

        logger.info(f"Models saved to: {model_path}")

        # Summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Training gameweeks: {training_gameweeks}")
        logger.info(f"Training samples: {len(training_data)}")
        logger.info(f"xMins samples: {len(xmins_labels)}")
        logger.info(f"Attack samples: {len(attack_xg_labels)}")
        logger.info(f"Models saved to: {model_path}")
        logger.info("=" * 60)

        return {
            "status": "success",
            "gameweek": gameweek,
            "season": season,
            "training_gameweeks": training_gameweeks,
            "training_samples": len(training_data),
            "xmins_samples": len(xmins_labels),
            "attack_samples": len(attack_xg_labels),
            "model_path": model_path,
        }

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        db.close()
        if fpl_service:
            await fpl_service.close()


if __name__ == "__main__":
    # Parse command line arguments
    gameweek = None
    season = "2025-26"

    if len(sys.argv) > 1:
        try:
            gameweek = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid gameweek: {sys.argv[1]}")
            sys.exit(1)

    if len(sys.argv) > 2:
        season = sys.argv[2]

    # Run training
    result = asyncio.run(train_models(gameweek, season))
    
    if result.get("status") == "error":
        sys.exit(1)
