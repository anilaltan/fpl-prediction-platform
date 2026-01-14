"""
Script to train all PLEngine models on database data.
Trains xMins, Attack (xG/xA), and Defense models, then saves to disk.
"""
import sys
import os
import logging
from datetime import datetime
import asyncio

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.ml_engine import PLEngine
from app.services.backtest import BacktestEngine
from app.database import SessionLocal
from app.models import PlayerGameweekStats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_all_models(
    season: str = "2025-26",
    min_gameweek: int = 1,
    max_gameweek: int = None,
    model_save_path: str = None
):
    """
    Train all PLEngine models on database data.
    
    Args:
        season: Season to train on (default: "2025-26")
        min_gameweek: Minimum gameweek to include (default: 1)
        max_gameweek: Maximum gameweek to include (None = all available)
        model_save_path: Path to save trained models (None = auto-generate)
    """
    logger.info("=" * 60)
    logger.info("TRAIN ALL MODELS SCRIPT")
    logger.info("=" * 60)
    
    # Initialize BacktestEngine for data loading and feature preparation
    logger.info("Initializing BacktestEngine...")
    backtest_engine = BacktestEngine(season=season)
    logger.info(f"✓ BacktestEngine initialized (season='{season}')")
    
    # Check database
    db = SessionLocal()
    try:
        # Get available gameweeks
        query = db.query(PlayerGameweekStats.gameweek).filter(
            PlayerGameweekStats.season == season
        ).distinct().order_by(PlayerGameweekStats.gameweek)
        
        all_gameweeks = [gw[0] for gw in query.all()]
        
        if not all_gameweeks:
            logger.error(f"No data found for season '{season}'!")
            logger.error("Please run ETL first: docker compose exec backend python load_data.py")
            return
        
        logger.info(f"Found {len(all_gameweeks)} gameweeks: {all_gameweeks}")
        
        # Determine training gameweeks
        if max_gameweek is None:
            max_gameweek = max(all_gameweeks)
        
        training_gameweeks = [gw for gw in all_gameweeks if min_gameweek <= gw <= max_gameweek]
        
        if not training_gameweeks:
            logger.error(f"No gameweeks in range [{min_gameweek}, {max_gameweek}]!")
            return
        
        logger.info(f"Training on gameweeks: {training_gameweeks} ({len(training_gameweeks)} weeks)")
        
        # Load training data
        logger.info("")
        logger.info("Loading training data from database...")
        training_data = backtest_engine._load_training_data(db, training_gameweeks)
        
        if training_data.empty:
            logger.error("ERROR: Training data is EMPTY!")
            logger.error(f"Check database for season '{season}', gameweeks {training_gameweeks}")
            return
        
        logger.info(f"✓ Loaded {len(training_data)} rows")
        logger.info(f"  Columns: {list(training_data.columns)[:10]}...")
        
        # Verify critical columns
        required_cols = ['fpl_id', 'gameweek', 'minutes', 'position', 'price']
        missing_cols = [col for col in required_cols if col not in training_data.columns]
        if missing_cols:
            logger.error(f"ERROR: Missing required columns: {missing_cols}")
            return
        
        # Prepare features
        logger.info("")
        logger.info("Preparing features...")
        logger.info("  - xMins features...")
        xmins_features, xmins_labels = backtest_engine._prepare_xmins_features(training_data)
        logger.info(f"    ✓ Prepared {len(xmins_features)} xMins samples")
        
        logger.info("  - Attack features (xG/xA)...")
        attack_features, attack_xg_labels, attack_xa_labels = backtest_engine._prepare_attack_features(training_data)
        logger.info(f"    ✓ Prepared {len(attack_features)} Attack samples")
        
        # Initialize PLEngine
        logger.info("")
        logger.info("Initializing PLEngine...")
        plengine = PLEngine()
        logger.info("✓ PLEngine initialized")
        
        # Train models
        logger.info("")
        logger.info("Training models...")
        logger.info("=" * 60)
        
        plengine.train(
            training_data=training_data,
            xmins_features=xmins_features,
            xmins_labels=xmins_labels,
            attack_features=attack_features,
            attack_xg_labels=attack_xg_labels,
            attack_xa_labels=attack_xa_labels
        )
        
        logger.info("=" * 60)
        logger.info("✓ All models trained successfully!")
        
        # Verify models are trained
        logger.info("")
        logger.info("Verifying model training status...")
        logger.info(f"  xMins model trained: {plengine.xmins_model.is_trained}")
        logger.info(f"  Attack xG model trained: {plengine.attack_model.xg_trained}")
        logger.info(f"  Attack xA model trained: {plengine.attack_model.xa_trained}")
        logger.info(f"  Defense model fitted: {plengine.defense_model.is_fitted}")
        
        # Save models
        if model_save_path is None:
            # Auto-generate path
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(backend_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = os.path.join(models_dir, f"plengine_{season}_{timestamp_str}.pkl")
        
        logger.info("")
        logger.info(f"Saving models to: {model_save_path}")
        
        # Save synchronously (async_save_models requires event loop)
        os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
        
        import pickle
        model_data = {
            'xmins_model': plengine.xmins_model.model if plengine.xmins_model.is_trained else None,
            'xmins_scaler': plengine.xmins_model.scaler,
            'attack_xg_model': plengine.attack_model.xg_model if plengine.attack_model.xg_trained else None,
            'attack_xa_model': plengine.attack_model.xa_model if plengine.attack_model.xa_trained else None,
            'attack_scaler': plengine.attack_model.scaler,
            'defense_model': plengine.defense_model,
            'version': plengine.model_version,
            'season': season,
            'training_gameweeks': training_gameweeks,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Models saved successfully!")
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model file: {model_save_path}")
        logger.info(f"Season: {season}")
        logger.info(f"Training gameweeks: {training_gameweeks}")
        logger.info(f"Training samples: {len(training_data)} rows")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        db.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all PLEngine models')
    parser.add_argument('--season', type=str, default='2025-26', help='Season to train on')
    parser.add_argument('--min-gw', type=int, default=1, help='Minimum gameweek')
    parser.add_argument('--max-gw', type=int, default=None, help='Maximum gameweek (None = all)')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save models')
    
    args = parser.parse_args()
    
    train_all_models(
        season=args.season,
        min_gameweek=args.min_gw,
        max_gameweek=args.max_gw,
        model_save_path=args.save_path
    )


if __name__ == "__main__":
    main()
