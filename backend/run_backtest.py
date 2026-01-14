"""
Simple script to run a real backtest with actual data.
Generates a report file in backend/reports/ with real metrics.
"""
import sys
import os
import logging

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.backtest import BacktestEngine
from app.database import SessionLocal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run a backtest with real data from database."""
    logger.info("=" * 60)
    logger.info("REAL BACKTEST SCRIPT")
    logger.info("=" * 60)

    # Initialize BacktestEngine
    logger.info("Initializing BacktestEngine...")
    backtest_engine = BacktestEngine(
        season="2025-26",
        min_train_weeks=5,
        memory_limit_mb=3500
    )

    logger.info("✓ BacktestEngine initialized")

    # Check database has data
    db = SessionLocal()
    try:
        from app.models import PlayerGameweekStats
        data_count = db.query(PlayerGameweekStats).count()
        logger.info(f"Database has {data_count} PlayerGameweekStats records")

        if data_count == 0:
            logger.error("No data in database! Run load_data.py first:")
            logger.error("  docker compose exec backend python load_data.py")
            return

        # Get available seasons
        seasons = db.query(PlayerGameweekStats.season).distinct().all()
        available_seasons = [s[0] for s in seasons]
        logger.info(f"Available seasons: {available_seasons}")

        if "2025-26" not in available_seasons:
            logger.warning("Season '2025-26' not found, using first available season")
            season_to_use = available_seasons[0] if available_seasons else None
            if season_to_use:
                backtest_engine.season = season_to_use
                logger.info(f"Using season: {season_to_use}")
            else:
                logger.error("No seasons found in database!")
                return

        # Get available gameweeks
        gameweeks = db.query(PlayerGameweekStats.gameweek).filter(
            PlayerGameweekStats.season == backtest_engine.season
        ).distinct().order_by(PlayerGameweekStats.gameweek).all()

        if not gameweeks:
            logger.error(f"No gameweeks found for season {backtest_engine.season}")
            return

        available_gws = [gw[0] for gw in gameweeks]
        logger.info(f"Available gameweeks: {available_gws}")

        if len(available_gws) < 10:  # Need at least 10 for meaningful backtest
            logger.warning(f"Only {len(available_gws)} gameweeks available, minimum 10 recommended")
            logger.warning("Backtest may not produce meaningful results")

    finally:
        db.close()

    # Run the actual backtest
    logger.info("")
    logger.info("Running expanding window backtest...")
    logger.info("=" * 60)

    try:
        result = backtest_engine.run_expanding_window_backtest(
            start_gameweek=6,  # Start after we have enough training data
            end_gameweek=None,  # Use all available
            use_solver=True,
            solver_budget=100.0,
            solver_horizon=3
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        # Print summary
        overall_metrics = result.get('overall_metrics', {})
        if overall_metrics:
            logger.info("Overall Metrics:")
            logger.info(f"  RMSE: {overall_metrics.get('rmse', 0):.3f}")
            logger.info(f"  MAE: {overall_metrics.get('mae', 0):.3f}")
            logger.info(f"  Spearman Correlation: {overall_metrics.get('spearman', 0):.3f}")
            logger.info(f"  R²: {overall_metrics.get('r_squared', 0):.3f}")
            logger.info(f"  Weeks Tested: {overall_metrics.get('n_weeks', 0)}")
            logger.info(f"  Cumulative Points: {overall_metrics.get('cumulative_points', 0):.1f}")
            logger.info(f"  Total Transfer Cost: {overall_metrics.get('total_transfer_cost', 0)}")

        logger.info("")
        logger.info("✓ Report saved to backend/reports/ directory")
        logger.info("Check the generated JSON file for detailed results")

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()