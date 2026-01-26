"""
Backtest Runner Script
Runs expanding window backtest to validate ML model performance.
"""

import logging
import sys
import os
from pathlib import Path

# Add backend directory to path for imports
current_file = Path(__file__).resolve()
if os.path.exists("/app"):
    backend_dir = Path("/app")
else:
    backend_dir = current_file.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.backtest import BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run backtest."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FPL Prediction Platform Backtest')
    parser.add_argument(
        '--season',
        type=str,
        default='2025-26',
        help='Season to backtest (default: 2025-26)'
    )
    parser.add_argument(
        '--start-gw',
        type=int,
        default=1,
        help='Starting gameweek (default: 1)'
    )
    parser.add_argument(
        '--end-gw',
        type=int,
        default=None,
        help='Ending gameweek (default: all available)'
    )
    parser.add_argument(
        '--min-train-weeks',
        type=int,
        default=5,
        help='Minimum training weeks required (default: 5)'
    )
    parser.add_argument(
        '--use-solver',
        action='store_true',
        default=True,
        help='Use solver for team optimization (default: True)'
    )
    parser.add_argument(
        '--no-solver',
        dest='use_solver',
        action='store_false',
        help='Disable solver'
    )
    parser.add_argument(
        '--solver-budget',
        type=float,
        default=100.0,
        help='Solver budget (default: 100.0)'
    )
    parser.add_argument(
        '--solver-horizon',
        type=int,
        default=3,
        help='Solver horizon weeks (default: 3)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("FPL Prediction Platform - Backtest Runner")
    logger.info("=" * 60)
    logger.info(f"Season: {args.season}")
    logger.info(f"Start Gameweek: {args.start_gw}")
    logger.info(f"End Gameweek: {args.end_gw or 'All available'}")
    logger.info(f"Min Training Weeks: {args.min_train_weeks}")
    logger.info(f"Use Solver: {args.use_solver}")
    if args.use_solver:
        logger.info(f"Solver Budget: {args.solver_budget}")
        logger.info(f"Solver Horizon: {args.solver_horizon}")
    logger.info("=" * 60)
    
    try:
        # Initialize backtest engine
        backtest_engine = BacktestEngine(
            season=args.season,
            min_train_weeks=args.min_train_weeks
        )
        
        # Run backtest
        results = backtest_engine.run_expanding_window_backtest(
            start_gameweek=args.start_gw,
            end_gameweek=args.end_gw,
            use_solver=args.use_solver,
            solver_budget=args.solver_budget,
            solver_horizon=args.solver_horizon
        )
        
        # Print results
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        
        status = results.get('status', 'unknown')
        logger.info(f"Status: {status}")
        
        # Check if we have any results
        total_weeks_tested = results.get('total_weeks_tested', 0)
        if total_weeks_tested == 0:
            logger.warning("\n⚠️  No gameweeks were tested!")
            logger.warning("Reason: Insufficient training data")
            logger.warning(f"\nThe backtest requires at least {args.min_train_weeks} gameweeks of training data")
            logger.warning("before the first test gameweek.")
            logger.warning("\nTo run a backtest, you need:")
            logger.warning(f"  1. Historical data for gameweeks before {args.start_gw}")
            logger.warning(f"  2. At least {args.min_train_weeks} gameweeks of training data")
            logger.warning("\nExample: To test gameweek 22, you need data for gameweeks 1-21")
            logger.warning("         (or at least gameweeks 17-21 if min_train_weeks=5)")
            logger.warning("\nPlease load historical data first using:")
            logger.warning("  docker compose exec backend python3 scripts/populate_database.py")
            return 0  # Not an error, just no data to test
        
        if status == 'success' or total_weeks_tested > 0:
            # The results dict is the report returned from run_expanding_window_backtest
            # It contains 'overall_metrics', 'weekly_results', etc.
            overall_metrics = results.get('overall_metrics', {})
            
            # Helper function to safely format numeric values
            def format_metric(value, format_str='.4f'):
                if value is None or isinstance(value, str):
                    return value if value is not None else 'N/A'
                try:
                    return format(value, format_str)
                except (ValueError, TypeError):
                    return str(value)
            
            logger.info(f"\nMetrics:")
            logger.info(f"  Total Weeks Tested: {total_weeks_tested}")
            logger.info(f"  RMSE: {format_metric(overall_metrics.get('rmse'), '.4f')}")
            logger.info(f"  Spearman Correlation: {format_metric(overall_metrics.get('spearman'), '.4f')}")
            logger.info(f"  Cumulative Points: {format_metric(overall_metrics.get('cumulative_points'), '.2f')}")
            logger.info(f"  Total Transfer Cost: {format_metric(overall_metrics.get('total_transfer_cost'), '.0f')}")
            logger.info(f"  MAE: {format_metric(overall_metrics.get('mae'), '.4f')}")
            logger.info(f"  R²: {format_metric(overall_metrics.get('r_squared'), '.4f')}")
            
            weekly_results = results.get('weekly_results', [])
            if weekly_results:
                logger.info(f"\nProcessed {len(weekly_results)} gameweeks")
                logger.info(f"First gameweek: {weekly_results[0].get('gameweek', 'N/A')}")
                logger.info(f"Last gameweek: {weekly_results[-1].get('gameweek', 'N/A')}")
            
            # The report is already generated and returned from run_expanding_window_backtest
            # Just use it directly
            report = results
            
            logger.info("\nSaving backtest results to database...")
            model_version = "5.0.0"  # Default model version
            summary_id = backtest_engine.save_report_to_database(report, model_version=model_version)
            
            if summary_id:
                logger.info(f"✅ Backtest results saved to database (Summary ID: {summary_id})")
            else:
                logger.warning("⚠️  Failed to save backtest results to database")
        else:
            error = results.get('error', 'Unknown error')
            logger.error(f"Backtest failed: {error}")
            return 1
        
        logger.info("=" * 60)
        logger.info("Backtest completed successfully!")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
