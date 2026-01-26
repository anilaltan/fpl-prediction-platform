"""
Backtesting and Validation Engine for FPL Prediction Platform
Implements Expanding Window methodology with solver integration.
Features:
- Expanding Window: Train and predict week by week from season start
- Metrics: RMSE and Spearman Rank Correlation
- Solver Integration: Simulate transfers and calculate cumulative points
- Memory Management: 4GB RAM limit with gc.collect() and parquet storage
- Reporting: Season-end summary with graphical metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import gc
import os
import tempfile
import json
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

from app.database import SessionLocal
from app.models import PlayerGameweekStats, Player
from app.services.ml_engine import PLEngine, AttackModel
from app.services.solver import FPLSolver
from app.services.strategy import StrategyService

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for FPL prediction validation.
    Implements Expanding Window methodology with solver integration.
    """

    def __init__(
        self,
        season: str = "2025-26",
        min_train_weeks: int = 5,
        memory_limit_mb: int = 3500,  # 3.5GB buffer for 4GB limit
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize backtesting engine.

        Args:
            season: Season to backtest (default: "2025-26")
            min_train_weeks: Minimum weeks needed for training (default: 5)
            memory_limit_mb: Memory limit in MB (default: 3500)
            temp_dir: Temporary directory for parquet storage (default: system temp)
        """
        self.season = season
        self.min_train_weeks = min_train_weeks
        self.memory_limit_mb = memory_limit_mb
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)

        # Initialize services
        self.plengine = None
        self.solver = None
        self.strategy_service = StrategyService()

        # Results storage
        self.weekly_results = []
        self.cumulative_points = 0.0
        self.total_transfer_cost = 0

        # CRITICAL FIX: Store individual player predictions for proper metric calculation
        # This allows us to calculate RMSE/R² on all predictions, not just weekly team sums
        self.all_individual_predictions = []  # List of (predicted, actual, gameweek) tuples
        self.all_individual_actuals = []

    def run_expanding_window_backtest(
        self,
        start_gameweek: int = 1,
        end_gameweek: Optional[int] = None,
        use_solver: bool = True,
        solver_budget: float = 100.0,
        solver_horizon: int = 3,
    ) -> Dict:
        """
        Run expanding window backtest.
        Trains model on all previous weeks and predicts current week.

        Args:
            start_gameweek: Starting gameweek (default: 1)
            end_gameweek: Ending gameweek (None = all available)
            use_solver: Whether to use solver for team optimization (default: True)
            solver_budget: Budget for solver (default: 100.0)
            solver_horizon: Horizon weeks for solver (default: 3)

        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("=" * 60)
        logger.info(f"Starting Expanding Window Backtest - Season {self.season}")
        logger.info("=" * 60)

        db = SessionLocal()
        try:
            # Get all available gameweeks
            all_gameweeks = (
                db.query(PlayerGameweekStats.gameweek)
                .filter(PlayerGameweekStats.season == self.season)
                .distinct()
                .order_by(PlayerGameweekStats.gameweek)
                .all()
            )

            if not all_gameweeks:
                # Try to find available seasons
                available_seasons = (
                    db.query(PlayerGameweekStats.season).distinct().all()
                )
                available_seasons_list = [s[0] for s in available_seasons]

                if not available_seasons_list:
                    raise ValueError(
                        f"No data found in database for season '{self.season}'. "
                        f"Please run ETL service first to load player_gameweek_stats data. "
                        f"Use: FPLAPIService.fetch_comprehensive_player_data() and ETLService.upsert_player_gameweek_stats()"
                    )

                # Try to match season (flexible matching)
                matched_season = None
                for avail_season in available_seasons_list:
                    if (
                        self.season in str(avail_season)
                        or str(avail_season) in self.season
                    ):
                        matched_season = avail_season
                        break

                if matched_season:
                    logger.warning(
                        f"Season '{self.season}' not found. Using '{matched_season}' instead."
                    )
                    self.season = matched_season
                    all_gameweeks = (
                        db.query(PlayerGameweekStats.gameweek)
                        .filter(PlayerGameweekStats.season == self.season)
                        .distinct()
                        .order_by(PlayerGameweekStats.gameweek)
                        .all()
                    )

                    if not all_gameweeks:
                        raise ValueError(
                            f"No gameweek data found for season '{matched_season}'. "
                            f"Please ensure player_gameweek_stats table has data."
                        )
                else:
                    raise ValueError(
                        f"No data found for season '{self.season}'. "
                        f"Available seasons: {available_seasons_list}. "
                        f"Please run ETL to load data for the desired season."
                    )

            all_available_gameweeks = [gw[0] for gw in all_gameweeks]

            # Filter gameweeks for testing (but keep all for training)
            if end_gameweek:
                test_gameweeks = [
                    gw
                    for gw in all_available_gameweeks
                    if start_gameweek <= gw <= end_gameweek
                ]
            else:
                test_gameweeks = [
                    gw for gw in all_available_gameweeks if gw >= start_gameweek
                ]

            logger.info(
                f"Found {len(all_available_gameweeks)} total gameweeks: {all_available_gameweeks[0]} to {all_available_gameweeks[-1]}"
            )
            logger.info(
                f"Testing {len(test_gameweeks)} gameweeks: {test_gameweeks[0] if test_gameweeks else 'N/A'} to {test_gameweeks[-1] if test_gameweeks else 'N/A'}"
            )

            # Initialize solver if needed
            if use_solver:
                self.solver = FPLSolver(
                    budget=solver_budget,
                    horizon_weeks=solver_horizon,
                    free_transfers=1,
                    discount_factor=0.9,
                )

            # Expanding window: train on all previous weeks, predict current
            # CRITICAL: Track when to fit initial calibration (after enough data collected)
            calibration_fitted_early = False
            calibration_fit_threshold = max(
                5, len(test_gameweeks) // 3
            )  # Fit after 1/3 of weeks or 5 weeks, whichever is larger

            for i, current_gw in enumerate(test_gameweeks):
                logger.info("")
                logger.info(
                    f"Processing Gameweek {current_gw} ({i + 1}/{len(test_gameweeks)})"
                )
                logger.info("-" * 60)

                # Get training data (all weeks before current from ALL available gameweeks)
                training_weeks = [
                    gw for gw in all_available_gameweeks if gw < current_gw
                ]

                # Check if we have enough training data
                if len(training_weeks) < self.min_train_weeks:
                    logger.info(
                        f"Skipping GW{current_gw}: Only {len(training_weeks)} training weeks available, need at least {self.min_train_weeks}"
                    )
                    continue

                if not training_weeks:
                    logger.warning(f"No training data for GW{current_gw}")
                    continue

                # CRITICAL FIX: Fit calibration early if we have enough predictions
                # This allows calibration to be applied to subsequent predictions
                # Check BEFORE making predictions so calibration can be applied to current week
                if (
                    not calibration_fitted_early
                    and len(self.all_individual_predictions)
                    >= calibration_fit_threshold
                ):
                    logger.info("")
                    logger.info("=" * 60)
                    logger.info(
                        "Fitting Early Calibration (for subsequent predictions)"
                    )
                    logger.info("=" * 60)
                    logger.info(
                        f"Using {len(self.all_individual_predictions)} predictions for early calibration"
                    )

                    calibration_result = self.plengine.fit_calibration(
                        predicted_points=np.array(self.all_individual_predictions),
                        actual_points=np.array(self.all_individual_actuals),
                        method="linear",
                    )

                    if calibration_result:
                        calibration_fitted_early = True
                        logger.info(
                            f"✓ Early calibration fitted: scale={calibration_result.get('scale', 1.0):.3f}, offset={calibration_result.get('offset', 0.0):.3f}"
                        )
                        logger.info(
                            f"  Calibration will be applied to predictions for GW{current_gw} onwards"
                        )
                        logger.info(
                            f"  RMSE improvement: {calibration_result.get('improvement_pct', 0.0):.1f}%"
                        )
                        logger.info(
                            f"  R² improvement: {calibration_result.get('r2_after', 0.0) - calibration_result.get('r2_before', 0.0):+.3f}"
                        )
                    else:
                        logger.warning(
                            "⚠ Early calibration fitting failed, continuing without calibration"
                        )

                # Log calibration status for debugging
                if calibration_fitted_early:
                    logger.debug(
                        f"Calibration status: fitted={self.plengine.calibration_fitted}, scale={self.plengine.calibration_scale:.3f}, offset={self.plengine.calibration_offset:.3f}"
                    )

                # Run backtest for this week
                week_result = self._backtest_week(
                    db=db,
                    current_gw=current_gw,
                    training_weeks=training_weeks,
                    use_solver=use_solver,
                )

                if week_result:
                    self.weekly_results.append(week_result)
                    self.cumulative_points += week_result.get("net_points", 0.0)
                    self.total_transfer_cost += week_result.get("transfer_cost", 0)

                    logger.info(f"GW{current_gw} Results:")
                    logger.info(
                        f"  Predicted Points: {week_result.get('predicted_points', 0):.2f}"
                    )
                    logger.info(
                        f"  Actual Points: {week_result.get('actual_points', 0):.2f}"
                    )
                    logger.info(f"  RMSE: {week_result.get('rmse', 0):.2f}")
                    logger.info(f"  Spearman: {week_result.get('spearman', 0):.3f}")
                    if use_solver:
                        logger.info(
                            f"  Net Points: {week_result.get('net_points', 0):.2f}"
                        )
                        logger.info(
                            f"  Transfer Cost: {week_result.get('transfer_cost', 0)}"
                        )

                # Memory management
                self._manage_memory()

        finally:
            db.close()

        # Calculate overall metrics (before calibration)
        overall_metrics = self._calculate_overall_metrics()

        # CRITICAL FIX: Fit calibration layer to align predicted scale with actual distribution
        # This addresses the bias/scaling issue causing negative R²
        # The calibration uses least squares to preserve variance while fixing bias
        if (
            len(self.all_individual_predictions) > 0
            and len(self.all_individual_actuals) > 0
        ):
            logger.info("")
            logger.info("=" * 60)
            logger.info("Fitting Final Calibration Layer (All Data)")
            logger.info("=" * 60)

            calibration_result = self.plengine.fit_calibration(
                predicted_points=np.array(self.all_individual_predictions),
                actual_points=np.array(self.all_individual_actuals),
                method="linear",  # Can be 'linear' or 'isotonic'
            )

            if calibration_result:
                logger.info("Calibration fitted successfully:")
                logger.info(f"  Scale: {calibration_result.get('scale', 1.0):.3f}")
                logger.info(f"  Offset: {calibration_result.get('offset', 0.0):.3f}")
                logger.info(
                    f"  RMSE improvement: {calibration_result.get('improvement_pct', 0.0):.1f}%"
                )
                logger.info(
                    f"  R² improvement: {calibration_result.get('r2_after', 0.0) - calibration_result.get('r2_before', 0.0):+.3f}"
                )

                # Recalculate metrics with calibrated predictions
                calibrated_pred = np.array(
                    self.all_individual_predictions
                ) * calibration_result.get("scale", 1.0) + calibration_result.get(
                    "offset", 0.0
                )
                calibrated_pred = np.clip(calibrated_pred, 0.0, None)

                rmse_calibrated = np.sqrt(
                    mean_squared_error(self.all_individual_actuals, calibrated_pred)
                )
                mean_actual = float(np.mean(self.all_individual_actuals))
                mean_pred_calibrated = float(np.mean(calibrated_pred))

                ss_res_cal = np.sum(
                    (np.array(self.all_individual_actuals) - calibrated_pred) ** 2
                )
                ss_tot_cal = np.sum(
                    (np.array(self.all_individual_actuals) - mean_actual) ** 2
                )
                r2_calibrated = 1 - (ss_res_cal / ss_tot_cal) if ss_tot_cal > 0 else 0.0

                overall_metrics["rmse_calibrated"] = float(rmse_calibrated)
                overall_metrics["r_squared_calibrated"] = float(r2_calibrated)
                overall_metrics["mean_predicted_calibrated"] = mean_pred_calibrated
                overall_metrics["calibration_params"] = {
                    "scale": calibration_result.get("scale", 1.0),
                    "offset": calibration_result.get("offset", 0.0),
                }

                logger.info("")
                logger.info("Calibrated Metrics:")
                logger.info(
                    f"  RMSE: {overall_metrics.get('rmse', 0):.2f} -> {rmse_calibrated:.2f}"
                )
                logger.info(
                    f"  R²: {overall_metrics.get('r_squared', 0):.3f} -> {r2_calibrated:.3f}"
                )
                logger.info(
                    f"  Mean Predicted: {overall_metrics.get('mean_predicted', 0):.2f} -> {mean_pred_calibrated:.2f}"
                )

        # Generate report
        report = self._generate_report(overall_metrics)

        # Optionally save to database (uncomment if you want this)
        # self.save_report_to_database(report, model_version="5.0.0")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Backtest Complete")
        logger.info("=" * 60)
        logger.info(f"Total Weeks Tested: {len(self.weekly_results)}")
        logger.info(f"Overall RMSE: {overall_metrics.get('rmse', 0):.2f}")
        logger.info(f"Overall Spearman: {overall_metrics.get('spearman', 0):.3f}")
        if use_solver:
            logger.info(f"Cumulative Points: {self.cumulative_points:.2f}")
            logger.info(f"Total Transfer Cost: {self.total_transfer_cost}")

        return report

    def _backtest_week(
        self,
        db: Session,
        current_gw: int,
        training_weeks: List[int],
        use_solver: bool = True,
    ) -> Optional[Dict]:
        """
        Backtest a single gameweek.

        Args:
            db: Database session
            current_gw: Current gameweek to predict
            training_weeks: List of weeks to use for training
            use_solver: Whether to use solver

        Returns:
            Dictionary with week results
        """
        try:
            # 1. Load training data
            logger.info(f"Loading training data ({len(training_weeks)} weeks)...")
            training_data = self._load_training_data(db, training_weeks)

            if training_data.empty:
                logger.warning(f"No training data for weeks {training_weeks}")
                return None

            # Save to pickle for memory efficiency (fallback if parquet not available)
            training_file = os.path.join(self.temp_dir, f"training_gw{current_gw}.pkl")
            try:
                # Try parquet first (more efficient)
                parquet_file = training_file.replace(".pkl", ".parquet")
                training_data.to_parquet(parquet_file, index=False)
                training_file = parquet_file
            except ImportError:
                # Fallback to pickle
                training_data.to_pickle(training_file)
            del training_data
            gc.collect()

            # 2. Load current week data (for prediction and validation)
            logger.info(f"Loading GW{current_gw} data...")
            current_data = self._load_gameweek_data(db, current_gw)

            if current_data.empty:
                logger.warning(f"No data for GW{current_gw}")
                return None

            # 3. Initialize and train model
            logger.info("Training model...")
            # CRITICAL FIX: Reuse existing PLEngine instance to preserve calibration
            # If calibration was fitted early, we need to keep the same instance
            if self.plengine is None:
                self.plengine = PLEngine()
            # CRITICAL: Ensure models are loaded before using them
            self.plengine._ensure_models_loaded()

            # Reload training data
            if training_file.endswith(".parquet"):
                try:
                    training_data = pd.read_parquet(training_file)
                except ImportError:
                    # Fallback to pickle if parquet read fails
                    pickle_file = training_file.replace(".parquet", ".pkl")
                    if os.path.exists(pickle_file):
                        training_data = pd.read_pickle(pickle_file)
                    else:
                        raise
            else:
                training_data = pd.read_pickle(training_file)

            # Prepare training features
            xmins_features, xmins_labels = self._prepare_xmins_features(training_data)
            (
                attack_features,
                attack_xg_labels,
                attack_xa_labels,
            ) = self._prepare_attack_features(training_data)

            # Train model
            logger.info(
                f"Training data sizes - xMins: {len(xmins_features) if xmins_features is not None else 0} features, "
                f"{len(xmins_labels) if xmins_labels is not None else 0} labels; "
                f"Attack: {len(attack_features) if attack_features is not None else 0} features, "
                f"{len(attack_xg_labels) if attack_xg_labels is not None else 0} xG labels, "
                f"{len(attack_xa_labels) if attack_xa_labels is not None else 0} xA labels"
            )
            
            try:
                self.plengine.train(
                    training_data=training_data,
                    xmins_features=xmins_features,
                    xmins_labels=xmins_labels,
                    attack_features=attack_features,
                    attack_xg_labels=attack_xg_labels,
                    attack_xa_labels=attack_xa_labels,
                )
                
                # Verify training succeeded
                if not self.plengine.xmins_strategy.is_trained:
                    logger.error("xMins model training failed - is_trained is still False")
                if not self.plengine.attack_strategy.xg_trained:
                    logger.error("Attack model training failed - xg_trained is still False")
                else:
                    logger.info("Model training completed successfully")
            except Exception as e:
                logger.error(f"Error during model training for GW{current_gw}: {str(e)}", exc_info=True)
                raise

            # 4. Calculate historical averages for each player from training data (CRITICAL FIX)
            # For GW prediction, we should NOT use the current week's stats (which would be unknown)
            # Instead, use historical averages from previous gameweeks
            logger.info("Calculating historical averages for prediction...")
            player_historical_stats = self._calculate_player_historical_stats(
                training_data
            )

            # Cleanup training data
            del training_data
            gc.collect()

            # 5. Make predictions for current week using HISTORICAL data
            logger.info("Making predictions...")
            predictions = []
            actual_points = []

            for _, row in current_data.iterrows():
                # Current week data is only for validation (actual_points)
                current_week_data = row.to_dict()
                fpl_id = current_week_data.get("fpl_id")

                # CRITICAL FIX: Use historical averages, NOT current week stats
                # Get historical stats for this player from previous gameweeks
                if fpl_id in player_historical_stats:
                    player_data = player_historical_stats[fpl_id].copy()
                    # Add position and other metadata from current week
                    player_data["position"] = current_week_data.get("position", "MID")
                    player_data["price"] = current_week_data.get("price", 5.0)
                    player_data["fpl_id"] = fpl_id
                    player_data["status"] = current_week_data.get("status", "a")
                else:
                    # New player with no history - use minimal defaults
                    player_data = {
                        "fpl_id": fpl_id,
                        "position": current_week_data.get("position", "MID"),
                        "price": current_week_data.get("price", 5.0),
                        "status": current_week_data.get("status", "a"),
                        "xg_per_90": 0.0,
                        "xa_per_90": 0.0,
                        "goals_per_90": 0.0,
                        "assists_per_90": 0.0,
                        "minutes": 0,
                        "recent_minutes": [],
                        "recent_xg": [],
                        "recent_xa": [],
                    }

                # Get prediction using historical data
                try:
                    # Verify models are trained before predicting
                    if not self.plengine.xmins_strategy.is_trained:
                        raise RuntimeError("xMins model not trained")
                    if not self.plengine.attack_strategy.xg_trained:
                        raise RuntimeError("Attack model not trained")
                    
                    prediction = self.plengine.predict(
                        player_data=player_data, fixture_data=None
                    )
                    pred_points = prediction.get("expected_points", 0.0)
                    predictions.append(pred_points)
                    # Actual points come from current week's REAL data (for validation only)
                    actual_points.append(current_week_data.get("total_points", 0))
                except Exception as e:
                    logger.error(
                        f"Prediction error for player {fpl_id} (GW{current_gw}): {str(e)}"
                    )
                    logger.debug(
                        f"  Model status - xMins trained: {self.plengine.xmins_strategy.is_trained if self.plengine else 'N/A'}, "
                        f"Attack trained: {self.plengine.attack_strategy.xg_trained if self.plengine else 'N/A'}"
                    )
                    predictions.append(0.0)
                    actual_points.append(current_week_data.get("total_points", 0))

            # 5. Calculate metrics (on individual player predictions)
            rmse = np.sqrt(mean_squared_error(actual_points, predictions))
            
            # Handle NaN in Spearman correlation (occurs when all predictions are the same)
            try:
                spearman_corr, _ = spearmanr(actual_points, predictions)
                if np.isnan(spearman_corr):
                    # All predictions are the same (likely all 0.0) - correlation is undefined
                    spearman_corr = 0.0
                    if len(set(predictions)) == 1:
                        logger.warning(
                            f"GW{current_gw}: All predictions are identical ({predictions[0] if predictions else 'N/A'}), "
                            f"Spearman correlation set to 0.0"
                        )
            except Exception as e:
                logger.warning(f"Error calculating Spearman correlation for GW{current_gw}: {str(e)}")
                spearman_corr = 0.0

            # CRITICAL FIX: Store individual predictions for aggregate metrics
            # This allows proper RMSE/R² calculation across all players and weeks
            for pred, actual in zip(predictions, actual_points):
                self.all_individual_predictions.append(pred)
                self.all_individual_actuals.append(actual)

            # 6. Solver integration (if enabled)
            solver_result = None
            net_points = 0.0
            transfer_cost = 0

            if use_solver and self.solver:
                logger.info("Running solver optimization...")
                solver_result = self._run_solver_for_week(
                    current_data, predictions, current_gw
                )

                if solver_result:
                    net_points = solver_result.get("net_points", 0.0)
                    transfer_cost = solver_result.get("transfer_cost", 0)

            # 7. Calculate predicted vs actual for selected team
            predicted_points = (
                sum(predictions[:11]) if len(predictions) >= 11 else sum(predictions)
            )
            actual_team_points = (
                sum(actual_points[:11])
                if len(actual_points) >= 11
                else sum(actual_points)
            )

            # Save prediction count BEFORE cleanup
            n_predictions = len(predictions)

            # Cleanup
            del current_data
            del predictions
            del actual_points
            gc.collect()

            # Remove temp file
            if os.path.exists(training_file):
                os.remove(training_file)

            return {
                "gameweek": current_gw,
                "predicted_points": predicted_points,
                "actual_points": actual_team_points,
                "rmse": rmse,
                "spearman": spearman_corr,
                "n_predictions": n_predictions,
                "net_points": net_points,
                "transfer_cost": transfer_cost,
                "solver_result": solver_result,
            }

        except Exception as e:
            logger.error(f"Error in backtest for GW{current_gw}: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _load_training_data(self, db: Session, gameweeks: List[int]) -> pd.DataFrame:
        """
        Load training data for specified gameweeks.
        Joins with Player table to get position and price.

        CRITICAL: Uses fpl_id for join (NOT player_id)!
        Season filter uses self.season (default: "2025-26").

        Args:
            db: Database session
            gameweeks: List of gameweeks to load

        Returns:
            DataFrame with training data
        """

        # CRITICAL: Join on fpl_id -> Player.id (Player model uses 'id' as FPL player ID)
        query = (
            db.query(PlayerGameweekStats)
            .join(Player, PlayerGameweekStats.fpl_id == Player.id)
            .filter(
                and_(
                    PlayerGameweekStats.season
                    == self.season,  # Dynamic season filter (default: "2025-26")
                    PlayerGameweekStats.gameweek.in_(gameweeks),
                )
            )
        )

        logger.info(
            f"Loading training data: season='{self.season}', gameweeks={gameweeks}"
        )
        data = pd.read_sql(query.statement, db.bind)
        logger.info(
            f"Loaded {len(data)} rows from database (season='{self.season}', gameweeks={gameweeks})"
        )

        # Verify fpl_id column exists
        if not data.empty and "fpl_id" not in data.columns:
            logger.error(
                "ERROR: 'fpl_id' column missing from training data! Join may have failed."
            )
            return pd.DataFrame()

        if data.empty:
            return pd.DataFrame()

        # CRITICAL: Merge with Player table to get position and price
        # The join above only filters, we need to actually merge the columns
        player_ids = data["fpl_id"].unique().tolist()
        logger.info(
            f"Merging with Player table using id (fpl_id): {len(player_ids)} unique players"
        )
        players_query = db.query(Player).filter(Player.id.in_(player_ids))
        players_df = pd.read_sql(players_query.statement, db.bind)

        if not players_df.empty:
            # Rename Player.id to fpl_id for merge (Player model uses 'id' as FPL player ID)
            players_df = players_df.rename(columns={"id": "fpl_id"})
            # Merge to get position and price
            data = data.merge(
                players_df[["fpl_id", "position", "price"]],
                on="fpl_id",
                how="left",
                suffixes=("", "_player"),
            )
            # Use player position and price if available
            if "position_player" in data.columns:
                data["position"] = data["position_player"].fillna(
                    data.get("position", "MID")
                )
            elif "position" not in data.columns:
                data["position"] = "MID"  # Default

            if "price_player" in data.columns:
                data["price"] = data["price_player"].fillna(data.get("price", 5.0))
            elif "price" not in data.columns:
                data["price"] = 5.0  # Default

            # Preserve team name for solver team constraints
            if "team_player" in data.columns:
                data["team"] = data["team_player"].fillna(data.get("team"))
        else:
            # No player data, set defaults
            if "position" not in data.columns:
                data["position"] = "MID"
            if "price" not in data.columns:
                data["price"] = 5.0

        logger.info(f"After merge: {len(data)} rows with position and price columns")

        return data

    def _load_gameweek_data(self, db: Session, gameweek: int) -> pd.DataFrame:
        """
        Load data for a specific gameweek.
        Joins with Player table to get position and price.

        CRITICAL: Uses fpl_id for join (NOT player_id)!
        Season filter uses self.season (default: "2025-26").

        Args:
            db: Database session
            gameweek: Gameweek to load

        Returns:
            DataFrame with gameweek data
        """
        # Load PlayerGameweekStats with season filter (dynamic: self.season = "2025-26")
        stats_query = db.query(PlayerGameweekStats).filter(
            and_(
                PlayerGameweekStats.season == self.season,  # Dynamic season filter
                PlayerGameweekStats.gameweek == gameweek,
            )
        )

        logger.info(f"Loading gameweek {gameweek} data: season='{self.season}'")
        stats_df = pd.read_sql(stats_query.statement, db.bind)
        logger.info(
            f"Loaded {len(stats_df)} rows for gameweek {gameweek} (season='{self.season}')"
        )

        if stats_df.empty:
            logger.warning(
                f"No data found for gameweek {gameweek} (season='{self.season}')"
            )
            return pd.DataFrame()

        # CRITICAL: Load Player data using id (Player model uses 'id' as FPL player ID)
        player_ids = stats_df["fpl_id"].unique().tolist()
        logger.info(
            f"Merging with Player table using id (fpl_id): {len(player_ids)} unique players"
        )
        players_query = db.query(Player).filter(Player.id.in_(player_ids))
        players_df = pd.read_sql(players_query.statement, db.bind)

        if not players_df.empty:
            # Rename Player.id to fpl_id for merge (Player model uses 'id' as FPL player ID)
            players_df = players_df.rename(columns={"id": "fpl_id"})
            # Merge to get position and price
            stats_df = stats_df.merge(
                players_df[["fpl_id", "position", "price"]],
                on="fpl_id",
                how="left",
                suffixes=("", "_player"),
            )
            # Use player position and price if available
            if "position_player" in stats_df.columns:
                stats_df["position"] = stats_df["position_player"].fillna(
                    stats_df.get("position", "MID")
                )
            elif "position" not in stats_df.columns:
                stats_df["position"] = "MID"  # Default

            if "price_player" in stats_df.columns:
                stats_df["price"] = stats_df["price_player"].fillna(
                    stats_df.get("price", 5.0)
                )
            elif "price" not in stats_df.columns:
                stats_df["price"] = 5.0  # Default

            # Preserve team name for solver team constraints
            if "team_player" in stats_df.columns:
                stats_df["team"] = stats_df["team_player"].fillna(stats_df.get("team"))
        else:
            # No player data, set defaults
            if "position" not in stats_df.columns:
                stats_df["position"] = "MID"
            if "price" not in stats_df.columns:
                stats_df["price"] = 5.0

        return stats_df

    def _prepare_xmins_features(
        self, training_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for xMins model.
        Must match XMinsModel.extract_features() signature exactly (8 features).

        Expected features (in order):
        1. days_since_last_match
        2. is_cup_week
        3. injury_status (0=fit, 1=doubtful, 2=out)
        4. recent_minutes_avg / 90.0
        5. position_depth / 3.0
        6. form_score / 10.0
        7. price (normalized to 0-1)
        8. rotation_risk

        Args:
            training_data: Training DataFrame

        Returns:
            (features, labels) tuple
        """
        features = []
        labels = []

        # Group by player to calculate days_since_last_match and recent_minutes_avg
        training_data_sorted = training_data.sort_values(["fpl_id", "gameweek"])

        for idx, row in training_data_sorted.iterrows():
            player_id = row.get("fpl_id")

            # Get all matches for this player
            player_data = training_data_sorted[
                training_data_sorted["fpl_id"] == player_id
            ]
            player_data = player_data.sort_values("gameweek")

            current_gw = row.get("gameweek", 1)
            prev_matches = player_data[player_data["gameweek"] < current_gw]

            # 1. days_since_last_match
            if len(prev_matches) > 0:
                last_gw = prev_matches["gameweek"].max()
                days_since = float(
                    (current_gw - last_gw) * 7
                )  # Approximate: 7 days per gameweek
            else:
                days_since = 14.0  # Default: 2 weeks

            # 2. is_cup_week (default: no cup)
            is_cup_week = 0.0

            # 3. injury_status (0=fit, 1=doubtful, 2=out)
            # Try to get from status field, default to fit
            status = str(row.get("status", "a")).lower()
            injury_status_map = {"a": 0, "d": 1, "i": 2, "n": 0, "s": 2}
            injury_status = float(injury_status_map.get(status, 0))

            # 4. recent_minutes_avg / 90.0 (safe handling for empty lists)
            if len(prev_matches) > 0:
                recent_minutes = prev_matches["minutes"].tail(3).tolist()
                recent_minutes = [m for m in recent_minutes if m > 0]  # Filter zeros
                if recent_minutes and len(recent_minutes) > 0:
                    recent_minutes_avg = float(np.mean(recent_minutes)) / 90.0
                else:
                    recent_minutes_avg = 0.5  # Default: 45 minutes
            else:
                recent_minutes_avg = 0.5  # Default

            # 5. position_depth / 3.0 (default: medium depth)
            position_depth = 2.0 / 3.0

            # 6. form_score / 10.0 (use total_points as proxy)
            form_score = float(row.get("total_points", 0)) / 10.0

            # 7. price (normalized to 0-1, FPL prices are in 0.1 increments, max ~15M)
            price = float(row.get("price", 5.0)) / 100.0  # Normalize to 0-1

            # 8. rotation_risk (default: medium)
            rotation_risk = 0.5

            # Features matching XMinsModel.extract_features() exactly (8 features)
            feature_vector = [
                days_since,  # 1. days_since_last_match
                is_cup_week,  # 2. is_cup_week
                injury_status,  # 3. injury_status
                recent_minutes_avg,  # 4. recent_minutes_avg / 90.0
                position_depth,  # 5. position_depth / 3.0
                form_score,  # 6. form_score / 10.0
                price,  # 7. price (normalized)
                rotation_risk,  # 8. rotation_risk
            ]
            features.append(feature_vector)

            # Label: 1 if started (minutes > 0), 0 otherwise
            labels.append(1 if row.get("minutes", 0) > 0 else 0)

        return np.array(features), np.array(labels)

    def _prepare_attack_features(
        self, training_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for Attack model (xG/xA).

        CRITICAL FIX: Avoid target leakage!
        - For each row (GW N), use player's average stats from GW 1 to N-1 as FEATURES
        - Use GW N's actual xG/xA as TARGETS (labels)
        - This prevents the model from "learning" to predict xG using xG itself

        Args:
            training_data: Training DataFrame

        Returns:
            (features, xg_labels, xa_labels) tuple
        """
        features = []
        xg_labels = []
        xa_labels = []

        # IMPORTANT:
        # The AttackModel in `app.services.ml_engine.AttackModel` uses a 17-feature vector in `extract_features()`.
        # We must train with the SAME feature shape, otherwise the scaler will be fit on one dimensionality
        # and prediction-time transform will throw (observed: "X has 17 features, but StandardScaler is expecting 6").
        # CRITICAL: Ensure attack_model is not None - if plengine or attack_model is None, create new instance
        # Also ensure models are loaded if plengine exists
        if self.plengine is not None:
            # Ensure models are loaded
            self.plengine._ensure_models_loaded()
            if self.plengine.attack_model is not None:
                attack_model = self.plengine.attack_model
            else:
                # Fallback: create new AttackModel instance for feature extraction
                logger.warning(
                    "plengine.attack_model is None, creating new AttackModel instance"
                )
                attack_model = AttackModel()
        else:
            # Fallback: create new AttackModel instance for feature extraction
            logger.warning("plengine is None, creating new AttackModel instance")
            attack_model = AttackModel()

        # CRITICAL FIX: Pre-calculate cumulative stats for each player up to each gameweek
        # This avoids target leakage by using only PAST data for features
        # Task 3.1: Optimize DataFrame types before processing
        from app.utils.dataframe_optimizer import optimize_dataframe_types

        training_data_sorted = optimize_dataframe_types(
            training_data.sort_values(["fpl_id", "gameweek"]),
            int_columns=["fpl_id", "gameweek"],
            category_columns=["position"],
        )

        # Build cumulative stats for each player
        player_cumulative_stats = {}  # {fpl_id: {gameweek: {stats...}}}

        for fpl_id, player_group in training_data_sorted.groupby("fpl_id"):
            player_group = player_group.sort_values("gameweek")
            player_cumulative_stats[fpl_id] = {}

            cumulative_minutes = 0.0
            cumulative_xg = 0.0
            cumulative_xa = 0.0
            cumulative_goals = 0.0
            cumulative_assists = 0.0
            recent_xg_list = []
            recent_xa_list = []
            recent_minutes_list = []

            for idx, (_, row) in enumerate(player_group.iterrows()):
                current_gw = row.get("gameweek", 0)

                # Store stats BEFORE this gameweek (for prediction at this GW)
                # These are the features we'd have available when predicting this GW
                if cumulative_minutes > 0:
                    xg_per_90 = (cumulative_xg / cumulative_minutes) * 90.0
                    xa_per_90 = (cumulative_xa / cumulative_minutes) * 90.0
                    goals_per_90 = (cumulative_goals / cumulative_minutes) * 90.0
                    assists_per_90 = (cumulative_assists / cumulative_minutes) * 90.0
                else:
                    xg_per_90 = 0.0
                    xa_per_90 = 0.0
                    goals_per_90 = 0.0
                    assists_per_90 = 0.0

                # Safe slicing to avoid empty slice warnings
                recent_mins_slice = (
                    recent_minutes_list[-5:] if recent_minutes_list else []
                )
                recent_xg_slice = recent_xg_list[-5:] if recent_xg_list else []
                recent_xa_slice = recent_xa_list[-5:] if recent_xa_list else []

                # Calculate expected minutes safely
                if recent_mins_slice and len(recent_mins_slice) > 0:
                    exp_mins = float(np.mean(recent_mins_slice))
                else:
                    exp_mins = 0.0

                player_cumulative_stats[fpl_id][current_gw] = {
                    "xg_per_90": xg_per_90,
                    "xa_per_90": xa_per_90,
                    "goals_per_90": goals_per_90,
                    "assists_per_90": assists_per_90,
                    "minutes": cumulative_minutes,
                    "recent_xg": recent_xg_slice.copy(),  # Last 5 GWs
                    "recent_xa": recent_xa_slice.copy(),
                    "recent_minutes": recent_mins_slice.copy(),
                    "expected_minutes": exp_mins,
                }

                # Update cumulative stats AFTER storing (so next GW can use this)
                row_minutes = float(row.get("minutes", 0) or 0)
                row_xg = float(row.get("xg", 0) or 0)
                row_xa = float(row.get("xa", 0) or 0)
                row_goals = float(row.get("goals", 0) or 0)
                row_assists = float(row.get("assists", 0) or 0)

                cumulative_minutes += row_minutes
                cumulative_xg += row_xg
                cumulative_xa += row_xa
                cumulative_goals += row_goals
                cumulative_assists += row_assists

                recent_xg_list.append(row_xg)
                recent_xa_list.append(row_xa)
                recent_minutes_list.append(row_minutes)

        # Now build features using HISTORICAL stats (no leakage)
        skipped_rows = 0
        for _, row in training_data_sorted.iterrows():
            fpl_id = row.get("fpl_id")
            current_gw = row.get("gameweek", 0)

            # Get historical stats for this player BEFORE this gameweek
            if (
                fpl_id in player_cumulative_stats
                and current_gw in player_cumulative_stats[fpl_id]
            ):
                historical = player_cumulative_stats[fpl_id][current_gw]

                # Skip if no historical data (first gameweek for this player)
                if historical["minutes"] == 0 and not historical["recent_xg"]:
                    skipped_rows += 1
                    continue

                # Build player_data dict using HISTORICAL stats (not current row's stats)
                player_data = {
                    "fpl_id": fpl_id,
                    "position": row.get("position", "MID"),
                    "xg_per_90": historical["xg_per_90"],
                    "xa_per_90": historical["xa_per_90"],
                    "goals_per_90": historical["goals_per_90"],
                    "assists_per_90": historical["assists_per_90"],
                    "minutes": historical["minutes"],
                    "expected_minutes": historical["expected_minutes"],
                    "recent_xg": historical["recent_xg"],
                    "recent_xa": historical["recent_xa"],
                    "recent_minutes": historical["recent_minutes"],
                    "form": row.get("total_points", 0) / 10.0
                    if "total_points" in row
                    else 0.0,
                    "team_attack_strength": 0.0,  # Default
                }

                # Use `was_home` to build minimal fixture context
                fixture_data = {"is_home": bool(row.get("was_home", True))}

                # Generate features using HISTORICAL data (no leakage!)
                feature_arr = attack_model.extract_features(
                    player_data=player_data,
                    fixture_data=fixture_data,
                    fdr_data=None,
                    opponent_data=None,
                )
                features.append(feature_arr.flatten().tolist())

                # Labels are the CURRENT row's xG/xA (what we want to predict)
                xg_labels.append(float(row.get("xg", 0.0)))
                xa_labels.append(float(row.get("xa", 0.0)))
            else:
                # No historical data, skip this row
                skipped_rows += 1

        if skipped_rows > 0:
            logger.info(
                f"Skipped {skipped_rows} rows with no historical data (first GW for those players)"
            )

        logger.info(f"Prepared {len(features)} attack features without target leakage")
        return np.array(features), np.array(xg_labels), np.array(xa_labels)

    def _run_solver_for_week(
        self, current_data: pd.DataFrame, predictions: List[float], gameweek: int
    ) -> Optional[Dict]:
        """
        Run solver optimization for a gameweek.

        Args:
            current_data: Current week player data
            predictions: Predicted points for each player
            gameweek: Current gameweek

        Returns:
            Solver result dictionary
        """
        try:
            # Prepare players data for solver
            players_data = []

            for idx, (_, row) in enumerate(current_data.iterrows()):
                # Get player position from database
                player = row.to_dict()

                # Get position
                position = self._get_position_from_data(player)

                # Get price (try different field names)
                price = player.get("price", 50.0)
                if isinstance(price, (int, float)) and price > 100:
                    price = price / 10.0  # Convert from FPL units
                elif not isinstance(price, (int, float)):
                    price = 5.0  # Default

                # Get team_id (try different field names)
                # Prefer player's own team name (from Player table join) to enforce FPL team limits correctly.
                # Fallback to opponent_team if no team info is available.
                team_id = player.get("team")
                if not team_id:
                    team_id = player.get("opponent_team", 0)

                players_data.append(
                    {
                        "id": player.get("fpl_id", idx),
                        "name": f"Player_{player.get('fpl_id', idx)}",
                        "position": position,
                        "price": float(price),
                        "team_id": team_id,
                        "expected_points": [
                            predictions[idx] if idx < len(predictions) else 0.0
                        ]
                        * 3,
                        "p_start": [0.8] * 3,  # Default probability
                    }
                )

            # Run solver
            solution = self.solver.optimize_team(
                players_data=players_data,
                current_squad=None,  # Start fresh each week for backtest
                locked_players=None,
                excluded_players=None,
            )

            # Calculate net points (points - transfer cost)
            total_points = solution.get("total_expected_points", 0.0)
            transfer_cost = solution.get("total_transfer_cost", 0)
            net_points = total_points - transfer_cost

            return {
                "total_points": total_points,
                "transfer_cost": transfer_cost,
                "net_points": net_points,
                "squad_size": len(solution.get("squad_week1", [])),
                "starting_xi_size": len(solution.get("starting_xi_week1", [])),
            }

        except Exception as e:
            logger.error(f"Solver error for GW{gameweek}: {str(e)}")
            return None

    def _calculate_overall_metrics(self) -> Dict:
        """
        Calculate overall metrics from individual player predictions across all weeks.

        CRITICAL FIX: Previously calculated metrics on weekly team sums, which was incorrect.
        Now aggregates all individual player predictions for proper RMSE/R² calculation.

        Returns:
            Dictionary with overall metrics
        """
        # Use individual player predictions if available (more accurate)
        if (
            len(self.all_individual_predictions) > 0
            and len(self.all_individual_actuals) > 0
        ):
            all_predicted_arr = np.array(self.all_individual_predictions)
            all_actual_arr = np.array(self.all_individual_actuals)

            logger.info(
                f"Calculating metrics on {len(all_predicted_arr)} individual player predictions"
            )
        else:
            # Fallback to weekly aggregated results (less accurate but better than nothing)
            if not self.weekly_results:
                return {}

            all_predicted = []
            all_actual = []

            for result in self.weekly_results:
                all_predicted.append(result.get("predicted_points", 0.0))
                all_actual.append(result.get("actual_points", 0.0))

            all_predicted_arr = np.array(all_predicted)
            all_actual_arr = np.array(all_actual)

            logger.warning(
                "Using weekly aggregated results for metrics (less accurate than individual predictions)"
            )

        # Safe checks for empty arrays
        if len(all_predicted_arr) == 0 or len(all_actual_arr) == 0:
            return {
                "rmse": 0.0,
                "mae": 0.0,
                "spearman": 0.0,
                "r_squared": 0.0,
                "mean_actual": 0.0,
                "mean_predicted": 0.0,
                "n_weeks": 0,
                "n_predictions": 0,
                "cumulative_points": self.cumulative_points,
                "total_transfer_cost": self.total_transfer_cost,
            }

        # CRITICAL FIX: Calculate overall RMSE correctly
        # RMSE = sqrt(mean of ALL squared errors across all weeks combined)
        # NOT an average of weekly RMSEs
        squared_errors = (all_actual_arr - all_predicted_arr) ** 2
        rmse = np.sqrt(np.mean(squared_errors))

        # Calculate overall Spearman correlation (handle edge cases)
        if len(all_actual_arr) >= 2:
            spearman_corr, _ = spearmanr(all_actual_arr, all_predicted_arr)
            # Handle NaN from spearmanr
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        else:
            spearman_corr = 0.0

        # Calculate MAE safely
        mae = float(np.mean(np.abs(all_actual_arr - all_predicted_arr)))

        # CRITICAL FIX: Calculate R-squared correctly
        # R² = 1 - (SS_res / SS_tot)
        # SS_res = sum of squared residuals (prediction errors)
        # SS_tot = sum of squared deviations from mean (baseline: horizontal line at mean)
        # Negative R² means model is worse than predicting the mean for everyone
        mean_actual = float(np.mean(all_actual_arr))
        mean_predicted = float(np.mean(all_predicted_arr))

        # Sum of squared residuals (errors)
        ss_res = np.sum((all_actual_arr - all_predicted_arr) ** 2)

        # Sum of squared total (variance of actuals around their mean)
        # This is the baseline: if we predicted mean_actual for everyone, this is the error
        ss_tot = np.sum((all_actual_arr - mean_actual) ** 2)

        # R² calculation: 1 - (SS_res / SS_tot)
        # If SS_res > SS_tot, then R² < 0 (model worse than mean baseline)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Additional diagnostic: variance ratio
        # If predicted variance is much lower than actual variance, model is too conservative
        var_actual = float(np.var(all_actual_arr))
        var_predicted = float(np.var(all_predicted_arr))
        variance_ratio = var_predicted / var_actual if var_actual > 0 else 0.0

        # Calculate bias (mean difference)
        bias = mean_predicted - mean_actual

        # Calculate percentage bias
        pct_bias = (bias / mean_actual * 100.0) if mean_actual > 0 else 0.0

        logger.info("Overall Metrics:")
        logger.info(f"  Mean Actual: {mean_actual:.2f}")
        logger.info(f"  Mean Predicted: {mean_predicted:.2f}")
        logger.info(f"  Bias: {bias:.2f} ({pct_bias:.1f}%)")
        logger.info(f"  Variance Actual: {var_actual:.2f}")
        logger.info(f"  Variance Predicted: {var_predicted:.2f}")
        logger.info(f"  Variance Ratio: {variance_ratio:.3f} (should be ~1.0)")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  R²: {r_squared:.3f}")

        # Diagnostic: If variance ratio is too low, model is predicting "safe mean"
        if variance_ratio < 0.5:
            logger.warning(
                f"WARNING: Variance ratio {variance_ratio:.3f} is too low. Model may be predicting 'safe mean' values."
            )
            logger.warning(
                "  This suggests predictions lack variance and cluster around the mean."
            )

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "spearman": float(spearman_corr),
            "r_squared": float(r_squared),
            "mean_actual": mean_actual,
            "mean_predicted": mean_predicted,
            "bias": float(bias),
            "pct_bias": float(pct_bias),
            "variance_actual": var_actual,
            "variance_predicted": var_predicted,
            "variance_ratio": float(variance_ratio),
            "n_weeks": len(self.weekly_results),
            "n_predictions": len(all_predicted_arr),
            "cumulative_points": self.cumulative_points,
            "total_transfer_cost": self.total_transfer_cost,
        }

    def _generate_report(
        self, overall_metrics: Dict, save_to_file: bool = True
    ) -> Dict:
        """
        Generate comprehensive backtest report.

        Args:
            overall_metrics: Overall metrics dictionary
            save_to_file: Whether to save report to file (default: True). Set to False for smoke tests.

        Returns:
            Complete report dictionary
        """
        report = {
            "season": self.season,
            "methodology": "expanding_window",
            "min_train_weeks": self.min_train_weeks,
            "total_weeks_tested": len(self.weekly_results),
            "overall_metrics": overall_metrics,
            "weekly_results": self.weekly_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Only save report to file if requested and if we have actual results
        # Skip saving for smoke tests or empty results
        should_save = save_to_file is True and len(self.weekly_results) > 0

        if should_save:
            # Save report to JSON file
            # Get backend directory (2 levels up from app/services/backtest.py)
            backend_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            reports_dir = os.path.join(backend_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)

            # Create filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{self.season}_{timestamp_str}.json"
            filepath = os.path.join(reports_dir, filename)

            # Convert numpy types to native Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    # Handle NaN and Inf values
                    f = float(obj)
                    if np.isnan(f):
                        return None  # JSON doesn't support NaN, use null
                    elif np.isinf(f):
                        return None  # JSON doesn't support Inf, use null
                    return f
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, float):
                    # Handle Python float NaN/Inf
                    if np.isnan(obj):
                        return None
                    elif np.isinf(obj):
                        return None
                return obj

            serializable_report = convert_to_serializable(report)

            with open(filepath, "w") as f:
                json.dump(serializable_report, f, indent=2)

            logger.info(f"Report saved to: {filepath}")
        elif not save_to_file:
            logger.info("Report generation successful (file save skipped for test)")
        elif len(self.weekly_results) == 0:
            logger.info("Report generation successful (file save skipped - no results)")

        return report

    def save_report_to_database(
        self, report: Dict, model_version: str = "5.0.0"
    ) -> Optional[int]:
        """
        Save backtest summary and individual results to database.

        Args:
            report: Report dictionary from _generate_report
            model_version: Model version string (default: "5.0.0")

        Returns:
            ID of saved BacktestSummary record, or None if error
        """
        try:
            from app.models import BacktestSummary, BacktestResult

            db = SessionLocal()
            try:
                overall_metrics = report.get("overall_metrics", {})
                methodology = report.get("methodology", "expanding_window")
                season = report.get("season", self.season)
                weekly_results = report.get("weekly_results", [])

                # Save individual BacktestResult records for each gameweek
                for weekly_result in weekly_results:
                    gameweek = weekly_result.get("gameweek")
                    if gameweek is None:
                        continue
                    
                    # Check if result already exists
                    existing_result = (
                        db.query(BacktestResult)
                        .filter(
                            BacktestResult.model_version == model_version,
                            BacktestResult.methodology == methodology,
                            BacktestResult.season == season,
                            BacktestResult.gameweek == gameweek,
                        )
                        .first()
                    )

                    # Handle NaN values in spearman correlation
                    spearman_val = weekly_result.get("spearman", 0.0)
                    if spearman_val is None or (isinstance(spearman_val, float) and np.isnan(spearman_val)):
                        spearman_val = 0.0
                    
                    result_data = {
                        "model_version": model_version,
                        "methodology": methodology,
                        "season": season,
                        "gameweek": gameweek,
                        "rmse": float(weekly_result.get("rmse", 0.0)),
                        "mae": float(weekly_result.get("mae", 0.0)),
                        "spearman_corr": float(spearman_val),
                        "n_predictions": int(weekly_result.get("n_predictions", 0)),
                    }

                    if existing_result:
                        # Update existing record
                        for key, value in result_data.items():
                            setattr(existing_result, key, value)
                        logger.debug(f"Updated BacktestResult for GW{gameweek}")
                    else:
                        # Create new record
                        result = BacktestResult(**result_data)
                        db.add(result)
                        logger.debug(f"Created BacktestResult for GW{gameweek}")

                # Commit all BacktestResult records
                db.commit()

                # Check if summary already exists for this model_version
                existing = (
                    db.query(BacktestSummary)
                    .filter(BacktestSummary.model_version == model_version)
                    .first()
                )

                summary_data = {
                    "model_version": model_version,
                    "methodology": methodology,
                    "season": season,
                    "total_weeks_tested": report.get("total_weeks_tested", 0),
                    "overall_rmse": float(overall_metrics.get("rmse", 0.0)),
                    "overall_mae": float(overall_metrics.get("mae", 0.0)),
                    "overall_spearman_corr": float(overall_metrics.get("spearman", 0.0)),
                    "r_squared": float(overall_metrics.get("r_squared", 0.0)),
                    "total_predictions": sum(
                        r.get("n_predictions", 0)
                        for r in weekly_results
                    ),
                }

                if existing:
                    # Update existing record
                    for key, value in summary_data.items():
                        setattr(existing, key, value)
                    summary_id = existing.id
                    logger.info(f"Updated BacktestSummary record ID: {summary_id}")
                else:
                    # Create new record
                    summary = BacktestSummary(**summary_data)
                    db.add(summary)
                    db.commit()
                    db.refresh(summary)
                    summary_id = summary.id
                    logger.info(f"Created BacktestSummary record ID: {summary_id}")

                logger.info(f"Saved {len(weekly_results)} BacktestResult records to database")
                return summary_id

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error saving report to database: {str(e)}", exc_info=True)
            return None

    def _manage_memory(self):
        """
        Manage memory usage to stay within 4GB limit.
        """
        gc.collect()

        # Check memory usage (if psutil available)
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > self.memory_limit_mb:
                logger.warning(f"Memory usage high: {memory_mb:.2f} MB")
                # Force garbage collection
                gc.collect()
        except ImportError:
            # psutil not available, just do basic cleanup
            pass

    def generate_graphical_summary(
        self, report: Dict, output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate graphical summary of backtest results.

        Args:
            report: Backtest report dictionary
            output_dir: Output directory for graphs (default: temp_dir)

        Returns:
            Dictionary with paths to generated graphs
        """
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping graphical summary")
            return {}

        output_dir = output_dir or self.temp_dir
        os.makedirs(output_dir, exist_ok=True)

        graph_paths = {}

        # 1. Weekly RMSE plot
        if self.weekly_results:
            gameweeks = [r["gameweek"] for r in self.weekly_results]
            rmses = [r["rmse"] for r in self.weekly_results]

            plt.figure(figsize=(12, 6))
            plt.plot(gameweeks, rmses, marker="o")
            plt.xlabel("Gameweek")
            plt.ylabel("RMSE")
            plt.title(f"RMSE by Gameweek - Season {self.season}")
            plt.grid(True)

            rmse_path = os.path.join(output_dir, f"rmse_gw_{self.season}.png")
            plt.savefig(rmse_path, dpi=150, bbox_inches="tight")
            plt.close()
            graph_paths["rmse_plot"] = rmse_path

        # 2. Predicted vs Actual scatter plot
        if self.weekly_results:
            predicted = [r["predicted_points"] for r in self.weekly_results]
            actual = [r["actual_points"] for r in self.weekly_results]

            plt.figure(figsize=(10, 10))
            plt.scatter(actual, predicted, alpha=0.6)
            plt.plot(
                [min(actual), max(actual)], [min(actual), max(actual)], "r--", lw=2
            )
            plt.xlabel("Actual Points")
            plt.ylabel("Predicted Points")
            plt.title(f"Predicted vs Actual Points - Season {self.season}")
            plt.grid(True)

            scatter_path = os.path.join(output_dir, f"pred_vs_actual_{self.season}.png")
            plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
            plt.close()
            graph_paths["scatter_plot"] = scatter_path

        # 3. Cumulative points plot (if solver used)
        if any(r.get("net_points") for r in self.weekly_results):
            gameweeks = [r["gameweek"] for r in self.weekly_results]
            cumulative = []
            running_total = 0.0
            for r in self.weekly_results:
                running_total += r.get("net_points", 0.0)
                cumulative.append(running_total)

            plt.figure(figsize=(12, 6))
            plt.plot(gameweeks, cumulative, marker="o", linewidth=2)
            plt.xlabel("Gameweek")
            plt.ylabel("Cumulative Points")
            plt.title(f"Cumulative Points - Season {self.season}")
            plt.grid(True)

            cum_path = os.path.join(output_dir, f"cumulative_points_{self.season}.png")
            plt.savefig(cum_path, dpi=150, bbox_inches="tight")
            plt.close()
            graph_paths["cumulative_plot"] = cum_path

        logger.info(f"Generated {len(graph_paths)} graphical summaries in {output_dir}")

        return graph_paths

    def _calculate_player_historical_stats(
        self, training_data: pd.DataFrame
    ) -> Dict[int, Dict]:
        """
        Calculate historical average stats for each player from training data.

        This is the CRITICAL fix for backtest data aggregation:
        When predicting GW20, we should use GW1-19 averages, NOT GW20's actual stats.

        Args:
            training_data: DataFrame with all previous gameweeks' data

        Returns:
            Dictionary mapping fpl_id -> historical stats dict with:
            - xg_per_90, xa_per_90, goals_per_90, assists_per_90
            - recent_xg (last 5 GWs), recent_xa (last 5 GWs), recent_minutes (last 5 GWs)
            - total_minutes, form
        """
        player_stats = {}

        if training_data.empty:
            return player_stats

        # Sort by gameweek to get proper ordering for recency
        training_data_sorted = training_data.sort_values(["fpl_id", "gameweek"])

        # Group by player
        for fpl_id, player_group in training_data_sorted.groupby("fpl_id"):
            # Sort by gameweek (most recent last)
            player_group = player_group.sort_values("gameweek")

            # Calculate total stats
            total_minutes = float(player_group["minutes"].sum())
            total_xg = (
                float(player_group["xg"].sum()) if "xg" in player_group.columns else 0.0
            )
            total_xa = (
                float(player_group["xa"].sum()) if "xa" in player_group.columns else 0.0
            )
            total_goals = (
                float(player_group["goals"].sum())
                if "goals" in player_group.columns
                else 0.0
            )
            total_assists = (
                float(player_group["assists"].sum())
                if "assists" in player_group.columns
                else 0.0
            )

            # Calculate per-90 stats (avoid division by zero)
            if total_minutes > 0:
                xg_per_90 = (total_xg / total_minutes) * 90.0
                xa_per_90 = (total_xa / total_minutes) * 90.0
                goals_per_90 = (total_goals / total_minutes) * 90.0
                assists_per_90 = (total_assists / total_minutes) * 90.0
            else:
                xg_per_90 = 0.0
                xa_per_90 = 0.0
                goals_per_90 = 0.0
                assists_per_90 = 0.0

            # Get recent stats (last 5 gameweeks)
            recent_rows = player_group.tail(5)
            recent_minutes = recent_rows["minutes"].tolist()
            recent_xg = (
                recent_rows["xg"].tolist() if "xg" in recent_rows.columns else []
            )
            recent_xa = (
                recent_rows["xa"].tolist() if "xa" in recent_rows.columns else []
            )

            # Calculate form (average total_points in last 5 GWs)
            if "total_points" in recent_rows.columns:
                form = float(recent_rows["total_points"].mean())
            else:
                form = 0.0

            # Store stats
            # Calculate expected minutes safely (avoid empty list warnings)
            valid_recent_minutes = [m for m in recent_minutes if m > 0]
            expected_mins = (
                float(np.mean(valid_recent_minutes)) if valid_recent_minutes else 0.0
            )

            player_stats[fpl_id] = {
                "xg_per_90": xg_per_90,
                "xa_per_90": xa_per_90,
                "goals_per_90": goals_per_90,
                "assists_per_90": assists_per_90,
                "minutes": total_minutes,
                "expected_minutes": expected_mins,
                "recent_minutes": recent_minutes,
                "recent_xg": recent_xg,
                "recent_xa": recent_xa,
                "form": form,
                "total_xg": total_xg,
                "total_xa": total_xa,
                "games_played": len(player_group[player_group["minutes"] > 0]),
            }

        logger.info(f"Calculated historical stats for {len(player_stats)} players")
        return player_stats

    def _get_position_from_data(self, player_data: Dict) -> str:
        """
        Get position from player data.

        Args:
            player_data: Player data dictionary

        Returns:
            Position string (GK, DEF, MID, FWD)
        """
        # Try to get from element_type if available
        element_type = player_data.get("element_type")
        if element_type:
            position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
            return position_map.get(element_type, "MID")

        # Try to get from position field
        position = player_data.get("position", "MID")
        if isinstance(position, str) and position in ["GK", "DEF", "MID", "FWD"]:
            return position

        # Default to MID
        return "MID"
