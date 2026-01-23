"""
Backtesting Service for FPL Predictions
Implements expanding window methodology to test model performance
without look-ahead bias.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """
    Backtesting engine using expanding window methodology.
    Simulates season week-by-week, training on past data and testing on future.
    """

    def __init__(self, min_train_weeks: int = 5):
        """
        Initialize backtesting engine.

        Args:
            min_train_weeks: Minimum weeks needed for training (default: 5)
        """
        self.min_train_weeks = min_train_weeks
        self.results: List[Dict] = []

    def expanding_window_backtest(
        self,
        historical_data: pd.DataFrame,
        prediction_function,
        target_column: str = "points",
        gameweek_column: str = "gameweek",
        player_id_column: str = "player_id",
        min_train_weeks: Optional[int] = None,
    ) -> Dict:
        """
        Perform expanding window backtest.

        Methodology:
        - Week 1-5: Train initial model
        - Week 6: Predict using weeks 1-5, compare with actual
        - Week 7: Predict using weeks 1-6, compare with actual
        - Continue until end of season

        Args:
            historical_data: DataFrame with historical player data
            prediction_function: Function that takes (train_data, test_data) and returns predictions
            target_column: Column name for target variable (points)
            gameweek_column: Column name for gameweek
            player_id_column: Column name for player ID

        Returns:
            Dictionary with backtest results and metrics
        """
        if historical_data.empty:
            return {"error": "No historical data provided"}

        # Get unique gameweeks
        gameweeks = sorted(historical_data[gameweek_column].unique())

        # Use provided min_train_weeks or default
        min_weeks = (
            min_train_weeks if min_train_weeks is not None else self.min_train_weeks
        )

        if len(gameweeks) < min_weeks + 1:
            return {
                "error": f"Insufficient data. Need at least {min_weeks + 1} gameweeks"
            }

        all_predictions = []
        all_actuals = []
        weekly_results = []

        # Expanding window: start from min_train_weeks
        for test_gw in gameweeks[min_weeks:]:
            train_gws = [gw for gw in gameweeks if gw < test_gw]
            test_gw_data = historical_data[historical_data[gameweek_column] == test_gw]

            if test_gw_data.empty:
                continue

            # Get training data (all weeks before test week)
            train_data = historical_data[
                historical_data[gameweek_column].isin(train_gws)
            ]

            if train_data.empty:
                continue

            try:
                # Make predictions
                predictions = prediction_function(train_data, test_gw_data)

                # Extract actual values
                actuals = test_gw_data[target_column].values
                pred_values = (
                    predictions.values
                    if hasattr(predictions, "values")
                    else predictions
                )

                # Align predictions with actuals by player_id
                if isinstance(predictions, pd.Series):
                    pred_df = predictions.reset_index()
                    pred_df.columns = [player_id_column, "predicted"]
                    test_with_pred = test_gw_data.merge(
                        pred_df, on=player_id_column, how="left"
                    )
                    pred_values = test_with_pred["predicted"].fillna(0).values
                    actuals = test_with_pred[target_column].values

                # Calculate metrics for this week
                week_metrics = self._calculate_weekly_metrics(
                    actuals, pred_values, test_gw
                )

                weekly_results.append(week_metrics)
                all_predictions.extend(pred_values)
                all_actuals.extend(actuals)

                logger.info(
                    f"Gameweek {test_gw}: RMSE={week_metrics['rmse']:.2f}, MAE={week_metrics['mae']:.2f}"
                )

            except Exception as e:
                logger.error(f"Error in gameweek {test_gw}: {str(e)}")
                continue

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(
            np.array(all_actuals), np.array(all_predictions)
        )

        return {
            "methodology": "expanding_window",
            "min_train_weeks": self.min_train_weeks,
            "total_weeks_tested": len(weekly_results),
            "weekly_results": weekly_results,
            "overall_metrics": overall_metrics,
            "total_predictions": len(all_predictions),
        }

    def _calculate_weekly_metrics(
        self, actuals: np.ndarray, predictions: np.ndarray, gameweek: int
    ) -> Dict:
        """Calculate metrics for a single week"""
        if len(actuals) == 0 or len(predictions) == 0:
            return {
                "gameweek": gameweek,
                "rmse": 0.0,
                "mae": 0.0,
                "spearman_corr": 0.0,
                "n_predictions": 0,
            }

        # Remove NaN values
        mask = ~(np.isnan(actuals) | np.isnan(predictions))
        actuals_clean = actuals[mask]
        predictions_clean = predictions[mask]

        if len(actuals_clean) == 0:
            return {
                "gameweek": gameweek,
                "rmse": 0.0,
                "mae": 0.0,
                "spearman_corr": 0.0,
                "n_predictions": 0,
            }

        rmse = np.sqrt(mean_squared_error(actuals_clean, predictions_clean))
        mae = mean_absolute_error(actuals_clean, predictions_clean)

        # Spearman correlation (ranking accuracy)
        try:
            spearman_corr, _ = spearmanr(actuals_clean, predictions_clean)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        except Exception:
            spearman_corr = 0.0

        return {
            "gameweek": gameweek,
            "rmse": float(rmse),
            "mae": float(mae),
            "spearman_corr": float(spearman_corr),
            "n_predictions": len(actuals_clean),
            "mean_actual": float(np.mean(actuals_clean)),
            "mean_predicted": float(np.mean(predictions_clean)),
        }

    def _calculate_overall_metrics(
        self, actuals: np.ndarray, predictions: np.ndarray
    ) -> Dict:
        """Calculate overall metrics across all weeks"""
        if len(actuals) == 0 or len(predictions) == 0:
            return {"rmse": 0.0, "mae": 0.0, "spearman_corr": 0.0, "r_squared": 0.0}

        # Remove NaN values
        mask = ~(np.isnan(actuals) | np.isnan(predictions))
        actuals_clean = actuals[mask]
        predictions_clean = predictions[mask]

        if len(actuals_clean) == 0:
            return {"rmse": 0.0, "mae": 0.0, "spearman_corr": 0.0, "r_squared": 0.0}

        rmse = np.sqrt(mean_squared_error(actuals_clean, predictions_clean))
        mae = mean_absolute_error(actuals_clean, predictions_clean)

        # Spearman correlation
        try:
            spearman_corr, _ = spearmanr(actuals_clean, predictions_clean)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        except Exception:
            spearman_corr = 0.0

        # R-squared
        ss_res = np.sum((actuals_clean - predictions_clean) ** 2)
        ss_tot = np.sum((actuals_clean - np.mean(actuals_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "spearman_corr": float(spearman_corr),
            "r_squared": float(r_squared),
            "mean_actual": float(np.mean(actuals_clean)),
            "mean_predicted": float(np.mean(predictions_clean)),
            "n_predictions": len(actuals_clean),
        }

    def rolling_window_backtest(
        self,
        historical_data: pd.DataFrame,
        prediction_function,
        window_size: int = 5,
        target_column: str = "points",
        gameweek_column: str = "gameweek",
    ) -> Dict:
        """
        Perform rolling window backtest (alternative methodology).
        Uses fixed-size window instead of expanding.

        Args:
            historical_data: Historical data
            prediction_function: Prediction function
            window_size: Size of rolling window
            target_column: Target column name
            gameweek_column: Gameweek column name

        Returns:
            Backtest results
        """
        gameweeks = sorted(historical_data[gameweek_column].unique())

        if len(gameweeks) < window_size + 1:
            return {
                "error": f"Insufficient data. Need at least {window_size + 1} gameweeks"
            }

        all_predictions = []
        all_actuals = []
        weekly_results = []

        for i in range(window_size, len(gameweeks)):
            test_gw = gameweeks[i]
            train_gws = gameweeks[i - window_size : i]

            train_data = historical_data[
                historical_data[gameweek_column].isin(train_gws)
            ]
            test_data = historical_data[historical_data[gameweek_column] == test_gw]

            if train_data.empty or test_data.empty:
                continue

            try:
                predictions = prediction_function(train_data, test_data)
                actuals = test_data[target_column].values

                week_metrics = self._calculate_weekly_metrics(
                    actuals, predictions, test_gw
                )

                weekly_results.append(week_metrics)
                all_predictions.extend(predictions)
                all_actuals.extend(actuals)

            except Exception as e:
                logger.error(f"Error in gameweek {test_gw}: {str(e)}")
                continue

        overall_metrics = self._calculate_overall_metrics(
            np.array(all_actuals), np.array(all_predictions)
        )

        return {
            "methodology": "rolling_window",
            "window_size": window_size,
            "total_weeks_tested": len(weekly_results),
            "weekly_results": weekly_results,
            "overall_metrics": overall_metrics,
        }

    def compare_models(
        self,
        historical_data: pd.DataFrame,
        models: Dict[str, callable],
        target_column: str = "points",
    ) -> Dict:
        """
        Compare multiple models using backtesting.

        Args:
            historical_data: Historical data
            models: Dictionary of {model_name: prediction_function}
            target_column: Target column name

        Returns:
            Comparison results
        """
        results = {}

        for model_name, prediction_function in models.items():
            logger.info(f"Backtesting model: {model_name}")
            result = self.expanding_window_backtest(
                historical_data, prediction_function, target_column=target_column
            )
            results[model_name] = result

        # Find best model
        best_model = None
        best_rmse = float("inf")

        for model_name, result in results.items():
            if "overall_metrics" in result:
                rmse = result["overall_metrics"].get("rmse", float("inf"))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_name

        return {"models": results, "best_model": best_model, "best_rmse": best_rmse}
