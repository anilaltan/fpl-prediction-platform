"""
PLEngine - Predictive Engine
Main orchestrator for FPL point prediction using modular ML strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
import gc
import pickle
import asyncio
import os

from .strategies.xmins_strategy import XMinsStrategy
from .strategies.attack_strategy import AttackStrategy
from .strategies.defense_strategy import DefenseStrategy
from .model_loader import ModelLoader
from app.services.feature_engineering import FeatureEngineeringService
from app.services.data_cleaning import DataCleaningService

logger = logging.getLogger(__name__)


class PLEngine:
    """
    Predictive Engine (PLEngine) for FPL Point Prediction.

    Implements component-based models with async resource management:
    - xMins Model: XGBoost/RandomForest for starting 11 probability
    - Attack Model: LightGBM for xG/xA predictions with opponent xGC normalization
    - Defense Model: LightGBM/Poisson for clean sheet probability
    - Final xP Calculation: Comprehensive expected points formula
    - Resource Management: Lazy loading/unloading with gc.collect()
    """

    # FPL Point Values (2025/26 rules)
    GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
    ASSIST_POINTS = 3
    CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize PLEngine with modular strategies.

        Args:
            model_path: Optional path to model file. If None, tries to find latest model.
        """
        self.model_version = "6.0.0"  # Modular version with lazy loading

        # Initialize strategies (not loaded yet - lazy loading)
        self.xmins_strategy = XMinsStrategy()
        self.attack_strategy = AttackStrategy()
        self.defense_strategy = DefenseStrategy()

        # Model loader for resource management
        self.model_loader = ModelLoader()

        # Feature engineering
        self.feature_engine = FeatureEngineeringService()
        self.data_cleaning = DataCleaningService()

        # Model loading state
        self.model_path = model_path
        if not self.model_path:
            self.model_path = self._load_latest_model()

        # Calibration layer
        self.calibration_enabled = True
        self.calibration_scale = 1.0
        self.calibration_offset = 0.0
        self.calibration_fitted = False
        self._historical_mean_actual = None
        self._historical_mean_predicted = None

    def _load_latest_model(self) -> Optional[str]:
        """
        Search for the most recent .pkl model file in the models directory.

        Returns:
            Path to the latest model file, or None if no models found.
        """
        try:
            possible_dirs = []

            # 1. Relative to this file: backend/models/
            file_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_dir)))
            possible_dirs.append(os.path.join(backend_dir, "models"))

            # 2. Absolute path: /app/models (Docker environment)
            possible_dirs.append("/app/models")

            # 3. Current working directory models/
            possible_dirs.append(os.path.join(os.getcwd(), "models"))

            models_dir = None
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    models_dir = dir_path
                    break

            if not models_dir:
                logger.debug(f"Models directory not found in any of: {possible_dirs}")
                return None

            # Find all .pkl files
            pkl_files = [
                os.path.join(models_dir, f)
                for f in os.listdir(models_dir)
                if f.endswith(".pkl") and os.path.isfile(os.path.join(models_dir, f))
            ]

            if not pkl_files:
                logger.debug(f"No .pkl files found in {models_dir}")
                return None

            # Sort by modification time (most recent first)
            pkl_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_model = pkl_files[0]

            logger.info(f"Found latest model: {latest_model}")
            return latest_model

        except Exception as e:
            logger.warning(f"Error searching for latest model: {str(e)}")
            return None

    async def _ensure_models_loaded(self) -> None:
        """
        Ensure models are loaded (async version).

        Loads models using ModelLoader if not already loaded.
        """
        if not self.xmins_strategy.is_loaded:
            await self.model_loader.load_model(self.xmins_strategy, self.model_path)
        if not self.attack_strategy.is_loaded:
            await self.model_loader.load_model(self.attack_strategy, self.model_path)
        if not self.defense_strategy.is_loaded:
            await self.model_loader.load_model(self.defense_strategy, self.model_path)

    def _ensure_models_loaded_sync(self) -> None:
        """
        Ensure models are loaded (synchronous fallback).

        Used for backward compatibility with synchronous code.
        """
        if not self.xmins_strategy.is_loaded:
            # Synchronous load as fallback
            asyncio.run(
                self.model_loader.load_model(self.xmins_strategy, self.model_path)
            )
        if not self.attack_strategy.is_loaded:
            asyncio.run(
                self.model_loader.load_model(self.attack_strategy, self.model_path)
            )
        if not self.defense_strategy.is_loaded:
            asyncio.run(
                self.model_loader.load_model(self.defense_strategy, self.model_path)
            )

    async def async_load_models(self, model_path: Optional[str] = None) -> None:
        """
        Asynchronously load models from pickle file.

        Args:
            model_path: Optional path to model file
        """
        path = model_path or self.model_path
        await self.model_loader.load_model(self.xmins_strategy, path)
        await self.model_loader.load_model(self.attack_strategy, path)
        await self.model_loader.load_model(self.defense_strategy, path)

    async def async_unload_models(self) -> None:
        """
        Asynchronously unload all models from memory.
        Frees up RAM for other operations.
        """
        await self.model_loader.unload_all()

    async def async_save_models(self, model_path: Optional[str] = None) -> None:
        """
        Asynchronously save models to pickle file.

        Args:
            model_path: Optional path to save models
        """
        path = model_path or self.model_path
        if not path:
            logger.warning("No model path specified for saving")
            return

        # Ensure models are loaded before saving
        await self._ensure_models_loaded()

        if not self.xmins_strategy.is_trained or not self.attack_strategy.is_trained:
            logger.warning("Models not trained, cannot save")
            return

        try:
            # Prepare model data
            model_data = {
                "xmins_model": self.xmins_strategy.model
                if self.xmins_strategy.is_trained
                else None,
                "xmins_scaler": self.xmins_strategy.scaler,
                "xmins_feature_names": self.xmins_strategy.feature_names,
                "attack_xg_model": self.attack_strategy.xg_model
                if self.attack_strategy.xg_trained
                else None,
                "attack_xa_model": self.attack_strategy.xa_model
                if self.attack_strategy.xa_trained
                else None,
                "attack_scaler": self.attack_strategy.scaler,
                "defense_model": self.defense_strategy.pcs_model
                if self.defense_strategy.is_fitted
                else None,
                "defense_scaler": self.defense_strategy.scaler,
                "version": self.model_version,
            }

            # Run pickle save in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_pickle, path, model_data)

            logger.info(f"Models saved asynchronously to {path}")
            gc.collect()

        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")

    def _save_pickle(self, path: str, data: Dict) -> None:
        """Synchronous pickle save (runs in executor)."""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def calculate_expected_points(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None,
        fdr_data: Optional[Dict] = None,
        team_data: Optional[Dict] = None,
        opponent_data: Optional[Dict] = None,
        historical_points: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate final expected points (xP) using FPL position-based scoring rules.

        Formula: xP = (xMins/90) * [(Goal_Points * xG) + (Assist_Points * xA) + (CS_Points * xCS) + DefCon_Points]

        Args:
            player_data: Player statistics
            fixture_data: Upcoming fixture information
            fdr_data: FDR data
            team_data: Player's team data
            opponent_data: Opponent team data
            historical_points: Historical points for form calculation

        Returns:
            Dictionary with xP and component breakdowns
        """
        # Ensure models are loaded (synchronous fallback for compatibility)
        self._ensure_models_loaded_sync()

        position = player_data.get("position", "MID")
        fpl_id = player_data.get("fpl_id", "unknown")

        # Check if models are trained
        if not self.xmins_strategy.is_trained:
            logger.warning(
                f"xMins model not trained! Returning 0 for player fpl_id={fpl_id}"
            )
            return {
                "expected_points": 0.0,
                "xmins": 0.0,
                "xmins_factor": 0.0,
                "xg": 0.0,
                "xa": 0.0,
                "xcs": 0.0,
                "defcon_points": 0.0,
                "defcon_points_90": 0.0,
                "goal_component": 0.0,
                "assist_component": 0.0,
                "cs_component": 0.0,
                "appearance_points": 0.0,
                "expected_bonus": 0.0,
                "p_start": 0.0,
            }

        # 1. Predict xMins
        xmins_result = self.xmins_strategy.predict(player_data, fixture_data)
        xmins = xmins_result["expected_minutes"]
        p_start = xmins_result["p_start"]
        xmins_factor = xmins / 90.0

        # 2. Predict xG and xA
        if not self.attack_strategy.is_trained:
            logger.warning(
                f"Attack model not trained! Returning 0 xG/xA for player fpl_id={fpl_id}"
            )
            xg = 0.0
            xa = 0.0
        else:
            attack_preds = self.attack_strategy.predict(
                player_data, fixture_data, fdr_data, opponent_data
            )
            xg = attack_preds["xg"]
            xa = attack_preds["xa"]

        # 3. Predict xCS (only for defenders/GKs)
        xcs = 0.0
        if position in ["DEF", "GK"] and team_data and opponent_data:
            is_home = fixture_data.get("is_home", True) if fixture_data else True
            xcs = self.defense_strategy.predict_clean_sheet_probability(
                team_data=team_data, opponent_data=opponent_data, is_home=is_home
            )

        # 4. Calculate DefCon points
        defcon_metrics = self.data_cleaning.get_defcon_metrics(player_data, position)
        defcon_points_90 = defcon_metrics["floor_points_90"]

        # 5. Get FPL point values for position
        goal_points = self.GOAL_POINTS.get(position, 4)
        assist_points = self.ASSIST_POINTS
        cs_points = self.CLEAN_SHEET_POINTS.get(position, 0)

        # 6. Calculate expected points components
        goal_component = goal_points * xg
        assist_component = assist_points * xa
        cs_component = cs_points * xcs

        # 7. FPL Appearance Points
        appearance_points = xmins_factor * 2.0

        # 8. Expected Bonus Points
        ict_index = float(
            player_data.get("ict_index", player_data.get("influence", 0)) or 0
        )
        expected_bonus = (goal_component + assist_component) * 0.15 + (ict_index * 0.02)
        expected_bonus = float(np.clip(expected_bonus, 0.0, 3.0))

        # 9. Big Chance multiplier for high-value attackers
        big_chance_multiplier = 1.0
        is_high_value_attacker = (
            (xg > 0.5 or xa > 0.3)
            or (ict_index > 50)
            or (player_data.get("form", 0) > 5.0)
        )

        xp_temp = xmins_factor * (
            goal_component + assist_component + cs_component + defcon_points_90
        )
        xp_temp += appearance_points

        if is_high_value_attacker and xp_temp < 5.0:
            big_chance_multiplier = 1.15 + min(0.15, (xg + xa) * 0.1)
            goal_component = goal_component * big_chance_multiplier
            assist_component = assist_component * big_chance_multiplier
            expected_bonus = expected_bonus * big_chance_multiplier

        # 10. Final xP calculation
        xp = xmins_factor * (
            goal_component + assist_component + cs_component + defcon_points_90
        )
        xp += appearance_points
        xp += expected_bonus

        # Handle zero-inflated distribution
        if xp < 1.0 and xmins_factor > 0.1:
            xp = max(xp, appearance_points * 0.5)

        xp = max(0.0, xp)

        # Apply calibration layer
        if self.calibration_enabled and self.calibration_fitted:
            xp_calibrated = (xp * self.calibration_scale) + self.calibration_offset
            xp = max(0.0, xp_calibrated)
        elif hasattr(self, "_historical_mean_actual") and hasattr(
            self, "_historical_mean_predicted"
        ):
            if self._historical_mean_predicted > 0:
                conservative_scale = (
                    self._historical_mean_actual / self._historical_mean_predicted
                )
                if abs(conservative_scale - 1.0) > 0.1:
                    xp = xp * conservative_scale

        # Memory management
        gc.collect()

        return {
            "expected_points": float(xp),
            "xmins": float(xmins),
            "xmins_factor": float(xmins_factor),
            "xg": float(xg),
            "xa": float(xa),
            "xcs": float(xcs),
            "defcon_points": float(defcon_points_90),
            "defcon_points_90": float(defcon_points_90),
            "goal_component": float(goal_component),
            "assist_component": float(assist_component),
            "cs_component": float(cs_component),
            "appearance_points": float(appearance_points),
            "expected_bonus": float(expected_bonus),
            "p_start": float(p_start),
        }

    def train(
        self,
        training_data: pd.DataFrame,
        xmins_features: Optional[np.ndarray] = None,
        xmins_labels: Optional[np.ndarray] = None,
        attack_features: Optional[np.ndarray] = None,
        attack_xg_labels: Optional[np.ndarray] = None,
        attack_xa_labels: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train all component models.

        Args:
            training_data: Historical training data
            xmins_features: Features for xMins model
            xmins_labels: Labels for xMins (1=started, 0=didn't start)
            attack_features: Features for Attack model
            attack_xg_labels: Target xG values
            attack_xa_labels: Target xA values
        """
        # Ensure strategies are initialized
        if not self.xmins_strategy.is_loaded:
            asyncio.run(self.xmins_strategy.load())
        if not self.attack_strategy.is_loaded:
            asyncio.run(self.attack_strategy.load())
        if not self.defense_strategy.is_loaded:
            asyncio.run(self.defense_strategy.load())

        # Train xMins model
        if xmins_features is not None and xmins_labels is not None:
            if len(xmins_features) > 0 and len(xmins_labels) > 0:
                self.xmins_strategy.train(xmins_features, xmins_labels)
                gc.collect()

        # Train Attack model
        if (
            attack_features is not None
            and attack_xg_labels is not None
            and attack_xa_labels is not None
        ):
            if (
                len(attack_features) > 0
                and len(attack_xg_labels) > 0
                and len(attack_xa_labels) > 0
            ):
                self.attack_strategy.train(
                    attack_features, attack_xg_labels, attack_xa_labels
                )
                gc.collect()

        # Fit Defense model (Poisson - no training data needed, uses team strengths)
        if not training_data.empty:
            self.defense_strategy.is_fitted = True
            logger.info("Defense model marked as fitted (using Poisson calculation)")
            gc.collect()

        logger.info("All PLEngine models trained successfully")

    def predict(
        self,
        player_data: Dict,
        historical_points: Optional[List[float]] = None,
        fixture_data: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Predict points for a player using comprehensive xP calculation.

        Args:
            player_data: Current player statistics
            historical_points: Historical points (most recent first)
            fixture_data: Upcoming fixture information

        Returns:
            Dictionary with predicted points and component breakdowns
        """
        # Get FDR and team data
        fdr_data = None
        team_data = None
        opponent_data = None

        if fixture_data:
            engineered_features = self.feature_engine.calculate_all_features(
                player_data=player_data,
                historical_points=historical_points or [],
                fixture_data=fixture_data,
                position=player_data.get("position", "MID"),
            )

            fdr_data = {
                "fdr": engineered_features.get("fdr", 3.0),
                "opponent_defense_strength": engineered_features.get(
                    "fdr_defense", 0.0
                ),
                "opponent_attack_strength": engineered_features.get("fdr_attack", 0.0),
            }

            opponent_team_id = fixture_data.get("opponent_team")
            if opponent_team_id:
                opponent_xgc = 1.5 - (fdr_data["opponent_defense_strength"] * 0.1)
                opponent_data = {
                    "xgc_per_90": opponent_xgc,
                    "expected_goals_conceded": opponent_xgc,
                    "attack_strength": fdr_data["opponent_attack_strength"],
                    "defense_strength": fdr_data["opponent_defense_strength"],
                }

        # Calculate comprehensive xP
        xp_result = self.calculate_expected_points(
            player_data=player_data,
            fixture_data=fixture_data,
            fdr_data=fdr_data,
            team_data=team_data,
            opponent_data=opponent_data,
            historical_points=historical_points,
        )

        # Add confidence score
        confidence = self._calculate_confidence(player_data, xp_result)

        result = {
            "predicted_points": xp_result["expected_points"],
            "confidence_score": confidence,
            **xp_result,
        }

        # Memory management
        gc.collect()

        return result

    def _calculate_confidence(
        self, player_data: Dict, xp_result: Dict[str, float]
    ) -> float:
        """
        Calculate confidence score based on data quality and model certainty.

        Args:
            player_data: Player statistics
            xp_result: Prediction result

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.7  # Base confidence

        if player_data.get("minutes", 0) > 500:
            confidence += 0.1

        if xp_result.get("p_start", 0) > 0.8:
            confidence += 0.1

        status = player_data.get("status", "a").lower()
        if status == "d":
            confidence -= 0.2
        elif status == "i" or status == "s":
            confidence -= 0.3

        return float(np.clip(confidence, 0.0, 1.0))

    def fit_calibration(
        self,
        predicted_points: np.ndarray,
        actual_points: np.ndarray,
        method: str = "linear",
    ) -> Dict[str, float]:
        """
        Fit calibration layer to align predicted scale with actual FPL point distribution.

        Args:
            predicted_points: Array of predicted points
            actual_points: Array of actual points
            method: Calibration method ('linear' or 'isotonic')

        Returns:
            Dictionary with calibration parameters and metrics
        """
        if len(predicted_points) == 0 or len(actual_points) == 0:
            logger.warning("Cannot fit calibration: empty data")
            return {}

        predicted_arr = np.array(predicted_points)
        actual_arr = np.array(actual_points)

        # Remove NaN or inf values
        valid_mask = np.isfinite(predicted_arr) & np.isfinite(actual_arr)
        predicted_arr = predicted_arr[valid_mask]
        actual_arr = actual_arr[valid_mask]

        if len(predicted_arr) == 0:
            logger.warning("Cannot fit calibration: no valid data after filtering")
            return {}

        # Store historical means
        self._historical_mean_actual = float(np.mean(actual_arr))
        self._historical_mean_predicted = float(np.mean(predicted_arr))

        if method == "linear":
            mean_pred = float(np.mean(predicted_arr))
            mean_actual = float(np.mean(actual_arr))
            var_pred = float(np.var(predicted_arr))
            _var_actual = float(np.var(actual_arr))

            if var_pred > 0:
                covariance = float(
                    np.mean((predicted_arr - mean_pred) * (actual_arr - mean_actual))
                )
                scale_ls = covariance / var_pred if var_pred > 0 else 1.0
                scale_mean = mean_actual / mean_pred if mean_pred > 0 else 1.0

                if var_pred > 0.1 and abs(scale_ls) < 5.0:
                    self.calibration_scale = 0.6 * scale_ls + 0.4 * scale_mean
                else:
                    self.calibration_scale = scale_mean
            else:
                self.calibration_scale = (
                    mean_actual / mean_pred if mean_pred > 0 else 1.0
                )

            self.calibration_offset = mean_actual - (self.calibration_scale * mean_pred)
            self.calibration_offset = float(np.clip(self.calibration_offset, -2.0, 2.0))

            logger.info(
                f"Linear calibration fitted: scale={self.calibration_scale:.3f}, offset={self.calibration_offset:.3f}"
            )

        self.calibration_fitted = True

        # Calculate metrics
        calibrated_pred = (
            predicted_arr * self.calibration_scale
        ) + self.calibration_offset
        calibrated_pred = np.clip(calibrated_pred, 0.0, None)

        rmse_before = float(np.sqrt(np.mean((actual_arr - predicted_arr) ** 2)))
        rmse_after = float(np.sqrt(np.mean((actual_arr - calibrated_pred) ** 2)))

        return {
            "method": method,
            "scale": float(self.calibration_scale),
            "offset": float(self.calibration_offset),
            "rmse_before": rmse_before,
            "rmse_after": rmse_after,
            "improvement_pct": float((rmse_before - rmse_after) / rmse_before * 100)
            if rmse_before > 0
            else 0.0,
        }

    # Backward compatibility properties
    @property
    def xmins_model(self):
        """Backward compatibility: return xmins_strategy."""
        return self.xmins_strategy

    @property
    def attack_model(self):
        """Backward compatibility: return attack_strategy."""
        return self.attack_strategy

    @property
    def defense_model(self):
        """Backward compatibility: return defense_strategy."""
        return self.defense_strategy

    @property
    def models_loaded(self) -> bool:
        """Check if all models are loaded."""
        return (
            self.xmins_strategy.is_loaded
            and self.attack_strategy.is_loaded
            and self.defense_strategy.is_loaded
        )
