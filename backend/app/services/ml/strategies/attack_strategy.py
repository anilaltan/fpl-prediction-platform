"""
Attack Strategy
LightGBM-based model for predicting xG and xA with opponent xGC normalization.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import gc
import pickle
import asyncio
import os

# LightGBM for Attack Model
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ..interfaces import ModelInterface

logger = logging.getLogger(__name__)


class AttackStrategy(ModelInterface):
    """
    LightGBM model for predicting xG and xA.

    Combines player xG/xA data with opponent xGC (Expected Goals Conceded).
    Implements ModelInterface for lazy loading and memory management.
    """

    def __init__(self):
        """Initialize Attack strategy with empty models (lazy loaded)."""
        self.xg_model: Optional[object] = None
        self.xa_model: Optional[object] = None
        self.scaler: Optional[StandardScaler] = None
        self.xg_trained: bool = False
        self.xa_trained: bool = False
        self._loaded: bool = False
        self._model_path: Optional[str] = None
        # Auto-initialize empty models for immediate use
        self._initialize_empty_model()
        self._loaded = True

    async def load(self, model_path: Optional[str] = None) -> None:
        """
        Load model into memory from file or initialize empty model.

        Args:
            model_path: Optional path to model file. If None, initializes empty model.
        """
        if self._loaded:
            return

        self._model_path = model_path

        if model_path and os.path.exists(model_path):
            try:
                # Load from pickle file asynchronously
                loop = asyncio.get_event_loop()
                model_data = await loop.run_in_executor(
                    None, self._load_pickle, model_path
                )

                if model_data.get("attack_xg_model") and model_data.get(
                    "attack_xa_model"
                ):
                    self.xg_model = model_data["attack_xg_model"]
                    self.xa_model = model_data["attack_xa_model"]
                    self.scaler = model_data.get("attack_scaler", StandardScaler())
                    self.xg_trained = True
                    self.xa_trained = True
                    logger.info(f"Loaded Attack model from {model_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load Attack model from {model_path}: {str(e)}"
                )
                self._initialize_empty_model()
        else:
            self._initialize_empty_model()

        self._loaded = True

    def _initialize_empty_model(self) -> None:
        """Initialize empty models (not trained)."""
        if LIGHTGBM_AVAILABLE:
            self.xg_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
                n_jobs=1,  # Memory efficient
                max_bin=255,
            )
            self.xa_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
                n_jobs=1,  # Memory efficient
                max_bin=255,
            )
        else:
            self.xg_model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=1
            )
            self.xa_model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=1
            )
        self.scaler = StandardScaler()
        self.xg_trained = False
        self.xa_trained = False

    def _load_pickle(self, path: str) -> Dict:
        """Load model data from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    async def unload(self) -> None:
        """
        Unload model from memory and call gc.collect() to free RAM.

        Critical for 4GB RAM constraint - models should only be loaded
        during inference and unloaded immediately after.
        """
        if not self._loaded:
            return

        self.xg_model = None
        self.xa_model = None
        self.scaler = None
        self._loaded = False

        # Force garbage collection
        gc.collect()
        logger.debug("Attack model unloaded from memory")

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._loaded and self.xg_model is not None and self.xa_model is not None

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self.xg_trained and self.xa_trained

    def extract_features(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None,
        fdr_data: Optional[Dict] = None,
        opponent_data: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Extract features for xG/xA prediction.

        Features:
        - Player historical xG/xA per 90
        - Opponent xGC (Expected Goals Conceded) - KEY FEATURE
        - Home/away advantage
        - Recent form
        - Fixture difficulty
        """
        # Player historical stats with robust fallbacks
        minutes = float(player_data.get("minutes", 0) or 0)
        per90_scale = 90.0 / max(minutes, 1.0)

        xg_per_90 = float(
            player_data.get(
                "xg_per_90",
                player_data.get(
                    "expected_goals", player_data.get("xg", 0.0) * per90_scale
                ),
            )
        )
        xa_per_90 = float(
            player_data.get(
                "xa_per_90",
                player_data.get(
                    "expected_assists", player_data.get("xa", 0.0) * per90_scale
                ),
            )
        )
        goals_per_90 = float(player_data.get("goals_per_90", 0.0))
        assists_per_90 = float(player_data.get("assists_per_90", 0.0))

        # Fill per-90 goals/assists if absent but raw match totals exist
        if goals_per_90 == 0.0 and "goals" in player_data:
            goals_per_90 = float(player_data.get("goals", 0.0)) * per90_scale
        if assists_per_90 == 0.0 and "assists" in player_data:
            assists_per_90 = float(player_data.get("assists", 0.0)) * per90_scale

        # Recent form (last 5 games)
        recent_xg = player_data.get("recent_xg", [])
        recent_xa = player_data.get("recent_xa", [])
        recent_xg_avg = (
            float(np.mean(recent_xg)) if recent_xg and len(recent_xg) > 0 else xg_per_90
        )
        recent_xa_avg = (
            float(np.mean(recent_xa)) if recent_xa and len(recent_xa) > 0 else xa_per_90
        )

        # OPPONENT xGC (Expected Goals Conceded) - KEY FEATURE for normalization
        if opponent_data:
            opponent_xgc = float(
                opponent_data.get(
                    "xgc_per_90", opponent_data.get("expected_goals_conceded", 1.5)
                )
            )
            opponent_defense_strength = float(
                opponent_data.get("defense_strength", 0.0)
            )
        elif fdr_data:
            opponent_defense_strength = float(
                fdr_data.get("opponent_defense_strength", 0.0)
            )
            opponent_xgc = 1.5 - (opponent_defense_strength * 0.1)  # Estimate
        else:
            opponent_xgc = 1.5
            opponent_defense_strength = 0.0

        # Normalize player xG/xA by opponent xGC
        xgc_normalization_factor = opponent_xgc / 1.5
        normalized_xg_per_90 = float(xg_per_90 * xgc_normalization_factor)
        normalized_xa_per_90 = float(xa_per_90 * xgc_normalization_factor)

        # Fixture difficulty
        if fdr_data:
            fdr = float(fdr_data.get("fdr", 3.0))
            opponent_attack = float(fdr_data.get("opponent_attack_strength", 0.0))
        else:
            fdr = 3.0
            opponent_attack = 0.0

        # Home/away
        is_home = 1.0 if fixture_data and fixture_data.get("is_home", True) else 0.0

        # Position
        position = player_data.get("position", "MID")
        position_encoded = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}.get(position, 2)

        # Team attack strength
        team_attack = float(player_data.get("team_attack_strength", 0.0))

        # Expected minutes factor
        expected_minutes = float(player_data.get("expected_minutes", 90.0)) / 90.0

        features = np.array(
            [
                normalized_xg_per_90,
                normalized_xa_per_90,
                xg_per_90,
                xa_per_90,
                goals_per_90,
                assists_per_90,
                recent_xg_avg,
                recent_xa_avg,
                opponent_xgc,
                opponent_defense_strength,
                opponent_attack,
                fdr / 5.0,
                is_home,
                position_encoded / 3.0,
                team_attack,
                expected_minutes,
                xgc_normalization_factor,
            ]
        )

        return features.reshape(1, -1)

    def train(
        self,
        X: np.ndarray,
        y_xg: np.ndarray,
        y_xa: np.ndarray,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        optimize_hyperparameters: bool = False,
    ) -> None:
        """
        Train both xG and xA LightGBM models.

        Args:
            X: Feature matrix
            y_xg: Target xG values
            y_xa: Target xA values
            feature_names: Optional list of feature names
            categorical_features: Optional list of categorical feature names
            validation_data: Optional (X_val, y_xg_val, y_xa_val) tuple for early stopping
            optimize_hyperparameters: Whether to perform grid search
        """
        if not self._loaded:
            self._initialize_empty_model()
            self._loaded = True

        if len(X) == 0 or len(y_xg) == 0 or len(y_xa) == 0:
            logger.error("Cannot train Attack model: empty training data")
            return

        # Handle categorical features for LightGBM
        categorical_indices = None
        if categorical_features and feature_names and LIGHTGBM_AVAILABLE:
            categorical_indices = [
                feature_names.index(feat)
                for feat in categorical_features
                if feat in feature_names
            ]
            if categorical_indices:
                logger.info(
                    f"Treating {len(categorical_indices)} features as categorical: {categorical_features}"
                )

        # Hyperparameter optimization
        if optimize_hyperparameters and LIGHTGBM_AVAILABLE:
            logger.info("Optimizing hyperparameters using grid search...")
            best_params_xg = self._optimize_hyperparameters(
                X, y_xg, categorical_indices, validation_data
            )
            best_params_xa = self._optimize_hyperparameters(
                X, y_xa, categorical_indices, validation_data
            )

            self.xg_model = lgb.LGBMRegressor(
                **best_params_xg, random_state=42, verbose=-1, n_jobs=1
            )
            self.xa_model = lgb.LGBMRegressor(
                **best_params_xa, random_state=42, verbose=-1, n_jobs=1
            )
            logger.info("Hyperparameter optimization complete")

        X_scaled = self.scaler.fit_transform(X)

        # Train with validation set for early stopping if provided
        if validation_data is not None and LIGHTGBM_AVAILABLE:
            X_val, y_xg_val, y_xa_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)

            callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]

            self.xg_model.fit(
                X_scaled,
                y_xg,
                eval_set=[(X_val_scaled, y_xg_val)],
                eval_metric="mae",
                callbacks=callbacks,
                categorical_feature=categorical_indices,
            )

            self.xa_model.fit(
                X_scaled,
                y_xa,
                eval_set=[(X_val_scaled, y_xa_val)],
                eval_metric="mae",
                callbacks=callbacks,
                categorical_feature=categorical_indices,
            )
        else:
            if LIGHTGBM_AVAILABLE and categorical_indices:
                self.xg_model.fit(
                    X_scaled, y_xg, categorical_feature=categorical_indices
                )
                self.xa_model.fit(
                    X_scaled, y_xa, categorical_feature=categorical_indices
                )
            else:
                self.xg_model.fit(X_scaled, y_xg)
                self.xa_model.fit(X_scaled, y_xa)

        self.xg_trained = True
        self.xa_trained = True
        model_type = "LightGBM" if LIGHTGBM_AVAILABLE else "RandomForest"
        logger.info(
            f"Attack model ({model_type} xG/xA) trained successfully on {len(X)} samples"
        )

        # Memory management
        gc.collect()

    def _optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_indices: Optional[List[int]] = None,
        validation_data: Optional[Tuple] = None,
    ) -> Dict:
        """Optimize hyperparameters using grid search."""
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [6, 8, 10],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 50],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        base_model = lgb.LGBMRegressor(
            random_state=42,
            verbose=-1,
            n_jobs=1,
            categorical_feature=categorical_indices,
        )

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            verbose=0,
        )

        X_scaled = self.scaler.fit_transform(X)
        grid_search.fit(X_scaled, y)

        best_params = grid_search.best_params_
        logger.info(f"Best hyperparameters: {best_params}")

        return best_params

    def evaluate(
        self, X: np.ndarray, y_xg: np.ndarray, y_xa: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the Attack model using MAE and RMSE for xG and xA.

        Args:
            X: Feature matrix
            y_xg: Actual xG values
            y_xa: Actual xA values

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.xg_trained or not self.xa_trained:
            logger.warning("Model not trained, cannot evaluate")
            return {
                "xg_mae": float("inf"),
                "xa_mae": float("inf"),
                "xg_rmse": float("inf"),
                "xa_rmse": float("inf"),
            }

        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            X_scaled = self.scaler.transform(X)

            y_xg_pred = self.xg_model.predict(X_scaled)
            y_xa_pred = self.xa_model.predict(X_scaled)

            xg_mae = mean_absolute_error(y_xg, y_xg_pred)
            xa_mae = mean_absolute_error(y_xa, y_xa_pred)
            xg_rmse = np.sqrt(mean_squared_error(y_xg, y_xg_pred))
            xa_rmse = np.sqrt(mean_squared_error(y_xa, y_xa_pred))

            logger.info(
                f"Attack model evaluation: xG MAE={xg_mae:.4f}, xA MAE={xa_mae:.4f}"
            )

            return {
                "xg_mae": float(xg_mae),
                "xa_mae": float(xa_mae),
                "xg_rmse": float(xg_rmse),
                "xa_rmse": float(xa_rmse),
            }
        except Exception as e:
            logger.error(f"Error evaluating Attack model: {str(e)}")
            return {
                "xg_mae": float("inf"),
                "xa_mae": float("inf"),
                "xg_rmse": float("inf"),
                "xa_rmse": float("inf"),
            }

    def predict(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None,
        fdr_data: Optional[Dict] = None,
        opponent_data: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Predict xG and xA for a player.

        Uses opponent xGC to adjust predictions and applies FDR scaling.

        Args:
            player_data: Player statistics
            fixture_data: Optional fixture information
            fdr_data: Optional FDR data
            opponent_data: Optional opponent team data

        Returns:
            Dictionary with 'xg' and 'xa' predictions
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Calculate FDR scaling multiplier
        fdr_multiplier = 1.0
        if fdr_data:
            opponent_defense = float(fdr_data.get("opponent_defense_strength", 0.0))
            if opponent_defense != 0:
                fdr_multiplier = 1.0 - (opponent_defense * 0.3)
                fdr_multiplier = float(np.clip(fdr_multiplier, 0.7, 1.4))
        elif opponent_data:
            opponent_defense = float(opponent_data.get("defense_strength", 0.0))
            if opponent_defense != 0:
                fdr_multiplier = 1.0 - (opponent_defense * 0.3)
                fdr_multiplier = float(np.clip(fdr_multiplier, 0.7, 1.4))

        if not self.xg_trained or not self.xa_trained:
            # Return default based on historical averages
            base_xg = float(player_data.get("xg_per_90", 0.0))
            base_xa = float(player_data.get("xa_per_90", 0.0))

            if opponent_data:
                opponent_xgc = float(opponent_data.get("xgc_per_90", 1.5))
                xgc_factor = opponent_xgc / 1.5
                base_xg = base_xg * xgc_factor

            base_xg = base_xg * fdr_multiplier
            base_xa = base_xa * fdr_multiplier

            return {"xg": float(max(0.0, base_xg)), "xa": float(max(0.0, base_xa))}

        features = self.extract_features(
            player_data, fixture_data, fdr_data, opponent_data
        )
        features_scaled = self.scaler.transform(features)

        xg_pred = self.xg_model.predict(features_scaled)[0]
        xa_pred = self.xa_model.predict(features_scaled)[0]

        # Apply FDR scaling
        xg_pred = xg_pred * fdr_multiplier
        xa_pred = xa_pred * fdr_multiplier

        # Ensure non-negative
        xg_pred = max(0.0, xg_pred)
        xa_pred = max(0.0, xa_pred)

        return {"xg": float(xg_pred), "xa": float(xa_pred)}
