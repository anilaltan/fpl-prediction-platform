"""
Defense Strategy
LightGBM model for predicting clean sheet probability (P_CS) and DefCon points.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import gc
import pickle
import asyncio
import os

# LightGBM for Defense Model
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ..interfaces import ModelInterface

logger = logging.getLogger(__name__)


class DefenseStrategy(ModelInterface):
    """
    LightGBM model for predicting clean sheet probability (P_CS) and DefCon points.

    Predicts P_CS based on team defensive strength and opponent offensive metrics.
    Calculates DefCon_Pts by weighting predicted blocks, tackles, and interceptions.
    Implements ModelInterface for lazy loading and memory management.
    """

    def __init__(self):
        """Initialize Defense strategy with empty model (lazy loaded)."""
        self.pcs_model: Optional[object] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None
        self._loaded: bool = False
        self._model_path: Optional[str] = None
        # Auto-initialize empty model for immediate use
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

                if model_data.get("defense_model"):
                    # Defense model might be stored as the whole object or just the pcs_model
                    defense_model = model_data["defense_model"]
                    if hasattr(defense_model, "pcs_model"):
                        self.pcs_model = defense_model.pcs_model
                        self.scaler = defense_model.scaler
                        self.is_fitted = defense_model.is_fitted
                        self.feature_names = getattr(
                            defense_model, "feature_names", None
                        )
                    else:
                        self.pcs_model = defense_model
                        self.scaler = StandardScaler()
                        self.is_fitted = True
                    logger.info(f"Loaded Defense model from {model_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load Defense model from {model_path}: {str(e)}"
                )
                self._initialize_empty_model()
        else:
            self._initialize_empty_model()

        self._loaded = True

    def _initialize_empty_model(self) -> None:
        """Initialize empty model (not trained)."""
        if LIGHTGBM_AVAILABLE:
            self.pcs_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
                n_jobs=1,  # Memory efficient
                max_bin=255,
                objective="binary",
            )
        else:
            self.pcs_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=1
            )
        self.scaler = StandardScaler()
        self.is_fitted = False

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

        self.pcs_model = None
        self.scaler = None
        self._loaded = False

        # Force garbage collection
        gc.collect()
        logger.debug("Defense model unloaded from memory")

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._loaded and self.pcs_model is not None

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self.is_fitted

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        optimize_hyperparameters: bool = False,
    ) -> None:
        """
        Train LightGBM model for clean sheet probability (P_CS).

        Args:
            X: Feature matrix
            y: Binary labels (1 = clean sheet, 0 = no clean sheet)
            feature_names: Optional list of feature names
            categorical_features: Optional list of categorical feature names
            validation_data: Optional (X_val, y_val) tuple for early stopping
            optimize_hyperparameters: Whether to perform grid search
        """
        if not self._loaded:
            self._initialize_empty_model()
            self._loaded = True

        if len(X) == 0 or len(y) == 0:
            logger.error("Cannot train Defense model: empty training data")
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
            best_params = self._optimize_hyperparameters(
                X, y, categorical_indices, validation_data
            )

            self.pcs_model = lgb.LGBMClassifier(
                **best_params, random_state=42, verbose=-1, n_jobs=1, objective="binary"
            )
            logger.info("Hyperparameter optimization complete")

        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feature_names

        # Train with validation set for early stopping if provided
        if validation_data is not None and LIGHTGBM_AVAILABLE:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)

            callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]

            self.pcs_model.fit(
                X_scaled,
                y,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric="binary_logloss",
                callbacks=callbacks,
                categorical_feature=categorical_indices,
            )
        else:
            if LIGHTGBM_AVAILABLE and categorical_indices:
                self.pcs_model.fit(X_scaled, y, categorical_feature=categorical_indices)
            else:
                self.pcs_model.fit(X_scaled, y)

        self.is_fitted = True
        model_type = "LightGBM" if LIGHTGBM_AVAILABLE else "RandomForest"
        logger.info(
            f"Defense model ({model_type} P_CS) trained successfully on {len(X)} samples"
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

        base_model = lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            n_jobs=1,
            objective="binary",
            categorical_feature=categorical_indices,
        )

        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring="neg_log_loss", n_jobs=1, verbose=0
        )

        X_scaled = self.scaler.fit_transform(X)
        grid_search.fit(X_scaled, y)

        best_params = grid_search.best_params_
        logger.info(f"Best hyperparameters: {best_params}")

        return best_params

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the Defense model using Log Loss, AUC-ROC, and accuracy.

        Args:
            X: Feature matrix
            y: Binary labels (1 = clean sheet, 0 = no clean sheet)

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            logger.warning("Model not trained, cannot evaluate")
            return {
                "log_loss": float("inf"),
                "auc_roc": 0.0,
                "accuracy": 0.0,
                "predicted_cs_rate": 0.0,
                "actual_cs_rate": 0.0,
            }

        try:
            from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

            X_scaled = self.scaler.transform(X)

            y_pred_proba = self.pcs_model.predict_proba(X_scaled)[:, 1]
            y_pred = self.pcs_model.predict(X_scaled)

            log_loss_score = log_loss(y, y_pred_proba)
            auc_roc_score = roc_auc_score(y, y_pred_proba)
            accuracy = accuracy_score(y, y_pred)

            predicted_cs_rate = float(np.mean(y_pred_proba))
            actual_cs_rate = float(np.mean(y))

            logger.info(
                f"Defense model evaluation: Log Loss={log_loss_score:.4f}, AUC-ROC={auc_roc_score:.4f}, Accuracy={accuracy:.4f}"
            )

            return {
                "log_loss": float(log_loss_score),
                "auc_roc": float(auc_roc_score),
                "accuracy": float(accuracy),
                "predicted_cs_rate": predicted_cs_rate,
                "actual_cs_rate": actual_cs_rate,
            }
        except Exception as e:
            logger.error(f"Error evaluating Defense model: {str(e)}")
            return {
                "log_loss": float("inf"),
                "auc_roc": 0.0,
                "accuracy": 0.0,
                "predicted_cs_rate": 0.0,
                "actual_cs_rate": 0.0,
            }

    def predict(
        self,
        player_data: Optional[Dict] = None,
        team_data: Optional[Dict] = None,
        opponent_data: Optional[Dict] = None,
        features: Optional[np.ndarray] = None,
        is_home: bool = True,
    ) -> Dict[str, float]:
        """
        Predict clean sheet probability and DefCon points.

        Args:
            player_data: Optional player data with DefCon features
            team_data: Defending team data
            opponent_data: Attacking team data
            features: Optional pre-computed feature array
            is_home: Whether defending team is at home

        Returns:
            Dictionary with 'xcs' and optionally 'defcon_points'
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        xcs = self.predict_clean_sheet_probability(
            player_data=player_data,
            team_data=team_data,
            opponent_data=opponent_data,
            features=features,
            is_home=is_home,
        )

        result = {"xcs": xcs}

        # Calculate DefCon points if player_data provided
        if player_data:
            position = player_data.get("position", "MID")
            expected_minutes = player_data.get("expected_minutes", 90.0)
            defcon_points = self.calculate_defcon_points(
                player_data, position, expected_minutes
            )
            result["defcon_points"] = defcon_points

        return result

    def predict_clean_sheet_probability(
        self,
        player_data: Optional[Dict] = None,
        team_data: Optional[Dict] = None,
        opponent_data: Optional[Dict] = None,
        features: Optional[np.ndarray] = None,
        is_home: bool = True,
    ) -> float:
        """
        Predict clean sheet probability (P_CS) using LightGBM model.
        Falls back to Poisson calculation if model not trained.

        Args:
            player_data: Optional player data with DefCon features
            team_data: Defending team data (with defense strength)
            opponent_data: Attacking team data (with attack strength)
            features: Optional pre-computed feature array
            is_home: Whether defending team is at home

        Returns:
            Clean sheet probability (0-1)
        """
        if self.is_fitted and features is not None:
            # Use trained LightGBM model
            features_scaled = self.scaler.transform(features)
            prob = self.pcs_model.predict_proba(features_scaled)[0][1]
            return float(np.clip(prob, 0.0, 1.0))

        # Fallback to Poisson calculation if model not trained
        if team_data and opponent_data:
            # Get team defense strength
            team_defense = float(team_data.get("defense_strength", 1.0))
            if is_home:
                team_defense = float(
                    team_data.get("strength_defence_home", team_defense)
                )
            else:
                team_defense = float(
                    team_data.get("strength_defence_away", team_defense)
                )

            # Get opponent attack strength
            opponent_attack = float(opponent_data.get("attack_strength", 1.0))
            if is_home:
                opponent_attack = float(
                    opponent_data.get("strength_attack_away", opponent_attack)
                )
            else:
                opponent_attack = float(
                    opponent_data.get("strength_attack_home", opponent_attack)
                )

            # Calculate expected goals conceded (λ)
            base_lambda = 1.5

            def _norm_strength(v: float) -> float:
                v = float(v)
                if v <= 0:
                    return 1.0
                if v <= 10.0:
                    return max(0.1, min(v, 2.0))
                return max(0.1, v / 1000.0)

            defense_factor = _norm_strength(team_defense)
            attack_factor = _norm_strength(opponent_attack)
            home_factor = 0.9 if is_home else 1.0

            lambda_value = (
                base_lambda * (1.0 / defense_factor) * attack_factor * home_factor
            )
            lambda_value = float(np.clip(lambda_value, 0.0, 3.0))

            # Poisson probability: P(X=0) = e^(-λ)
            xcs = np.exp(-lambda_value)
            return float(np.clip(xcs, 0.0, 1.0))

        # Default fallback
        return 0.25  # Average clean sheet probability

    def calculate_defcon_points(
        self, player_data: Dict, position: str, expected_minutes: float = 90.0
    ) -> float:
        """
        Calculate DefCon points by weighting predicted blocks, tackles, and interceptions.

        FPL 2025/26 scoring rules:
        - Blocks: 1 point per block (DEF/MID only)
        - Interventions (tackles/interceptions): 1 point per intervention
        - Passes: 0.1 points per 10 successful passes

        Args:
            player_data: Player data with DefCon features
            position: Player position (GK, DEF, MID, FWD)
            expected_minutes: Expected minutes played

        Returns:
            DefCon points estimate
        """
        if expected_minutes == 0:
            return 0.0

        # Get DefCon stats (per-90 averages)
        blocks_per_90 = float(
            player_data.get("blocks_per_90", player_data.get("avg_blocks", 0.0))
        )
        interventions_per_90 = float(
            player_data.get(
                "interventions_per_90", player_data.get("avg_interventions", 0.0)
            )
        )
        passes_per_90 = float(
            player_data.get("passes_per_90", player_data.get("avg_passes", 0.0))
        )

        # Scale to expected minutes
        minutes_factor = expected_minutes / 90.0

        # Calculate points based on position
        position_lower = position.lower()

        # Blocks: only for DEF and MID
        block_points = 0.0
        if "def" in position_lower or "mid" in position_lower:
            block_points = blocks_per_90 * 1.0 * minutes_factor

        # Interventions: all positions
        intervention_points = interventions_per_90 * 1.0 * minutes_factor

        # Pass bonus: 0.1 points per 10 successful passes
        pass_bonus = (passes_per_90 / 10.0) * 0.1 * minutes_factor

        defcon_points = block_points + intervention_points + pass_bonus

        return float(max(0.0, defcon_points))

    def calculate_expected_goals_conceded(
        self, team_data: Dict, opponent_data: Dict, is_home: bool = True
    ) -> float:
        """
        Calculate expected goals conceded (λ) for Poisson calculation.

        Args:
            team_data: Defending team data
            opponent_data: Attacking team data
            is_home: Whether defending team is at home

        Returns:
            Expected goals conceded (λ)
        """
        team_defense = float(team_data.get("defense_strength", 1.0))
        if is_home:
            team_defense = float(team_data.get("strength_defence_home", team_defense))
        else:
            team_defense = float(team_data.get("strength_defence_away", team_defense))

        opponent_attack = float(opponent_data.get("attack_strength", 1.0))
        if is_home:
            opponent_attack = float(
                opponent_data.get("strength_attack_away", opponent_attack)
            )
        else:
            opponent_attack = float(
                opponent_data.get("strength_attack_home", opponent_attack)
            )

        base_lambda = 1.5

        def _norm_strength(v: float) -> float:
            v = float(v)
            if v <= 0:
                return 1.0
            if v <= 10.0:
                return max(0.1, min(v, 2.0))
            return max(0.1, v / 1000.0)

        defense_factor = _norm_strength(team_defense)
        attack_factor = _norm_strength(opponent_attack)
        home_factor = 0.9 if is_home else 1.0

        lambda_value = (
            base_lambda * (1.0 / defense_factor) * attack_factor * home_factor
        )

        return float(np.clip(lambda_value, 0.0, 3.0))
