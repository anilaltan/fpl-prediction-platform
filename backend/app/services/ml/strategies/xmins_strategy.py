"""
XMins Strategy
XGBoost-based model for predicting starting 11 probability and expected minutes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import gc
import pickle
import asyncio
import os

# XGBoost for xMins
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, using Random Forest for xMins")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Handle both relative and absolute imports for script/module compatibility
try:
    from ..interfaces import ModelInterface
    from ..calculations import (
        calculate_rolling_average,
        calculate_lag_feature,
        calculate_clean_sheet_rate,
        pad_list,
    )
except ImportError:
    # Fallback for direct script execution
    import sys
    import os
    from pathlib import Path
    # Add backend directory to path (works in Docker where /app is the backend root)
    current_file = Path(__file__).resolve()
    # Try /app first (Docker container), then fallback to relative path
    if os.path.exists("/app"):
        backend_dir = Path("/app")
    else:
        backend_dir = current_file.parent.parent.parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from app.services.ml.interfaces import ModelInterface
    from app.services.ml.calculations import (
        calculate_rolling_average,
        calculate_lag_feature,
        calculate_clean_sheet_rate,
        pad_list,
    )

logger = logging.getLogger(__name__)


class XMinsStrategy(ModelInterface):
    """
    XGBoost Classifier-based model for predicting starting 11 probability (P_start).

    Key features: days_since_last_match, is_cup_week, DefCon features, and lag features.
    Implements ModelInterface for lazy loading and memory management.
    """

    def __init__(self):
        """
        Initialize XMins strategy with empty model.

        For backward compatibility, models are initialized immediately
        (not lazy loaded). Use load() for async loading from file.
        """
        self._initialize_empty_model()
        self.feature_names: Optional[List[str]] = None
        self._loaded: bool = False  # Start as False, will be set to True after load()
        self._model_path: Optional[str] = None

    async def load(self, model_path: Optional[str] = None) -> None:
        """
        Load trained model from file.

        Args:
            model_path: Optional path to model file. If None, keeps current empty model.
        """
        logger.info(f"[XMinsStrategy.load] Starting load, model_path={model_path}")
        
        if not model_path:
            logger.warning("[XMinsStrategy.load] No model path provided, keeping empty model")
            return
        
        if not os.path.exists(model_path):
            error_msg = f"Model file does not exist: {model_path}"
            logger.error(f"[XMinsStrategy.load] {error_msg}")
            raise FileNotFoundError(error_msg)

        self._model_path = model_path
        logger.info(f"[XMinsStrategy.load] Model file exists, loading pickle from {model_path}")

        try:
            # Load from pickle file asynchronously
            # Use get_running_loop() if available, otherwise get_event_loop()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            logger.info(f"[XMinsStrategy.load] Running pickle load in executor...")
            model_data = await loop.run_in_executor(None, self._load_pickle, model_path)
            logger.info(f"[XMinsStrategy.load] Pickle loaded, type: {type(model_data)}, keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'not a dict'}")

            xmins_model = model_data.get("xmins_model")
            logger.info(f"[XMinsStrategy.load] xmins_model extracted: {xmins_model is not None}, type: {type(xmins_model) if xmins_model is not None else 'None'}")
            
            if xmins_model is not None:
                # Set model first, then flags, to ensure atomic state update
                self.model = xmins_model
                self.scaler = model_data.get("xmins_scaler", StandardScaler())
                self.feature_names = model_data.get("xmins_feature_names")
                # Set flags after model is set to ensure consistency
                self._is_trained = True
                self._loaded = True
                logger.info(f"[XMinsStrategy.load] Successfully loaded xMins model from {model_path} (type: {type(xmins_model).__name__})")
                logger.info(f"[XMinsStrategy.load] State after load: is_trained={self._is_trained}, is_loaded={self._loaded}, model={'exists' if self.model is not None else 'None'}")
                # Double-check that model is actually set
                if self.model is None:
                    raise RuntimeError("Model was set to None after assignment - this should not happen")
            else:
                available_keys = list(model_data.keys()) if isinstance(model_data, dict) else "not a dict"
                error_msg = (
                    f"No xmins_model found in {model_path} or model is None. "
                    f"Available keys: {available_keys}. "
                    f"xmins_model value: {xmins_model}"
                )
                logger.error(f"[XMinsStrategy.load] {error_msg}")
                # Reset flags to ensure we don't have inconsistent state
                self._is_trained = False
                self._loaded = False
                self.model = None
                # Raise exception to signal load failure
                raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"[XMinsStrategy.load] Failed to load xMins model from {model_path}: {str(e)}", exc_info=True)
            # Reset flags on error
            self._is_trained = False
            self._loaded = False
            self.model = None
            # Re-raise exception to signal load failure
            raise

    def _initialize_empty_model(self) -> None:
        """Initialize empty model (not trained)."""
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                eval_metric="logloss",
                n_jobs=1,  # Memory efficient for 4GB RAM
                tree_method="hist",  # Memory efficient
                subsample=0.8,
                colsample_bytree=0.8,
            )
        else:
            logger.warning(
                "XGBoost not available, falling back to RandomForestClassifier"
            )
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=1,
                max_samples=0.8,
            )
        self.scaler = StandardScaler()
        self._is_trained = False

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

        self.model = None
        self.scaler = None
        self._loaded = False

        # Force garbage collection
        gc.collect()
        logger.debug("xMins model unloaded from memory")

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._loaded and self.model is not None

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        """Set trained status."""
        self._is_trained = value

    def extract_features(
        self, player_data: Dict, fixture_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Extract features for xMins prediction.
        
        Must match the 24 features used during training:
        - price (1)
        - DefCon features (7): blocks_per_90, interventions_per_90, passes_per_90, 
          defcon_floor_points, avg_blocks, avg_interventions, avg_passes
        - Lag features (16): xg_lag_1, xg_lag_3, xg_lag_5, xg_rolling_3, xg_rolling_5,
          xa_lag_1, xa_lag_3, xa_lag_5, xa_rolling_3, xa_rolling_5,
          cs_lag_1, cs_lag_3, cs_lag_5, cs_rolling_3, cs_rolling_5, cs_rate
        """
        # 1. Price (normalized to 0-1 range, typically 40-150 -> 0.4-1.5, but we'll use raw/100)
        price = float(player_data.get("price", 50.0))
        
        # 2. DefCon features (7 features)
        blocks_per_90 = float(player_data.get("blocks_per_90", 0.0))
        interventions_per_90 = float(player_data.get("interventions_per_90", 0.0))
        passes_per_90 = float(player_data.get("passes_per_90", 0.0))
        defcon_floor_points = float(player_data.get("defcon_floor_points", 0.0))
        avg_blocks = float(player_data.get("avg_blocks", 0.0))
        avg_interventions = float(player_data.get("avg_interventions", 0.0))
        avg_passes = float(player_data.get("avg_passes", 0.0))
        
        # 3. Lag features (16 features) - extract from recent stats using pure functions
        recent_xg = pad_list(player_data.get("recent_xg", []), 5)
        recent_xa = pad_list(player_data.get("recent_xa", []), 5)
        recent_cs = pad_list(player_data.get("recent_cs", []), 5)  # Clean sheets
        recent_minutes = player_data.get("recent_minutes", [])
        
        # xG lag features (1, 3, 5)
        xg_lag_1 = calculate_lag_feature(recent_xg, 1)
        xg_lag_3 = calculate_lag_feature(recent_xg, 3)
        xg_lag_5 = calculate_lag_feature(recent_xg, 5)
        
        # xG rolling averages (3, 5)
        xg_rolling_3 = calculate_rolling_average(recent_xg, 3)
        xg_rolling_5 = calculate_rolling_average(recent_xg, 5)
        
        # xA lag features (1, 3, 5)
        xa_lag_1 = calculate_lag_feature(recent_xa, 1)
        xa_lag_3 = calculate_lag_feature(recent_xa, 3)
        xa_lag_5 = calculate_lag_feature(recent_xa, 5)
        
        # xA rolling averages (3, 5)
        xa_rolling_3 = calculate_rolling_average(recent_xa, 3)
        xa_rolling_5 = calculate_rolling_average(recent_xa, 5)
        
        # Clean sheet lag features (1, 3, 5)
        cs_lag_1 = calculate_lag_feature(recent_cs, 1)
        cs_lag_3 = calculate_lag_feature(recent_cs, 3)
        cs_lag_5 = calculate_lag_feature(recent_cs, 5)
        
        # Clean sheet rolling averages (3, 5)
        cs_rolling_3 = calculate_rolling_average(recent_cs, 3)
        cs_rolling_5 = calculate_rolling_average(recent_cs, 5)
        
        # Clean sheet rate (percentage of games with clean sheet)
        cs_rate = calculate_clean_sheet_rate(recent_cs, recent_minutes)
        
        # Build feature array in the same order as training: price, defcon (7), lag (16)
        features = np.array([
            price,  # 1
            blocks_per_90,  # 2
            interventions_per_90,  # 3
            passes_per_90,  # 4
            defcon_floor_points,  # 5
            avg_blocks,  # 6
            avg_interventions,  # 7
            avg_passes,  # 8
            xg_lag_1,  # 9
            xg_lag_3,  # 10
            xg_lag_5,  # 11
            xg_rolling_3,  # 12
            xg_rolling_5,  # 13
            xa_lag_1,  # 14
            xa_lag_3,  # 15
            xa_lag_5,  # 16
            xa_rolling_3,  # 17
            xa_rolling_5,  # 18
            cs_lag_1,  # 19
            cs_lag_3,  # 20
            cs_lag_5,  # 21
            cs_rolling_3,  # 22
            cs_rolling_5,  # 23
            cs_rate,  # 24
        ], dtype=np.float32)

        return features.reshape(1, -1)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """
        Train the xMins XGBoost Classifier.

        Args:
            X: Feature matrix
            y: Binary labels (1 = started, 0 = didn't start)
            feature_names: Optional list of feature names
            validation_data: Optional (X_val, y_val) tuple for early stopping
        """
        if not self._loaded:
            self._initialize_empty_model()
            self._loaded = True

        if len(X) == 0 or len(y) == 0:
            logger.error("Cannot train xMins model: empty training data")
            return

        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feature_names

        # Train with validation set for early stopping if provided
        if validation_data is not None and XGBOOST_AVAILABLE:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_scaled,
                y,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=10,
                verbose=False,
            )
        else:
            self.model.fit(X_scaled, y)

        self._is_trained = True
        model_type = (
            "XGBoost Classifier" if XGBOOST_AVAILABLE else "RandomForestClassifier"
        )
        logger.info(
            f"xMins model ({model_type}) trained successfully on {len(X)} samples"
        )

        # Memory management
        gc.collect()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the xMins model using Log Loss and AUC-ROC metrics.

        Args:
            X: Feature matrix
            y: Binary labels (1 = started, 0 = didn't start)

        Returns:
            Dictionary with evaluation metrics:
            - log_loss: Log Loss score
            - auc_roc: AUC-ROC score
            - accuracy: Classification accuracy
        """
        if not self._is_trained:
            logger.warning("Model not trained, cannot evaluate")
            return {"log_loss": float("inf"), "auc_roc": 0.0, "accuracy": 0.0}

        try:
            from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

            X_scaled = self.scaler.transform(X)

            # Predict probabilities
            y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
            y_pred = self.model.predict(X_scaled)

            # Calculate metrics
            log_loss_score = log_loss(y, y_pred_proba)
            auc_roc_score = roc_auc_score(y, y_pred_proba)
            accuracy = accuracy_score(y, y_pred)

            logger.info(
                f"xMins model evaluation: Log Loss={log_loss_score:.4f}, AUC-ROC={auc_roc_score:.4f}, Accuracy={accuracy:.4f}"
            )

            return {
                "log_loss": float(log_loss_score),
                "auc_roc": float(auc_roc_score),
                "accuracy": float(accuracy),
            }
        except Exception as e:
            logger.error(f"Error evaluating xMins model: {str(e)}")
            return {"log_loss": float("inf"), "auc_roc": 0.0, "accuracy": 0.0}

    def predict(
        self, player_data: Dict, fixture_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Predict starting probability and expected minutes.

        Args:
            player_data: Player statistics
            fixture_data: Optional fixture information

        Returns:
            Dictionary with 'p_start' and 'expected_minutes'
        """
        # Check if model is available (either loaded or trained with model object)
        # Allow prediction if model is trained and model object exists
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not loaded or not trained. Call load() first.")

        p_start = self.predict_start_probability(player_data, fixture_data)
        expected_minutes = self.predict_expected_minutes(player_data, fixture_data)

        return {"p_start": p_start, "expected_minutes": expected_minutes}

    def predict_start_probability(
        self, player_data: Dict, fixture_data: Optional[Dict] = None
    ) -> float:
        """
        Predict probability of starting (P_start).

        Returns:
            Probability between 0 and 1
        """
        if not self._is_trained:
            # Default prediction based on recent minutes
            recent_minutes = player_data.get("recent_minutes", [])
            if recent_minutes and len(recent_minutes) > 0:
                slice_mins = (
                    recent_minutes[:3] if len(recent_minutes) >= 3 else recent_minutes
                )
                avg_minutes = (
                    float(np.mean(slice_mins)) if len(slice_mins) > 0 else 63.0
                )
                return min(1.0, avg_minutes / 90.0)
            return 0.7

        features = self.extract_features(player_data, fixture_data)
        features_scaled = self.scaler.transform(features)

        # Predict probability
        prob = self.model.predict_proba(features_scaled)[0][1]

        return float(np.clip(prob, 0.0, 1.0))

    def predict_expected_minutes(
        self, player_data: Dict, fixture_data: Optional[Dict] = None
    ) -> float:
        """
        Predict expected minutes based on start probability.

        Returns:
            Expected minutes (0-90)
        """
        p_start = self.predict_start_probability(player_data, fixture_data)

        # If starting, estimate minutes based on recent average
        recent_minutes = player_data.get("recent_minutes", [90])
        valid_minutes = [m for m in recent_minutes if m > 0] if recent_minutes else []
        avg_minutes_when_starting = (
            float(np.mean(valid_minutes)) if len(valid_minutes) > 0 else 85.0
        )

        # Expected minutes = P(start) * avg_minutes_when_starting
        expected_minutes = p_start * avg_minutes_when_starting

        return float(np.clip(expected_minutes, 0.0, 90.0))
