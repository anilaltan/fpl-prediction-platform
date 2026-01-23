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

from ..interfaces import ModelInterface

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
        self._loaded: bool = True  # Models are initialized, so considered "loaded"
        self._model_path: Optional[str] = None

    async def load(self, model_path: Optional[str] = None) -> None:
        """
        Load trained model from file.

        Args:
            model_path: Optional path to model file. If None, keeps current empty model.
        """
        if not model_path or not os.path.exists(model_path):
            return

        self._model_path = model_path

        try:
            # Load from pickle file asynchronously
            loop = asyncio.get_event_loop()
            model_data = await loop.run_in_executor(None, self._load_pickle, model_path)

            if model_data.get("xmins_model"):
                self.model = model_data["xmins_model"]
                self.scaler = model_data.get("xmins_scaler", StandardScaler())
                self._is_trained = True
                self.feature_names = model_data.get("xmins_feature_names")
                logger.info(f"Loaded xMins model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load xMins model from {model_path}: {str(e)}")

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

        Features:
        - days_since_last_match: Days since last match (PRIMARY FEATURE)
        - is_cup_week: Binary (1 if cup match week, 0 otherwise) (PRIMARY FEATURE)
        - injury_status: 0=fit, 1=doubtful, 2=out
        - recent_minutes_avg: Average minutes in last 3 matches
        - position_depth: Squad depth at position (1-3)
        - form_score: Recent form
        - price: Player price (proxy for importance)
        """
        # PRIMARY FEATURE 1: Days since last match
        last_match_date = player_data.get("last_match_date")
        if last_match_date:
            if isinstance(last_match_date, str):
                try:
                    last_match = datetime.fromisoformat(
                        last_match_date.replace("Z", "+00:00")
                    )
                    days_since_last_match = (
                        datetime.now() - last_match.replace(tzinfo=None)
                    ).days
                except Exception:
                    days_since_last_match = 7
            else:
                days_since_last_match = (
                    datetime.now() - last_match_date.replace(tzinfo=None)
                ).days
        else:
            # Try to calculate from recent matches
            recent_matches = player_data.get("recent_matches", [])
            if recent_matches:
                last_match_date_str = (
                    recent_matches[0].get("date")
                    if isinstance(recent_matches[0], dict)
                    else None
                )
                if last_match_date_str:
                    try:
                        last_match = datetime.fromisoformat(
                            last_match_date_str.replace("Z", "+00:00")
                        )
                        days_since_last_match = (
                            datetime.now() - last_match.replace(tzinfo=None)
                        ).days
                    except Exception:
                        days_since_last_match = 7
                else:
                    days_since_last_match = 7
            else:
                days_since_last_match = 7  # Default: full week rest

        # PRIMARY FEATURE 2: is_cup_week
        is_cup_week = 0
        if fixture_data:
            has_cup_match = fixture_data.get("has_cup_match", False)
            is_midweek = fixture_data.get("is_midweek", False)
            is_cup_week = 1 if (has_cup_match or is_midweek) else 0
        else:
            is_cup_week = 1 if player_data.get("is_cup_week", False) else 0

        # Injury status (0=fit, 1=doubtful, 2=out)
        injury_status_map = {"a": 0, "d": 1, "i": 2, "n": 0, "s": 2}
        status = player_data.get("status", "a").lower()
        injury_status = injury_status_map.get(status, 0)

        # Recent minutes average
        recent_minutes = player_data.get("recent_minutes", [])
        if recent_minutes and len(recent_minutes) > 0:
            slice_mins = (
                recent_minutes[:3] if len(recent_minutes) >= 3 else recent_minutes
            )
            recent_minutes_avg = (
                float(np.mean(slice_mins)) if len(slice_mins) > 0 else 90.0
            )
        else:
            recent_minutes_avg = player_data.get("minutes_per_game", 90.0)

        # Position depth
        position_depth = player_data.get("position_depth", 2.0)

        # Form score
        form_score = player_data.get("form", 0.0)

        # Price (normalized)
        price = player_data.get("price", 50.0) / 100.0

        # Team rotation risk
        rotation_risk = player_data.get("rotation_risk", 0.5)

        features = np.array(
            [
                days_since_last_match,
                is_cup_week,
                injury_status,
                recent_minutes_avg / 90.0,
                position_depth / 3.0,
                form_score / 10.0,
                price,
                rotation_risk,
            ]
        )

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
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

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
