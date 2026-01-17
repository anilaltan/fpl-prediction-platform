"""
Enhanced ML Engine for FPL Point Prediction
Implements component-based predictive models with memory management:
- xMins Model: RandomForestClassifier for starting 11 probability
- Attack Model: LightGBM for xG/xA predictions with opponent xGC normalization
- Defense Model: Poisson Regression for clean sheet probability
- Final xP Calculation: Comprehensive expected points formula
- Resource Management: Async pickle load/unload with gc.collect()
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import gc
import joblib
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

# LightGBM for Attack Model
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

# Scipy for Poisson
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from app.services.feature_engineering import FeatureEngineeringService
from app.services.data_cleaning import DataCleaningService

logger = logging.getLogger(__name__)


class XMinsModel:
    """
    RandomForestClassifier-based model for predicting starting 11 probability (P_start).
    Key features: days_since_last_match and is_cup_week.
    """
    
    def __init__(self):
        # Use RandomForestClassifier as specified
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=1,  # Memory efficient for 4GB RAM
            max_samples=0.8  # Memory efficient
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Extract features for xMins prediction.
        Key features: days_since_last_match and is_cup_week.
        
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
        last_match_date = player_data.get('last_match_date')
        if last_match_date:
            if isinstance(last_match_date, str):
                try:
                    last_match = datetime.fromisoformat(last_match_date.replace('Z', '+00:00'))
                    days_since_last_match = (datetime.now() - last_match.replace(tzinfo=None)).days
                except:
                    days_since_last_match = 7
            else:
                days_since_last_match = (datetime.now() - last_match_date.replace(tzinfo=None)).days
        else:
            # Try to calculate from recent matches
            recent_matches = player_data.get('recent_matches', [])
            if recent_matches:
                last_match_date_str = recent_matches[0].get('date') if isinstance(recent_matches[0], dict) else None
                if last_match_date_str:
                    try:
                        last_match = datetime.fromisoformat(last_match_date_str.replace('Z', '+00:00'))
                        days_since_last_match = (datetime.now() - last_match.replace(tzinfo=None)).days
                    except:
                        days_since_last_match = 7
                else:
                    days_since_last_match = 7
            else:
                days_since_last_match = 7  # Default: full week rest
        
        # PRIMARY FEATURE 2: is_cup_week
        # Check if there's a cup match this week (midweek fixture)
        is_cup_week = 0
        if fixture_data:
            # Check if there's a midweek match or cup fixture
            has_cup_match = fixture_data.get('has_cup_match', False)
            is_midweek = fixture_data.get('is_midweek', False)
            is_cup_week = 1 if (has_cup_match or is_midweek) else 0
        else:
            # Check player data for cup week indicator
            is_cup_week = 1 if player_data.get('is_cup_week', False) else 0
        
        # Injury status (0=fit, 1=doubtful, 2=out)
        injury_status_map = {'a': 0, 'd': 1, 'i': 2, 'n': 0, 's': 2}
        status = player_data.get('status', 'a').lower()
        injury_status = injury_status_map.get(status, 0)
        
        # Recent minutes average (safe handling for empty lists)
        recent_minutes = player_data.get('recent_minutes', [])
        if recent_minutes and len(recent_minutes) > 0:
            slice_mins = recent_minutes[:3] if len(recent_minutes) >= 3 else recent_minutes
            recent_minutes_avg = float(np.mean(slice_mins)) if len(slice_mins) > 0 else 90.0
        else:
            recent_minutes_avg = player_data.get('minutes_per_game', 90.0)
        
        # Position depth (squad depth - lower = more depth)
        position_depth = player_data.get('position_depth', 2.0)
        
        # Form score
        form_score = player_data.get('form', 0.0)
        
        # Price (normalized)
        price = player_data.get('price', 50.0) / 100.0  # Normalize to 0-1
        
        # Team rotation risk (higher = more rotation)
        rotation_risk = player_data.get('rotation_risk', 0.5)
        
        features = np.array([
            days_since_last_match,  # PRIMARY FEATURE 1
            is_cup_week,  # PRIMARY FEATURE 2
            injury_status,
            recent_minutes_avg / 90.0,  # Normalize
            position_depth / 3.0,  # Normalize
            form_score / 10.0,  # Normalize
            price,
            rotation_risk
        ])
        
        return features.reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the xMins RandomForestClassifier.
        
        Args:
            X: Feature matrix (must include days_since_last_match and is_cup_week as first two features)
            y: Binary labels (1 = started, 0 = didn't start)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("xMins model (RandomForestClassifier) trained successfully")
        
        # Memory management
        gc.collect()
    
    def predict_start_probability(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None
    ) -> float:
        """
        Predict probability of starting (P_start).
        
        Returns:
            Probability between 0 and 1
        """
        if not self.is_trained:
            # Default prediction based on recent minutes (safe handling for empty lists)
            recent_minutes = player_data.get('recent_minutes', [])
            if recent_minutes and len(recent_minutes) > 0:
                slice_mins = recent_minutes[:3] if len(recent_minutes) >= 3 else recent_minutes
                avg_minutes = float(np.mean(slice_mins)) if len(slice_mins) > 0 else 63.0
                return min(1.0, avg_minutes / 90.0)
            return 0.7  # Default
        
        features = self.extract_features(player_data, fixture_data)
        features_scaled = self.scaler.transform(features)
        
        # Predict probability
        prob = self.model.predict_proba(features_scaled)[0][1]
        
        return float(np.clip(prob, 0.0, 1.0))
    
    def predict_expected_minutes(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None
    ) -> float:
        """
        Predict expected minutes based on start probability.
        
        Returns:
            Expected minutes (0-90)
        """
        p_start = self.predict_start_probability(player_data, fixture_data)
        
        # If starting, estimate minutes based on recent average (safe handling for empty lists)
        recent_minutes = player_data.get('recent_minutes', [90])
        valid_minutes = [m for m in recent_minutes if m > 0] if recent_minutes else []
        avg_minutes_when_starting = float(np.mean(valid_minutes)) if len(valid_minutes) > 0 else 85.0
        
        # Expected minutes = P(start) * avg_minutes_when_starting
        expected_minutes = p_start * avg_minutes_when_starting
        
        return float(np.clip(expected_minutes, 0.0, 90.0))


class AttackModel:
    """
    LightGBM model for predicting xG and xA.
    Combines player xG/xA data with opponent xGC (Expected Goals Conceded).
    """
    
    def __init__(self):
        if LIGHTGBM_AVAILABLE:
            self.xg_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
                n_jobs=1,  # Memory efficient
                max_bin=255
            )
            self.xa_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
                n_jobs=1,  # Memory efficient
                max_bin=255
            )
        else:
            self.xg_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=1
            )
            self.xa_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=1
            )
        
        self.scaler = StandardScaler()
        self.xg_trained = False
        self.xa_trained = False
    
    def extract_features(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None,
        fdr_data: Optional[Dict] = None,
        opponent_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Extract features for xG/xA prediction.
        Combines player xG/xA with opponent xGC.
        
        Features:
        - Player historical xG/xA per 90
        - Opponent xGC (Expected Goals Conceded) - KEY FEATURE
        - Home/away advantage
        - Recent form
        - Fixture difficulty
        """
        # Player historical stats
        # NOTE: backtest player rows often contain `xg`/`xa` and `minutes` (not *_per_90).
        # Provide robust fallbacks so prediction doesn't collapse to all-zeros due to missing keys.
        minutes = float(player_data.get('minutes', 0) or 0)
        per90_scale = 90.0 / max(minutes, 1.0)

        xg_per_90 = float(
            player_data.get(
                'xg_per_90',
                player_data.get('expected_goals', player_data.get('xg', 0.0) * per90_scale),
            )
        )
        xa_per_90 = float(
            player_data.get(
                'xa_per_90',
                player_data.get('expected_assists', player_data.get('xa', 0.0) * per90_scale),
            )
        )
        goals_per_90 = float(player_data.get('goals_per_90', 0.0))
        assists_per_90 = float(player_data.get('assists_per_90', 0.0))

        # Fill per-90 goals/assists if absent but raw match totals exist
        if goals_per_90 == 0.0 and 'goals' in player_data:
            goals_per_90 = float(player_data.get('goals', 0.0)) * per90_scale
        if assists_per_90 == 0.0 and 'assists' in player_data:
            assists_per_90 = float(player_data.get('assists', 0.0)) * per90_scale
        
        # Recent form (last 5 games) - safe handling for empty lists
        recent_xg = player_data.get('recent_xg', [])
        recent_xa = player_data.get('recent_xa', [])
        recent_xg_avg = float(np.mean(recent_xg)) if recent_xg and len(recent_xg) > 0 else xg_per_90
        recent_xa_avg = float(np.mean(recent_xa)) if recent_xa and len(recent_xa) > 0 else xa_per_90
        
        # OPPONENT xGC (Expected Goals Conceded) - KEY FEATURE for normalization
        if opponent_data:
            opponent_xgc = float(opponent_data.get('xgc_per_90', opponent_data.get('expected_goals_conceded', 1.5)))
            opponent_defense_strength = float(opponent_data.get('defense_strength', 0.0))
        elif fdr_data:
            # Estimate xGC from defense strength (lower defense strength = higher xGC)
            opponent_defense_strength = float(fdr_data.get('opponent_defense_strength', 0.0))
            # Normalize: average xGC is 1.5, adjust based on defense strength
            opponent_xgc = 1.5 - (opponent_defense_strength * 0.1)  # Estimate
        else:
            opponent_xgc = 1.5  # Default: average team concedes ~1.5 goals per game
            opponent_defense_strength = 0.0
        
        # Normalize player xG/xA by opponent xGC
        # Higher opponent xGC = easier to score = boost xG/xA
        xgc_normalization_factor = opponent_xgc / 1.5  # Normalize to average
        normalized_xg_per_90 = float(xg_per_90 * xgc_normalization_factor)
        normalized_xa_per_90 = float(xa_per_90 * xgc_normalization_factor)
        
        # Fixture difficulty
        if fdr_data:
            fdr = float(fdr_data.get('fdr', 3.0))
            opponent_attack = float(fdr_data.get('opponent_attack_strength', 0.0))
        else:
            fdr = 3.0
            opponent_attack = 0.0
        
        # Home/away
        is_home = 1.0 if fixture_data and fixture_data.get('is_home', True) else 0.0
        
        # Position
        position = player_data.get('position', 'MID')
        position_encoded = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}.get(position, 2)
        
        # Team attack strength
        team_attack = float(player_data.get('team_attack_strength', 0.0))
        
        # Expected minutes factor
        expected_minutes = float(player_data.get('expected_minutes', 90.0)) / 90.0
        
        features = np.array([
            normalized_xg_per_90,  # Normalized by opponent xGC
            normalized_xa_per_90,  # Normalized by opponent xGC
            xg_per_90,  # Original (for reference)
            xa_per_90,  # Original (for reference)
            goals_per_90,
            assists_per_90,
            recent_xg_avg,
            recent_xa_avg,
            opponent_xgc,  # KEY FEATURE: Opponent's expected goals conceded (for normalization)
            opponent_defense_strength,
            opponent_attack,
            fdr / 5.0,  # Normalize to 0-1
            is_home,
            position_encoded / 3.0,  # Normalize
            team_attack,
            expected_minutes,
            xgc_normalization_factor  # Normalization factor applied
        ])
        
        return features.reshape(1, -1)
    
    def train(
        self,
        X: np.ndarray,
        y_xg: np.ndarray,
        y_xa: np.ndarray
    ):
        """
        Train both xG and xA LightGBM models.
        
        Args:
            X: Feature matrix (must include opponent_xgc)
            y_xg: Target xG values
            y_xa: Target xA values
        """
        X_scaled = self.scaler.fit_transform(X)
        
        self.xg_model.fit(X_scaled, y_xg)
        self.xa_model.fit(X_scaled, y_xa)
        
        self.xg_trained = True
        self.xa_trained = True
        logger.info("Attack model (LightGBM xG/xA) trained successfully")
        
        # Memory management
        gc.collect()
    
    def predict(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None,
        fdr_data: Optional[Dict] = None,
        opponent_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Predict xG and xA for a player.
        Uses opponent xGC to adjust predictions.
        Applies FDR scaling based on opponent defense strength.
        
        Returns:
            Dictionary with 'xg' and 'xa' predictions
        """
        # Calculate FDR scaling multiplier based on opponent defense strength
        # Weak defense (strength < 1.0) -> multiplier > 1.0 (easier to score)
        # Strong defense (strength > 1.0) -> multiplier < 1.0 (harder to score)
        fdr_multiplier = 1.0
        if fdr_data:
            opponent_defense = float(fdr_data.get('opponent_defense_strength', 0.0))
            # Defense strength typically in range -1 to +1 or 0 to 2
            # Normalize: if defense_strength > 0, it's a strong defense
            if opponent_defense != 0:
                # Linear scaling: defense_strength of -0.5 -> multiplier 1.15
                # defense_strength of +0.5 -> multiplier 0.85
                fdr_multiplier = 1.0 - (opponent_defense * 0.3)
                fdr_multiplier = float(np.clip(fdr_multiplier, 0.7, 1.4))  # Clamp to reasonable range
        elif opponent_data:
            opponent_defense = float(opponent_data.get('defense_strength', 0.0))
            if opponent_defense != 0:
                fdr_multiplier = 1.0 - (opponent_defense * 0.3)
                fdr_multiplier = float(np.clip(fdr_multiplier, 0.7, 1.4))
        
        if not self.xg_trained or not self.xa_trained:
            # Return default based on historical averages adjusted by opponent xGC and FDR
            base_xg = float(player_data.get('xg_per_90', 0.0))
            base_xa = float(player_data.get('xa_per_90', 0.0))
            
            # Adjust by opponent xGC (higher xGC = easier to score)
            if opponent_data:
                opponent_xgc = float(opponent_data.get('xgc_per_90', 1.5))
                xgc_factor = opponent_xgc / 1.5  # Normalize to average
                base_xg = base_xg * xgc_factor
            
            # Apply FDR scaling
            base_xg = base_xg * fdr_multiplier
            base_xa = base_xa * fdr_multiplier
            
            return {
                'xg': float(max(0.0, base_xg)),
                'xa': float(max(0.0, base_xa))
            }
        
        features = self.extract_features(player_data, fixture_data, fdr_data, opponent_data)
        features_scaled = self.scaler.transform(features)
        
        xg_pred = self.xg_model.predict(features_scaled)[0]
        xa_pred = self.xa_model.predict(features_scaled)[0]
        
        # Apply FDR scaling to predictions
        xg_pred = xg_pred * fdr_multiplier
        xa_pred = xa_pred * fdr_multiplier
        
        # Ensure non-negative
        xg_pred = max(0.0, xg_pred)
        xa_pred = max(0.0, xa_pred)
        
        return {
            'xg': float(xg_pred),
            'xa': float(xa_pred)
        }


class DefenseModel:
    """
    Poisson Regression model for predicting clean sheet probability (xCS).
    Formula: xCS = e^(-λ) where λ is expected goals conceded.
    """
    
    def __init__(self):
        self.is_fitted = False
    
    def predict_clean_sheet_probability(
        self,
        team_data: Dict,
        opponent_data: Dict,
        is_home: bool = True
    ) -> float:
        """
        Predict clean sheet probability using Poisson Distribution.
        Formula: xCS = e^(-λ) where λ = expected goals conceded.
        
        Args:
            team_data: Defending team data (with defense strength)
            opponent_data: Attacking team data (with attack strength)
            is_home: Whether defending team is at home
        
        Returns:
            Clean sheet probability (0-1)
        """
        # Get team defense strength
        team_defense = float(team_data.get('defense_strength', 1.0))
        if is_home:
            team_defense = float(team_data.get('strength_defence_home', team_defense))
        else:
            team_defense = float(team_data.get('strength_defence_away', team_defense))
        
        # Get opponent attack strength
        opponent_attack = float(opponent_data.get('attack_strength', 1.0))
        if is_home:
            opponent_attack = float(opponent_data.get('strength_attack_away', opponent_attack))
        else:
            opponent_attack = float(opponent_data.get('strength_attack_home', opponent_attack))
        
        # Calculate expected goals conceded (λ)
        # Base: average team concedes ~1.5 goals per game
        base_lambda = 1.5
        
        # Mathematical Logic Audit: λ scaling
        # We sometimes have strengths in ~1000 scale (FPL-style), and sometimes in a small scale (0-2).
        # Ensure we normalize safely and keep λ in a sane range.
        def _norm_strength(v: float) -> float:
            v = float(v)
            if v <= 0:
                return 1.0
            # If already looks like a 0-2-ish ratio, keep it (but clamp away from 0)
            if v <= 10.0:
                return max(0.1, min(v, 2.0))
            # Otherwise treat as 1000-scale
            return max(0.1, v / 1000.0)

        defense_factor = _norm_strength(team_defense)
        attack_factor = _norm_strength(opponent_attack)
        
        # Home advantage reduces goals conceded by ~10%
        home_factor = 0.9 if is_home else 1.0
        
        # Calculate λ (expected goals conceded)
        lambda_value = base_lambda * (1.0 / defense_factor) * attack_factor * home_factor

        # Clamp λ to a reasonable football range to avoid extreme tails dominating xCS.
        # Typical team xGA per match is roughly within [0.2, 3.0].
        lambda_value = float(np.clip(lambda_value, 0.0, 3.0))
        
        # Poisson probability: P(X=0) = e^(-λ)
        xcs = np.exp(-lambda_value)
        
        return float(np.clip(xcs, 0.0, 1.0))
    
    def fit(self, historical_data: pd.DataFrame):
        """
        Fit Poisson regression model (optional enhancement).
        For now, uses direct calculation based on team strengths.
        """
        self.is_fitted = True
        logger.info("Defense model (Poisson) fitted successfully")
        
        # Memory management
        gc.collect()


class PLEngine:
    """
    Predictive Engine (PLEngine) for FPL Point Prediction.
    Implements component-based models with async resource management:
    - xMins Model: RandomForestClassifier with days_since_last_match and is_cup_week
    - Attack Model: LightGBM with opponent xGC normalization
    - Defense Model: Poisson Regression for xCS
    - Final xP: FPL position-based scoring rules
    - Resource Management: Async pickle load/unload with gc.collect()
    """
    
    # FPL Point Values (2025/26 rules)
    GOAL_POINTS = {
        'GK': 6,
        'DEF': 6,
        'MID': 5,
        'FWD': 4
    }
    ASSIST_POINTS = 3
    CLEAN_SHEET_POINTS = {
        'GK': 4,
        'DEF': 4,
        'MID': 1,
        'FWD': 0
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_version = "5.0.0"  # PLEngine version with async resource management
        
        # Component models (lazy loaded)
        self.xmins_model: Optional[XMinsModel] = None
        self.attack_model: Optional[AttackModel] = None
        self.defense_model: Optional[DefenseModel] = None
        
        # Feature engineering
        self.feature_engine = FeatureEngineeringService()
        self.data_cleaning = DataCleaningService()
        
        # Model loading state
        self.models_loaded = False
        self.model_path = model_path
        self._load_lock = asyncio.Lock()
        
        # If no model_path provided, try to find the latest model
        # (Models will be loaded lazily via _ensure_models_loaded() when needed)
        if not self.model_path:
            self.model_path = self._load_latest_model()
    
    async def async_load_models(self, model_path: Optional[str] = None):
        """
        Asynchronously load models from pickle file.
        Uses asyncio to avoid blocking.
        """
        async with self._load_lock:
            if self.models_loaded:
                return
            
            path = model_path or self.model_path
            if not path or not os.path.exists(path):
                # Initialize empty models
                self.xmins_model = XMinsModel()
                self.attack_model = AttackModel()
                self.defense_model = DefenseModel()
                self.models_loaded = True
                return
            
            try:
                # Run pickle load in executor to avoid blocking
                loop = asyncio.get_event_loop()
                model_data = await loop.run_in_executor(None, self._load_pickle, path)
                
                # Restore models
                if model_data.get('xmins_model'):
                    self.xmins_model = XMinsModel()
                    self.xmins_model.model = model_data['xmins_model']
                    self.xmins_model.scaler = model_data.get('xmins_scaler', StandardScaler())
                    self.xmins_model.is_trained = True
                
                if model_data.get('attack_xg_model') and model_data.get('attack_xa_model'):
                    self.attack_model = AttackModel()
                    self.attack_model.xg_model = model_data['attack_xg_model']
                    self.attack_model.xa_model = model_data['attack_xa_model']
                    self.attack_model.scaler = model_data.get('attack_scaler', StandardScaler())
                    self.attack_model.xg_trained = True
                    self.attack_model.xa_trained = True
                
                if model_data.get('defense_model'):
                    self.defense_model = model_data['defense_model']
                else:
                    self.defense_model = DefenseModel()
                
                self.models_loaded = True
                logger.info(f"Models loaded asynchronously from {path}")
                
                # Memory management
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
                # Initialize empty models on error
                self.xmins_model = XMinsModel()
                self.attack_model = AttackModel()
                self.defense_model = DefenseModel()
                self.models_loaded = True
    
    def _load_pickle(self, path: str) -> Dict:
        """Synchronous pickle load (runs in executor)"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    async def async_unload_models(self):
        """
        Asynchronously unload models from memory.
        Frees up RAM for other operations.
        """
        async with self._load_lock:
            if not self.models_loaded:
                return
            
            # Save models before unloading if path exists
            if self.model_path:
                await self.async_save_models()
            
            # Clear models
            self.xmins_model = None
            self.attack_model = None
            self.defense_model = None
            self.models_loaded = False
            
            # Force garbage collection
            gc.collect()
            logger.info("Models unloaded from memory")
    
    async def async_save_models(self, model_path: Optional[str] = None):
        """
        Asynchronously save models to pickle file.
        """
        path = model_path or self.model_path
        if not path:
            logger.warning("No model path specified for saving")
            return
        
        if not self.models_loaded or not self.xmins_model or not self.attack_model:
            logger.warning("Models not loaded, cannot save")
            return
        
        try:
            # Prepare model data
            model_data = {
                'xmins_model': self.xmins_model.model if self.xmins_model.is_trained else None,
                'xmins_scaler': self.xmins_model.scaler,
                'attack_xg_model': self.attack_model.xg_model if self.attack_model.xg_trained else None,
                'attack_xa_model': self.attack_model.xa_model if self.attack_model.xa_trained else None,
                'attack_scaler': self.attack_model.scaler,
                'defense_model': self.defense_model,
                'version': self.model_version
            }
            
            # Run pickle save in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_pickle, path, model_data)
            
            logger.info(f"Models saved asynchronously to {path}")
            
            # Memory management
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def _save_pickle(self, path: str, data: Dict):
        """Synchronous pickle save (runs in executor)"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_latest_model(self) -> Optional[str]:
        """
        Search for the most recent .pkl model file in the models directory.
        Returns the path to the latest model file, or None if no models found.
        """
        try:
            # Try multiple possible locations for models directory
            possible_dirs = []
            
            # 1. Relative to this file: backend/models/ (when running from backend/)
            file_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up: app/services -> app -> backend
            backend_dir1 = os.path.dirname(os.path.dirname(file_dir))
            possible_dirs.append(os.path.join(backend_dir1, 'models'))
            
            # 2. Absolute path: /app/models (Docker environment)
            possible_dirs.append('/app/models')
            
            # 3. Current working directory models/
            possible_dirs.append(os.path.join(os.getcwd(), 'models'))
            
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
                if f.endswith('.pkl') and os.path.isfile(os.path.join(models_dir, f))
            ]
            
            if not pkl_files:
                logger.debug(f"No .pkl files found in {models_dir}")
                return None
            
            # Sort by modification time (most recent first)
            pkl_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_model = pkl_files[0]
            
            logger.info(f"Found latest model: {latest_model} (modified: {datetime.fromtimestamp(os.path.getmtime(latest_model))})")
            return latest_model
            
        except Exception as e:
            logger.warning(f"Error searching for latest model: {str(e)}", exc_info=True)
            return None
    
    def _ensure_models_loaded(self):
        """Ensure models are loaded (synchronous check)"""
        if not self.models_loaded:
            # If no model_path set, try to find latest model
            if not self.model_path:
                self.model_path = self._load_latest_model()
            
            if self.model_path and os.path.exists(self.model_path):
                # Synchronous load as fallback
                try:
                    # Try pickle first (as used by train_models.py)
                    try:
                        with open(self.model_path, 'rb') as f:
                            model_data = pickle.load(f)
                    except:
                        # Fallback to joblib if pickle fails
                        model_data = joblib.load(self.model_path)
                    
                    self.xmins_model = XMinsModel()
                    if model_data.get('xmins_model'):
                        self.xmins_model.model = model_data['xmins_model']
                        self.xmins_model.scaler = model_data.get('xmins_scaler', StandardScaler())
                        self.xmins_model.is_trained = True
                        logger.info("xMins model loaded and marked as trained")
                    
                    self.attack_model = AttackModel()
                    if model_data.get('attack_xg_model') and model_data.get('attack_xa_model'):
                        self.attack_model.xg_model = model_data['attack_xg_model']
                        self.attack_model.xa_model = model_data['attack_xa_model']
                        self.attack_model.scaler = model_data.get('attack_scaler', StandardScaler())
                        self.attack_model.xg_trained = True
                        self.attack_model.xa_trained = True
                        logger.info("Attack model loaded and marked as trained")
                    
                    self.defense_model = model_data.get('defense_model', DefenseModel())
                    self.models_loaded = True
                    logger.info(f"Models loaded successfully from {self.model_path}")
                except Exception as e:
                    logger.error(f"Error in fallback model load: {str(e)}", exc_info=True)
                    self.xmins_model = XMinsModel()
                    self.attack_model = AttackModel()
                    self.defense_model = DefenseModel()
                    self.models_loaded = True
            else:
                logger.warning("No model path available, initializing empty models")
                self.xmins_model = XMinsModel()
                self.attack_model = AttackModel()
                self.defense_model = DefenseModel()
                self.models_loaded = True
    
    def calculate_expected_points(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None,
        fdr_data: Optional[Dict] = None,
        team_data: Optional[Dict] = None,
        opponent_data: Optional[Dict] = None,
        historical_points: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate final expected points (xP) using FPL position-based scoring rules.
        Formula: xP = (xMins/90) * [(Gol_Puanı * xG) + (Asist_Puanı * xA) + (CS_Puanı * xCS) + DefCon_Puanı]
        
        FPL Scoring Rules (2025/26):
        - Goals: GK/DEF=6, MID=5, FWD=4
        - Assists: 3 points
        - Clean Sheets: GK/DEF=4, MID=1, FWD=0
        - DefCon: Position-based floor points
        
        Args:
            player_data: Player statistics
            fixture_data: Upcoming fixture information
            fdr_data: FDR data
            team_data: Player's team data
            opponent_data: Opponent team data (with xGC for normalization)
            historical_points: Historical points for form calculation
        
        Returns:
            Dictionary with xP and component breakdowns
        """
        self._ensure_models_loaded()
        position = player_data.get('position', 'MID')
        fpl_id = player_data.get('fpl_id', 'unknown')
        
        # CRITICAL: Check if models are trained before prediction
        if not self.xmins_model.is_trained:
            logger.warning(f"xMins model not trained! Returning 0 for player fpl_id={fpl_id}")
            return {
                'expected_points': 0.0,
                'xmins': 0.0,
                'xmins_factor': 0.0,
                'xg': 0.0,
                'xa': 0.0,
                'xcs': 0.0,
                'defcon_points': 0.0,
                'defcon_points_90': 0.0,
                'goal_component': 0.0,
                'assist_component': 0.0,
                'cs_component': 0.0,
                'appearance_points': 0.0,
                'expected_bonus': 0.0,
                'p_start': 0.0
            }
        
        if not self.attack_model.xg_trained or not self.attack_model.xa_trained:
            logger.warning(f"Attack model not trained! Returning 0 xG/xA for player fpl_id={fpl_id}")
            xg = 0.0
            xa = 0.0
        else:
            # 2. Predict xG and xA (Attack Model with opponent xGC normalization)
            attack_preds = self.attack_model.predict(
                player_data,
                fixture_data,
                fdr_data,
                opponent_data
            )
            xg = attack_preds['xg']
            xa = attack_preds['xa']
        
        # 1. Predict xMins (expected minutes) using RandomForestClassifier
        xmins = self.xmins_model.predict_expected_minutes(player_data, fixture_data)
        xmins_factor = xmins / 90.0
        
        # 3. Predict xCS (Clean Sheet probability) using Poisson Regression
        xcs = 0.0
        if position in ['DEF', 'GK'] and team_data and opponent_data:
            is_home = fixture_data.get('is_home', True) if fixture_data else True
            xcs = self.defense_model.predict_clean_sheet_probability(
                team_data,
                opponent_data,
                is_home
            )
        
        # 4. Calculate DefCon points (Mathematical Logic Audit)
        # Our final xP formula already multiplies by (xMins/90).
        # Therefore DefCon should be treated as a *per-90* contribution inside the parentheses,
        # otherwise it gets scaled twice (once in DefCon computation and again via xMins/90).
        defcon_metrics = self.data_cleaning.get_defcon_metrics(player_data, position)
        defcon_points_90 = defcon_metrics['floor_points_90']
        
        # 5. Get FPL point values for position
        goal_points = self.GOAL_POINTS.get(position, 4)
        assist_points = self.ASSIST_POINTS
        cs_points = self.CLEAN_SHEET_POINTS.get(position, 0)
        
        # 6. Calculate expected points components
        goal_component = goal_points * xg
        assist_component = assist_points * xa
        cs_component = cs_points * xcs
        
        # 7. FPL Appearance Points (CRITICAL FIX)
        # FPL Rules: +1 for 1-59 minutes, +2 for 60+ minutes
        # Expected value calculation:
        # - P(play any minutes) ≈ p_start (start probability)
        # - P(play 60+ | play) ≈ based on average minutes when playing
        # Simplified formula: (xMins/90) * 2 gives expected appearance points
        # This correctly scales from 0 (no play) to 2 (full 90 minutes)
        appearance_points = xmins_factor * 2.0
        
        # 8. Expected Bonus Points (xB)
        # FPL awards 1-3 bonus points to top performers in each match based on BPS (Bonus Point System)
        # BPS correlates heavily with goals, assists, and ICT index
        # Formula: xB = (goal_component + assist_component) * 0.15 + (ict_index * 0.02)
        # This typically yields 0.5-1.5 for top players
        ict_index = float(player_data.get('ict_index', player_data.get('influence', 0)) or 0)
        
        # Calculate expected bonus based on attacking contributions and influence
        expected_bonus = (goal_component + assist_component) * 0.15 + (ict_index * 0.02)
        
        # Clamp bonus to realistic range (0 to 3 max per game)
        expected_bonus = float(np.clip(expected_bonus, 0.0, 3.0))
        
        # 9. Final xP calculation: xP = appearance + (xMins/90) * [components + DefCon + Bonus]
        # Note: Components are scaled by xmins_factor (probability of playing)
        xp = appearance_points + xmins_factor * (goal_component + assist_component + cs_component + defcon_points_90 + expected_bonus)
        
        # Ensure non-negative
        xp = max(0.0, xp)
        
        # Memory management after calculation
        gc.collect()
        
        return {
            'expected_points': float(xp),
            'xmins': float(xmins),
            'xmins_factor': float(xmins_factor),
            'xg': float(xg),
            'xa': float(xa),
            'xcs': float(xcs),
            'defcon_points': float(defcon_points_90),
            'defcon_points_90': float(defcon_points_90),
            'goal_component': float(goal_component),
            'assist_component': float(assist_component),
            'cs_component': float(cs_component),
            'appearance_points': float(appearance_points),
            'expected_bonus': float(expected_bonus),
            'p_start': self.xmins_model.predict_start_probability(player_data, fixture_data)
        }
    
    def train(
        self,
        training_data: pd.DataFrame,
        xmins_features: Optional[np.ndarray] = None,
        xmins_labels: Optional[np.ndarray] = None,
        attack_features: Optional[np.ndarray] = None,
        attack_xg_labels: Optional[np.ndarray] = None,
        attack_xa_labels: Optional[np.ndarray] = None
    ):
        """
        Train all component models.
        
        Args:
            training_data: Historical training data
            xmins_features: Features for xMins model (must include days_since_last_match and is_cup_week)
            xmins_labels: Labels for xMins (1=started, 0=didn't start)
            attack_features: Features for Attack model (must include opponent_xgc for normalization)
            attack_xg_labels: Target xG values
            attack_xa_labels: Target xA values
        """
        self._ensure_models_loaded()
        
        # CRITICAL: Boş veri kontrolü
        logger.info(f"Training on {len(training_data)} rows from training_data DataFrame")
        if training_data.empty:
            logger.error("ERROR: training_data DataFrame is EMPTY! Cannot train models.")
            return
        
        # Train xMins model (RandomForestClassifier)
        if xmins_features is not None and xmins_labels is not None:
            logger.info(f"Training xMins model with {len(xmins_features)} samples, {len(xmins_labels)} labels")
            if len(xmins_features) == 0 or len(xmins_labels) == 0:
                logger.error("ERROR: xmins_features or xmins_labels is EMPTY! Skipping xMins model training.")
            else:
                self.xmins_model.train(xmins_features, xmins_labels)
                gc.collect()  # Memory management after each model
        else:
            logger.warning("xmins_features or xmins_labels is None - skipping xMins model training")
        
        # Train Attack model (LightGBM with xGC normalization)
        if attack_features is not None and attack_xg_labels is not None and attack_xa_labels is not None:
            logger.info(f"Training Attack model with {len(attack_features)} samples, {len(attack_xg_labels)} xG labels, {len(attack_xa_labels)} xA labels")
            if len(attack_features) == 0 or len(attack_xg_labels) == 0 or len(attack_xa_labels) == 0:
                logger.error("ERROR: attack_features, attack_xg_labels, or attack_xa_labels is EMPTY! Skipping Attack model training.")
            else:
                self.attack_model.train(attack_features, attack_xg_labels, attack_xa_labels)
                gc.collect()  # Memory management
        else:
            logger.warning("attack_features, attack_xg_labels, or attack_xa_labels is None - skipping Attack model training")
        
        # Fit Defense model (Poisson Regression)
        logger.info(f"Fitting Defense model with {len(training_data)} rows")
        self.defense_model.fit(training_data)
        gc.collect()  # Memory management
        
        logger.info("All PLEngine models trained successfully")
    
    def predict(
        self,
        player_data: Dict,
        historical_points: Optional[List[float]] = None,
        fixture_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Predict points for a player using comprehensive xP calculation.
        Uses all component models: xMins (RandomForest), Attack (LightGBM), Defense (Poisson).
        
        Args:
            player_data: Current player statistics
            historical_points: Historical points (most recent first)
            fixture_data: Upcoming fixture information
        
        Returns:
            Dictionary with predicted points and component breakdowns
        """
        self._ensure_models_loaded()
        
        # Get FDR and team data
        fdr_data = None
        team_data = None
        opponent_data = None
        
        if fixture_data:
            # Extract FDR data
            engineered_features = self.feature_engine.calculate_all_features(
                player_data=player_data,
                historical_points=historical_points or [],
                fixture_data=fixture_data,
                position=player_data.get('position', 'MID')
            )
            
            fdr_data = {
                'fdr': engineered_features.get('fdr', 3.0),
                'opponent_defense_strength': engineered_features.get('fdr_defense', 0.0),
                'opponent_attack_strength': engineered_features.get('fdr_attack', 0.0)
            }
            
            # Get opponent xGC for normalization
            opponent_team_id = fixture_data.get('opponent_team')
            if opponent_team_id:
                # Estimate opponent xGC from defense strength
                # Lower defense strength = higher xGC (easier to score against)
                opponent_xgc = 1.5 - (fdr_data['opponent_defense_strength'] * 0.1)
                opponent_data = {
                    'xgc_per_90': opponent_xgc,
                    'expected_goals_conceded': opponent_xgc,
                    'attack_strength': fdr_data['opponent_attack_strength'],
                    'defense_strength': fdr_data['opponent_defense_strength']
                }
        
        # Calculate comprehensive xP using FPL scoring rules
        xp_result = self.calculate_expected_points(
            player_data=player_data,
            fixture_data=fixture_data,
            fdr_data=fdr_data,
            team_data=team_data,
            opponent_data=opponent_data,
            historical_points=historical_points
        )
        
        # Add confidence score
        confidence = self._calculate_confidence(player_data, xp_result)
        
        result = {
            'predicted_points': xp_result['expected_points'],
            'confidence_score': confidence,
            **xp_result
        }
        
        # Memory management
        gc.collect()
        
        return result
    
    def _calculate_confidence(
        self,
        player_data: Dict,
        xp_result: Dict[str, float]
    ) -> float:
        """
        Calculate confidence score based on data quality and model certainty.
        """
        confidence = 0.7  # Base confidence
        
        # Increase confidence if player has good historical data
        if player_data.get('minutes', 0) > 500:
            confidence += 0.1
        
        # Increase confidence if xMins is high (more certain about playing)
        if xp_result.get('p_start', 0) > 0.8:
            confidence += 0.1
        
        # Decrease confidence if injury status is doubtful
        status = player_data.get('status', 'a').lower()
        if status == 'd':
            confidence -= 0.2
        
        return float(np.clip(confidence, 0.0, 1.0))