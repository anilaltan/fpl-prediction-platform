"""
Component-Based Predictive Engine for FPL Point Prediction
Implements specialized sub-models:
- xMins: Starting 11 probability (P_start)
- Attack Model: xG/xA predictions using LightGBM
- Defense Model: Clean sheet probability (xCS) using Poisson
- Momentum Layer: LSTM for time-series form analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import joblib
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

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available, LSTM momentum layer disabled")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import poisson

logger = logging.getLogger(__name__)


class XMinsModel:
    """
    Predicts starting 11 probability (P_start) using XGBoost or Random Forest.
    Considers:
    - Rest time (days since last match)
    - Cup match presence (midweek fixtures)
    - Injury status
    - Recent minutes played
    - Position depth
    """
    
    def __init__(self):
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, player_data: Dict, fixture_data: Optional[Dict] = None) -> np.ndarray:
        """
        Extract features for xMins prediction.
        
        Features:
        - rest_days: Days since last match
        - cup_match: Binary (1 if midweek cup match exists)
        - injury_status: 0=fit, 1=doubtful, 2=out
        - recent_minutes_avg: Average minutes in last 3 matches
        - position_depth: Squad depth at position (1-3)
        - form_score: Recent form
        - price: Player price (proxy for importance)
        """
        # Rest days calculation
        last_match_date = player_data.get('last_match_date')
        if last_match_date:
            if isinstance(last_match_date, str):
                last_match = datetime.fromisoformat(last_match_date.replace('Z', '+00:00'))
            else:
                last_match = last_match_date
            rest_days = (datetime.now() - last_match.replace(tzinfo=None)).days
        else:
            rest_days = 7  # Default: full week rest
        
        # Cup match presence
        cup_match = 1 if fixture_data and fixture_data.get('has_cup_match', False) else 0
        
        # Injury status (0=fit, 1=doubtful, 2=out)
        injury_status_map = {'a': 0, 'd': 1, 'i': 2, 'n': 0, 's': 2}
        status = player_data.get('status', 'a').lower()
        injury_status = injury_status_map.get(status, 0)
        
        # Recent minutes average
        recent_minutes = player_data.get('recent_minutes', [])
        if recent_minutes:
            recent_minutes_avg = np.mean(recent_minutes[:3]) if len(recent_minutes) >= 3 else np.mean(recent_minutes)
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
            rest_days,
            cup_match,
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
        Train the xMins model.
        
        Args:
            X: Feature matrix
            y: Binary labels (1 = started, 0 = didn't start)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("xMins model trained successfully")
    
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
            # Default prediction based on recent minutes
            recent_minutes = player_data.get('recent_minutes', [])
            if recent_minutes:
                avg_minutes = np.mean(recent_minutes[:3]) if len(recent_minutes) >= 3 else np.mean(recent_minutes)
                return min(1.0, avg_minutes / 90.0)
            return 0.7  # Default
        
        features = self.extract_features(player_data, fixture_data)
        features_scaled = self.scaler.transform(features)
        
        # Predict probability
        if XGBOOST_AVAILABLE:
            prob = self.model.predict_proba(features_scaled)[0][1]
        else:
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
        
        # If starting, estimate minutes based on recent average
        recent_minutes = player_data.get('recent_minutes', [90])
        avg_minutes_when_starting = np.mean([m for m in recent_minutes if m > 0]) if recent_minutes else 85.0
        
        # Expected minutes = P(start) * avg_minutes_when_starting
        expected_minutes = p_start * avg_minutes_when_starting
        
        return float(np.clip(expected_minutes, 0.0, 90.0))


class AttackModel:
    """
    Predicts expected goals (xG) and expected assists (xA) using LightGBM.
    Considers:
    - Opponent defense strength
    - Home/away advantage
    - Player historical xG/xA
    - Recent form
    - Fixture difficulty
    """
    
    def __init__(self):
        if LIGHTGBM_AVAILABLE:
            self.xg_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )
            self.xa_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )
        else:
            self.xg_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.xa_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.scaler = StandardScaler()
        self.xg_trained = False
        self.xa_trained = False
    
    def extract_features(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None,
        fdr_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Extract features for xG/xA prediction.
        """
        # Player historical stats
        xg_per_90 = player_data.get('xg_per_90', 0.0)
        xa_per_90 = player_data.get('xa_per_90', 0.0)
        goals_per_90 = player_data.get('goals_per_90', 0.0)
        assists_per_90 = player_data.get('assists_per_90', 0.0)
        
        # Recent form (last 5 games)
        recent_xg = player_data.get('recent_xg', [])
        recent_xa = player_data.get('recent_xa', [])
        recent_xg_avg = np.mean(recent_xg) if recent_xg else xg_per_90
        recent_xa_avg = np.mean(recent_xa) if recent_xa else xa_per_90
        
        # Fixture difficulty
        if fdr_data:
            opponent_defense = fdr_data.get('opponent_defense_strength', 0.0)
            opponent_attack = fdr_data.get('opponent_attack_strength', 0.0)
            fdr = fdr_data.get('fdr', 3.0)
        else:
            opponent_defense = 0.0
            opponent_attack = 0.0
            fdr = 3.0
        
        # Home/away
        is_home = 1.0 if fixture_data and fixture_data.get('is_home', True) else 0.0
        
        # Position
        position = player_data.get('position', 'MID')
        position_encoded = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}.get(position, 2)
        
        # Team attack strength
        team_attack = player_data.get('team_attack_strength', 0.0)
        
        # Expected minutes factor
        expected_minutes = player_data.get('expected_minutes', 90.0) / 90.0
        
        features = np.array([
            xg_per_90,
            xa_per_90,
            goals_per_90,
            assists_per_90,
            recent_xg_avg,
            recent_xa_avg,
            opponent_defense,
            opponent_attack,
            fdr / 5.0,  # Normalize to 0-1
            is_home,
            position_encoded / 3.0,  # Normalize
            team_attack,
            expected_minutes
        ])
        
        return features.reshape(1, -1)
    
    def train(
        self,
        X: np.ndarray,
        y_xg: np.ndarray,
        y_xa: np.ndarray
    ):
        """Train both xG and xA models"""
        X_scaled = self.scaler.fit_transform(X)
        
        self.xg_model.fit(X_scaled, y_xg)
        self.xa_model.fit(X_scaled, y_xa)
        
        self.xg_trained = True
        self.xa_trained = True
        logger.info("Attack model (xG/xA) trained successfully")
    
    def predict(
        self,
        player_data: Dict,
        fixture_data: Optional[Dict] = None,
        fdr_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Predict xG and xA for a player.
        
        Returns:
            Dictionary with 'xg' and 'xa' predictions
        """
        if not self.xg_trained or not self.xa_trained:
            # Return default based on historical averages
            return {
                'xg': player_data.get('xg_per_90', 0.0),
                'xa': player_data.get('xa_per_90', 0.0)
            }
        
        features = self.extract_features(player_data, fixture_data, fdr_data)
        features_scaled = self.scaler.transform(features)
        
        xg_pred = self.xg_model.predict(features_scaled)[0]
        xa_pred = self.xa_model.predict(features_scaled)[0]
        
        # Ensure non-negative
        xg_pred = max(0.0, xg_pred)
        xa_pred = max(0.0, xa_pred)
        
        return {
            'xg': float(xg_pred),
            'xa': float(xa_pred)
        }


class DefenseModel:
    """
    Predicts clean sheet probability (xCS) using Poisson Distribution.
    Formula: xCS = e^(-λ) where λ is expected goals conceded
    """
    
    def __init__(self):
        self.is_fitted = False
    
    def calculate_expected_goals_conceded(
        self,
        team_data: Dict,
        opponent_data: Dict,
        is_home: bool = True
    ) -> float:
        """
        Calculate expected goals conceded using team defense and opponent attack.
        
        Args:
            team_data: Team's defense strength data
            opponent_data: Opponent's attack strength data
            is_home: Whether playing at home
        
        Returns:
            Expected goals conceded (λ)
        """
        # Get team defense strength
        team_defense = team_data.get('defense_strength', 0.0)
        
        # Get opponent attack strength
        opponent_attack = opponent_data.get('attack_strength', 0.0)
        
        # Home advantage (reduce goals conceded at home)
        home_advantage = -0.2 if is_home else 0.0
        
        # Expected goals conceded: λ = exp(defense + opponent_attack + home_advantage)
        # Lower defense_strength = better defense = lower λ
        lambda_goals = np.exp(team_defense + opponent_attack + home_advantage)
        
        return float(lambda_goals)
    
    def predict_clean_sheet_probability(
        self,
        team_data: Dict,
        opponent_data: Dict,
        is_home: bool = True
    ) -> float:
        """
        Predict clean sheet probability: xCS = e^(-λ)
        
        Returns:
            Clean sheet probability (0-1)
        """
        lambda_goals = self.calculate_expected_goals_conceded(team_data, opponent_data, is_home)
        
        # Clean sheet probability: P(0 goals) = e^(-λ)
        xcs = np.exp(-lambda_goals)
        
        return float(np.clip(xcs, 0.0, 1.0))
    
    def predict_goals_conceded_distribution(
        self,
        team_data: Dict,
        opponent_data: Dict,
        is_home: bool = True,
        max_goals: int = 5
    ) -> Dict[int, float]:
        """
        Predict probability distribution for goals conceded.
        
        Returns:
            Dictionary mapping goals (0-5) to probabilities
        """
        lambda_goals = self.calculate_expected_goals_conceded(team_data, opponent_data, is_home)
        
        distribution = {}
        for goals in range(max_goals + 1):
            prob = poisson.pmf(goals, lambda_goals)
            distribution[goals] = float(prob)
        
        return distribution


class MomentumLayer:
    """
    LSTM-based momentum layer for analyzing time-series form trends.
    Captures long-term patterns and momentum shifts in player performance.
    """
    
    def __init__(self, sequence_length: int = 10, hidden_units: int = 50):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.model: Optional[Sequential] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, LSTM momentum layer disabled")
    
    def prepare_sequences(
        self,
        historical_data: List[float],
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            historical_data: Time series data (most recent first)
            sequence_length: Length of input sequences
        
        Returns:
            (X, y) where X is sequences and y is next value
        """
        if not historical_data or len(historical_data) < 2:
            return np.array([]), np.array([])
        
        seq_len = sequence_length or self.sequence_length
        
        # Reverse to chronological order (oldest first)
        data = list(reversed(historical_data))
        data_array = np.array(data).reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data_array)
        
        X, y = [], []
        for i in range(len(scaled_data) - seq_len):
            X.append(scaled_data[i:i+seq_len, 0])
            y.append(scaled_data[i+seq_len, 0])
        
        if len(X) == 0:
            return np.array([]), np.array([])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM: (samples, time_steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = Sequential([
            LSTM(self.hidden_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.hidden_units, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(
        self,
        historical_data: List[float],
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """
        Train LSTM model on historical data.
        
        Args:
            historical_data: Time series data
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("Cannot train LSTM: TensorFlow not available")
            return
        
        X, y = self.prepare_sequences(historical_data)
        
        if len(X) == 0:
            logger.warning("Insufficient data for LSTM training")
            return
        
        # Build model
        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_model(input_shape)
        
        if self.model is None:
            return
        
        # Train
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        self.is_trained = True
        logger.info("LSTM momentum layer trained successfully")
    
    def predict_momentum(
        self,
        historical_data: List[float],
        forecast_steps: int = 1
    ) -> Dict[str, float]:
        """
        Predict momentum and future trend.
        
        Args:
            historical_data: Recent time series data
            forecast_steps: Number of steps to forecast
        
        Returns:
            Dictionary with momentum indicators
        """
        if not self.is_trained or not TENSORFLOW_AVAILABLE:
            # Fallback: simple trend calculation
            if len(historical_data) < 3:
                return {'momentum': 0.0, 'trend': 0.0, 'forecast': historical_data[0] if historical_data else 0.0}
            
            recent = np.mean(historical_data[:3])
            previous = np.mean(historical_data[3:6]) if len(historical_data) >= 6 else recent
            trend = recent - previous
            
            return {
                'momentum': float(trend),
                'trend': float(trend),
                'forecast': float(recent + trend)
            }
        
        # Prepare last sequence
        data = list(reversed(historical_data))
        if len(data) < self.sequence_length:
            # Pad with last value
            data = data + [data[-1]] * (self.sequence_length - len(data))
        
        data_array = np.array(data[-self.sequence_length:]).reshape(-1, 1)
        scaled_data = self.scaler.transform(data_array)
        
        # Reshape for prediction
        X = scaled_data.reshape((1, self.sequence_length, 1))
        
        # Predict
        prediction = self.model.predict(X, verbose=0)[0][0]
        prediction = self.scaler.inverse_transform([[prediction]])[0][0]
        
        # Calculate momentum (rate of change)
        if len(historical_data) >= 3:
            recent_avg = np.mean(historical_data[:3])
            momentum = float(prediction - recent_avg)
        else:
            momentum = 0.0
        
        # Trend (improving/declining)
        if len(historical_data) >= 5:
            recent = np.mean(historical_data[:3])
            previous = np.mean(historical_data[3:6]) if len(historical_data) >= 6 else recent
            trend = recent - previous
        else:
            trend = momentum
        
        return {
            'momentum': float(momentum),
            'trend': float(trend),
            'forecast': float(prediction)
        }


class PredictiveEngine:
    """
    Main predictive engine that orchestrates all sub-models.
    """
    
    def __init__(self):
        self.xmins_model = XMinsModel()
        self.attack_model = AttackModel()
        self.defense_model = DefenseModel()
        self.momentum_layer = MomentumLayer()
    
    def predict_comprehensive(
        self,
        player_data: Dict,
        historical_points: List[float],
        fixture_data: Optional[Dict] = None,
        fdr_data: Optional[Dict] = None,
        team_data: Optional[Dict] = None,
        opponent_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Comprehensive prediction combining all sub-models.
        
        Returns:
            Dictionary with all predictions
        """
        predictions = {}
        
        # 1. xMins prediction
        p_start = self.xmins_model.predict_start_probability(player_data, fixture_data)
        expected_minutes = self.xmins_model.predict_expected_minutes(player_data, fixture_data)
        predictions['p_start'] = p_start
        predictions['expected_minutes'] = expected_minutes
        
        # 2. Attack predictions (xG/xA)
        attack_preds = self.attack_model.predict(player_data, fixture_data, fdr_data)
        predictions['xg'] = attack_preds['xg']
        predictions['xa'] = attack_preds['xa']
        
        # 3. Defense predictions (xCS) - only for defenders/GKs
        position = player_data.get('position', 'MID')
        if position in ['DEF', 'GK'] and team_data and opponent_data:
            is_home = fixture_data.get('is_home', True) if fixture_data else True
            xcs = self.defense_model.predict_clean_sheet_probability(
                team_data, opponent_data, is_home
            )
            predictions['xcs'] = xcs
        else:
            predictions['xcs'] = 0.0
        
        # 4. Momentum analysis
        if historical_points:
            momentum = self.momentum_layer.predict_momentum(historical_points)
            predictions['momentum'] = momentum['momentum']
            predictions['trend'] = momentum['trend']
            predictions['momentum_forecast'] = momentum['forecast']
        else:
            predictions['momentum'] = 0.0
            predictions['trend'] = 0.0
            predictions['momentum_forecast'] = 0.0
        
        return predictions