"""
Feature Engineering Service for FPL Point Prediction
Implements self-adaptive feature engineering based on Moneyball principles:
- Dynamic Form Alpha (α) with Bayesian Optimization
- Dynamic FDR (Dixon-Coles Poisson Regression)
- DefCon Integration for 2025/26 rules
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import poisson
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from statsmodels.discrete.discrete_model import Poisson
import logging

logger = logging.getLogger(__name__)


class DynamicFormAlpha:
    """
    Calculates dynamic form decay coefficient (α) using Bayesian Optimization.
    Minimizes RMSE by finding optimal exponential decay weight for recent form.
    
    Form calculation: weighted_average = Σ(α^(n-i) * points_i) / Σ(α^(n-i))
    where n is current gameweek, i is historical gameweek
    """
    
    def __init__(self, min_alpha: float = 0.1, max_alpha: float = 1.0):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.optimal_alpha: Optional[float] = None
        self.optimization_history: List[Dict] = []
    
    def calculate_form(
        self, 
        historical_points: List[float], 
        alpha: float,
        lookback_weeks: int = 5
    ) -> float:
        """
        Calculate weighted form using exponential decay.
        
        Args:
            historical_points: List of points from most recent to oldest
            alpha: Decay coefficient (higher = more weight on recent)
            lookback_weeks: Number of weeks to consider
        
        Returns:
            Weighted form score
        """
        if not historical_points or len(historical_points) == 0:
            return 0.0
        
        # Take only recent weeks
        recent_points = historical_points[:lookback_weeks]
        n = len(recent_points)
        
        if n == 0:
            return 0.0
        
        # IMPORTANT (Mathematical Logic Audit):
        # `historical_points` is ordered most-recent -> oldest.
        # We want recent weeks to have HIGHER weight and older weeks to decay.
        #
        # Using `alpha ** (n-1-i)` does the opposite (for alpha<1 it gives the MOST recent the SMALLEST weight).
        #
        # Correct decay for most-recent-first input:
        #   w0 = 1 (most recent), w1 = alpha, w2 = alpha^2, ...
        # where 0 < alpha < 1 means faster decay (more emphasis on recent).
        weights = [alpha ** i for i in range(n)]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return np.mean(recent_points)
        
        weighted_sum = sum(p * w for p, w in zip(recent_points, weights))
        return weighted_sum / total_weight
    
    def _calculate_rmse(
        self, 
        alpha: float, 
        historical_data: pd.DataFrame,
        lookback_weeks: int = 5
    ) -> float:
        """
        Calculate RMSE for a given alpha value.
        Uses expanding window approach for validation.
        """
        errors = []
        
        # Expanding window: train on weeks 1 to i, predict week i+1
        for i in range(lookback_weeks, len(historical_data)):
            train_data = historical_data.iloc[:i]
            actual = historical_data.iloc[i]['points']
            
            # Calculate form using historical points up to week i
            historical_points = train_data['points'].tolist()
            predicted_form = self.calculate_form(historical_points, alpha, lookback_weeks)
            
            # Simple prediction: form * minutes_factor (if available)
            minutes_factor = historical_data.iloc[i].get('minutes', 90) / 90.0
            predicted_points = predicted_form * minutes_factor
            
            error = (predicted_points - actual) ** 2
            errors.append(error)
        
        if not errors:
            return 1e6  # Large penalty if no data
        
        rmse = np.sqrt(np.mean(errors))
        return rmse
    
    def optimize_alpha(
        self, 
        historical_data: pd.DataFrame,
        lookback_weeks: int = 5,
        n_calls: int = 50
    ) -> float:
        """
        Use Bayesian Optimization to find optimal alpha that minimizes RMSE.
        
        Args:
            historical_data: DataFrame with columns ['points', 'minutes', ...]
            lookback_weeks: Number of weeks to consider for form
            n_calls: Number of optimization iterations
        
        Returns:
            Optimal alpha value
        """
        if len(historical_data) < lookback_weeks + 1:
            logger.warning("Insufficient data for alpha optimization, using default 0.5")
            self.optimal_alpha = 0.5
            return self.optimal_alpha
        
        # Define search space
        space = [Real(self.min_alpha, self.max_alpha, name='alpha')]
        
        # Objective function
        @use_named_args(dimensions=space)
        def objective(alpha):
            return self._calculate_rmse(alpha, historical_data, lookback_weeks)
        
        # Run Bayesian Optimization
        logger.info(f"Starting Bayesian Optimization for alpha (n_calls={n_calls})")
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI'  # Expected Improvement
        )
        
        self.optimal_alpha = result.x[0]
        best_rmse = result.fun
        
        logger.info(f"Optimal alpha: {self.optimal_alpha:.4f}, RMSE: {best_rmse:.4f}")
        
        return self.optimal_alpha
    
    def get_form(self, historical_points: List[float], lookback_weeks: int = 5) -> float:
        """Get form using optimal alpha (or default if not optimized)"""
        alpha = self.optimal_alpha if self.optimal_alpha is not None else 0.5
        return self.calculate_form(historical_points, alpha, lookback_weeks)


class DixonColesFDR:
    """
    Implements Dixon-Coles model for calculating team attack/defense strengths.
    Uses Poisson Regression to estimate:
    - Home attack strength (λ_home)
    - Away attack strength (λ_away)
    - Home defense strength (μ_home)
    - Away defense strength (μ_away)
    
    Expected goals: λ = exp(α_home + attack_home - defense_away)
    """
    
    def __init__(self):
        self.attack_strengths: Dict[str, float] = {}
        self.defense_strengths: Dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.is_fitted: bool = False
    
    def prepare_fixture_data(self, fixtures: List[Dict]) -> pd.DataFrame:
        """
        Prepare fixture data for Poisson regression.
        
        Args:
            fixtures: List of fixture dicts with 'team_h', 'team_a', 'goals_h', 'goals_a'
        
        Returns:
            DataFrame ready for regression
        """
        data = []
        for fixture in fixtures:
            if 'goals_h' not in fixture or 'goals_a' not in fixture:
                continue
            
            data.append({
                'team_h': fixture.get('team_h', fixture.get('team_h_name', '')),
                'team_a': fixture.get('team_a', fixture.get('team_a_name', '')),
                'goals_h': int(fixture.get('goals_h', 0)),
                'goals_a': int(fixture.get('goals_a', 0)),
                'is_home': 1
            })
        
        return pd.DataFrame(data)
    
    def fit(self, fixtures: List[Dict]):
        """
        Fit Dixon-Coles model using Poisson regression.
        
        Args:
            fixtures: Historical fixture data
        """
        if not fixtures:
            logger.warning("No fixture data provided for FDR calculation")
            return
        
        df = self.prepare_fixture_data(fixtures)
        
        if len(df) == 0:
            logger.warning("No valid fixture data after preparation")
            return
        
        # Get unique teams
        teams = sorted(set(df['team_h'].unique()) | set(df['team_a'].unique()))
        
        if len(teams) < 2:
            logger.warning("Insufficient teams for FDR calculation")
            return
        
        # Prepare design matrix for home goals
        X_home = []
        y_home = []
        
        for _, row in df.iterrows():
            # Home team features
            home_features = [0.0] * (len(teams) * 2 + 1)  # +1 for home advantage
            home_features[0] = 1.0  # Home advantage
            
            # Set attack strength for home team
            home_idx = teams.index(row['team_h'])
            home_features[1 + home_idx] = 1.0
            
            # Set defense strength for away team
            away_idx = teams.index(row['team_a'])
            home_features[1 + len(teams) + away_idx] = -1.0
            
            X_home.append(home_features)
            y_home.append(row['goals_h'])
        
        X_home = np.array(X_home)
        y_home = np.array(y_home)
        
        # Fit Poisson model for home goals
        try:
            poisson_model_home = Poisson(y_home, X_home)
            result_home = poisson_model_home.fit(method='lbfgs', maxiter=1000)
            
            params = result_home.params
            
            # Extract coefficients
            self.home_advantage = params[0]
            
            # Attack strengths (offset by average)
            attack_params = params[1:1+len(teams)]
            avg_attack = np.mean(attack_params)
            self.attack_strengths = {
                team: float(attack_params[i] - avg_attack)
                for i, team in enumerate(teams)
            }
            
            # Defense strengths (offset by average)
            defense_params = params[1+len(teams):]
            avg_defense = np.mean(defense_params)
            self.defense_strengths = {
                team: float(-(defense_params[i] - avg_defense))  # Negative for defense
                for i, team in enumerate(teams)
            }
            
            self.is_fitted = True
            logger.info(f"Fitted Dixon-Coles model for {len(teams)} teams")
            logger.info(f"Home advantage: {self.home_advantage:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting Dixon-Coles model: {str(e)}")
            # Fallback: use simple averages
            self._calculate_simple_fdr(df, teams)
    
    def _calculate_simple_fdr(self, df: pd.DataFrame, teams: List[str]):
        """Fallback: calculate simple average goals for/against"""
        for team in teams:
            home_goals = df[df['team_h'] == team]['goals_h'].mean()
            away_goals = df[df['team_a'] == team]['goals_a'].mean()
            goals_for = (home_goals + away_goals) / 2
            
            home_conceded = df[df['team_h'] == team]['goals_a'].mean()
            away_conceded = df[df['team_a'] == team]['goals_h'].mean()
            goals_against = (home_conceded + away_conceded) / 2
            
            self.attack_strengths[team] = float(goals_for - 1.5)  # Normalize
            self.defense_strengths[team] = float(1.5 - goals_against)  # Normalize
        
        self.home_advantage = 0.2
        self.is_fitted = True
    
    def get_expected_goals(
        self, 
        home_team: str, 
        away_team: str
    ) -> Tuple[float, float]:
        """
        Calculate expected goals for a fixture.
        
        Returns:
            (expected_goals_home, expected_goals_away)
        """
        if not self.is_fitted:
            return (1.5, 1.5)  # Default
        
        home_attack = self.attack_strengths.get(home_team, 0.0)
        away_defense = self.defense_strengths.get(away_team, 0.0)
        away_attack = self.attack_strengths.get(away_team, 0.0)
        home_defense = self.defense_strengths.get(home_team, 0.0)
        
        # Expected goals: λ = exp(α_home + attack - defense)
        exp_goals_home = np.exp(self.home_advantage + home_attack - away_defense)
        exp_goals_away = np.exp(away_attack - home_defense)
        
        return (float(exp_goals_home), float(exp_goals_away))
    
    def get_fixture_difficulty(
        self, 
        team: str, 
        opponent: str, 
        is_home: bool
    ) -> float:
        """
        Calculate fixture difficulty rating (FDR) for a team.
        Lower value = easier fixture, Higher value = harder fixture.
        
        Returns:
            FDR score (typically 1-5 scale, normalized)
        """
        if not self.is_fitted:
            return 3.0  # Neutral
        
        if is_home:
            exp_goals_home, exp_goals_away = self.get_expected_goals(team, opponent)
            # FDR based on expected goals conceded
            fdr = 1 + (exp_goals_away / 2.0)  # Normalize to 1-5 scale
        else:
            exp_goals_home, exp_goals_away = self.get_expected_goals(opponent, team)
            # FDR based on expected goals conceded (away)
            fdr = 1 + (exp_goals_home / 2.0)
        
        return float(np.clip(fdr, 1.0, 5.0))


class DefConFeatureEngine:
    """
    Extracts DefCon (Defensive Contribution) features for 2025/26 FPL rules.
    New rules award points for:
    - Blocks (defenders/midfielders)
    - Interventions (tackles, interceptions)
    - Passes (successful passes in build-up)
    
    Creates "floor points" feature representing minimum expected points.
    """
    
    def __init__(self):
        self.defcon_rules = {
            'defender': {
                'block': 1,  # 1 point per block
                'intervention': 1,  # 1 point per intervention
                'pass_bonus': 0.1  # 0.1 points per 10 successful passes
            },
            'midfielder': {
                'block': 1,
                'intervention': 1,
                'pass_bonus': 0.1
            },
            'forward': {
                'block': 0,  # Forwards don't get block points
                'intervention': 1,
                'pass_bonus': 0.1
            }
        }
    
    def calculate_floor_points(
        self, 
        player_data: Dict,
        position: str,
        minutes: int = 90
    ) -> float:
        """
        Calculate floor points based on DefCon rules.
        Represents minimum expected points from defensive contributions.
        
        Args:
            player_data: Player statistics dict
            position: Player position (DEF, MID, FWD)
            minutes: Expected minutes played
        
        Returns:
            Floor points estimate
        """
        if minutes == 0:
            return 0.0
        
        position_lower = position.lower()
        if 'def' in position_lower:
            rules = self.defcon_rules['defender']
        elif 'mid' in position_lower:
            rules = self.defcon_rules['midfielder']
        else:
            rules = self.defcon_rules['forward']
        
        # Extract DefCon stats (if available from API)
        blocks = player_data.get('blocks', player_data.get('total_blocks', 0))
        interventions = player_data.get('interceptions', player_data.get('total_interceptions', 0))
        passes = player_data.get('passes', player_data.get('total_passes', 0))
        
        # Calculate per-90 averages if we have historical data
        if 'minutes_played' in player_data and player_data['minutes_played'] > 0:
            games_played = player_data.get('games_played', 1)
            blocks_per_90 = (blocks / games_played) * (90 / max(player_data['minutes_played'] / games_played, 1))
            interventions_per_90 = (interventions / games_played) * (90 / max(player_data['minutes_played'] / games_played, 1))
            passes_per_90 = (passes / games_played) * (90 / max(player_data['minutes_played'] / games_played, 1))
        else:
            # Use season averages if available
            blocks_per_90 = player_data.get('blocks_per_90', blocks / max(minutes / 90, 1))
            interventions_per_90 = player_data.get('interventions_per_90', interventions / max(minutes / 90, 1))
            passes_per_90 = player_data.get('passes_per_90', passes / max(minutes / 90, 1))
        
        # Scale to expected minutes
        minutes_factor = minutes / 90.0
        
        # Calculate floor points
        floor_points = 0.0
        
        # Block points
        if rules['block'] > 0:
            floor_points += blocks_per_90 * rules['block'] * minutes_factor
        
        # Intervention points
        floor_points += interventions_per_90 * rules['intervention'] * minutes_factor
        
        # Pass bonus (per 10 successful passes)
        floor_points += (passes_per_90 / 10.0) * rules['pass_bonus'] * minutes_factor
        
        return float(max(0.0, floor_points))
    
    def extract_defcon_features(self, player_data: Dict, position: str) -> Dict[str, float]:
        """
        Extract all DefCon-related features for a player.
        
        Returns:
            Dictionary with DefCon features
        """
        minutes = player_data.get('minutes', player_data.get('expected_minutes', 90))
        
        return {
            'floor_points': self.calculate_floor_points(player_data, position, minutes),
            'blocks_per_90': player_data.get('blocks_per_90', 0.0),
            'interventions_per_90': player_data.get('interventions_per_90', 0.0),
            'passes_per_90': player_data.get('passes_per_90', 0.0),
            'defcon_score': self.calculate_floor_points(player_data, position, 90)  # Full match baseline
        }


class FeatureEngineeringService:
    """
    Main service that orchestrates all feature engineering components.
    """
    
    def __init__(self):
        self.form_alpha = DynamicFormAlpha()
        self.fdr_model = DixonColesFDR()
        self.defcon_engine = DefConFeatureEngine()
    
    def calculate_all_features(
        self,
        player_data: Dict,
        historical_points: List[float],
        fixture_data: Optional[Dict] = None,
        position: str = "MID"
    ) -> Dict[str, float]:
        """
        Calculate all engineered features for a player.
        
        Args:
            player_data: Current player statistics
            historical_points: Historical points (most recent first)
            fixture_data: Upcoming fixture information
            position: Player position
        
        Returns:
            Dictionary of all engineered features
        """
        features = {}
        
        # 1. Dynamic Form Alpha
        if historical_points:
            form = self.form_alpha.get_form(historical_points)
            features['dynamic_form'] = form
            features['form_trend'] = self._calculate_trend(historical_points)
        else:
            features['dynamic_form'] = 0.0
            features['form_trend'] = 0.0
        
        # 2. Dynamic FDR
        if fixture_data and self.fdr_model.is_fitted:
            team = player_data.get('team', '')
            opponent = fixture_data.get('opponent', '')
            is_home = fixture_data.get('is_home', True)
            fdr = self.fdr_model.get_fixture_difficulty(team, opponent, is_home)
            features['fdr'] = fdr
            features['fdr_attack'] = self.fdr_model.attack_strengths.get(opponent, 0.0)
            features['fdr_defense'] = self.fdr_model.defense_strengths.get(opponent, 0.0)
        else:
            features['fdr'] = 3.0  # Neutral
            features['fdr_attack'] = 0.0
            features['fdr_defense'] = 0.0
        
        # 3. DefCon Features
        defcon_features = self.defcon_engine.extract_defcon_features(player_data, position)
        features.update(defcon_features)
        
        return features
    
    def _calculate_trend(self, historical_points: List[float], weeks: int = 3) -> float:
        """Calculate form trend (positive = improving, negative = declining)"""
        if len(historical_points) < weeks:
            return 0.0
        
        recent = np.mean(historical_points[:weeks])
        previous = np.mean(historical_points[weeks:weeks*2]) if len(historical_points) >= weeks*2 else recent
        
        return float(recent - previous)
    
    def optimize_form_alpha(self, historical_data: pd.DataFrame):
        """Optimize form alpha for all players"""
        return self.form_alpha.optimize_alpha(historical_data)
    
    def fit_fdr_model(self, fixtures: List[Dict]):
        """Fit Dixon-Coles FDR model"""
        self.fdr_model.fit(fixtures)