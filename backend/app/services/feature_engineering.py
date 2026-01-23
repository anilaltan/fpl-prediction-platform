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
from sqlalchemy.orm import Session
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
        self, historical_points: List[float], alpha: float, lookback_weeks: int = 5
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
        weights = [alpha**i for i in range(n)]
        total_weight = sum(weights)

        if total_weight == 0:
            return np.mean(recent_points)

        weighted_sum = sum(p * w for p, w in zip(recent_points, weights))
        return weighted_sum / total_weight

    def _calculate_rmse(
        self, alpha: float, historical_data: pd.DataFrame, lookback_weeks: int = 5
    ) -> float:
        """
        Calculate RMSE for a given alpha value.
        Uses expanding window approach for validation.
        """
        errors = []

        # Expanding window: train on weeks 1 to i, predict week i+1
        for i in range(lookback_weeks, len(historical_data)):
            train_data = historical_data.iloc[:i]
            actual = historical_data.iloc[i]["points"]

            # Calculate form using historical points up to week i
            historical_points = train_data["points"].tolist()
            predicted_form = self.calculate_form(
                historical_points, alpha, lookback_weeks
            )

            # Simple prediction: form * minutes_factor (if available)
            minutes_factor = historical_data.iloc[i].get("minutes", 90) / 90.0
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
        n_calls: int = 50,
        convergence_threshold: float = 0.001,
        patience: int = 10,
    ) -> Dict:
        """
        Use Bayesian Optimization to find optimal alpha that minimizes RMSE.
        Tracks convergence and optimization history.

        Args:
            historical_data: DataFrame with columns ['points', 'minutes', ...]
            lookback_weeks: Number of weeks to consider for form
            n_calls: Number of optimization iterations
            convergence_threshold: Minimum improvement to consider converged
            patience: Number of iterations without improvement before early stopping

        Returns:
            Dictionary with:
            - optimal_alpha: Optimal alpha value
            - best_rmse: Best RMSE achieved
            - converged: Whether optimization converged
            - iterations: Number of iterations run
            - optimization_history: List of (alpha, rmse) pairs
        """
        if len(historical_data) < lookback_weeks + 1:
            logger.warning(
                "Insufficient data for alpha optimization, using default 0.5"
            )
            self.optimal_alpha = 0.5
            default_rmse = self._calculate_rmse(0.5, historical_data, lookback_weeks)
            return {
                "optimal_alpha": 0.5,
                "best_rmse": default_rmse,
                "converged": False,
                "iterations": 0,
                "optimization_history": [],
            }

        # Track optimization history
        optimization_history = []
        best_rmse_so_far = float("inf")
        no_improvement_count = 0

        # Define search space
        space = [Real(self.min_alpha, self.max_alpha, name="alpha")]

        # Objective function with history tracking
        @use_named_args(dimensions=space)
        def objective(alpha):
            nonlocal best_rmse_so_far, no_improvement_count
            rmse = self._calculate_rmse(alpha, historical_data, lookback_weeks)

            # Track history
            optimization_history.append(
                {
                    "alpha": alpha,
                    "rmse": rmse,
                    "iteration": len(optimization_history) + 1,
                }
            )

            # Check for improvement
            if rmse < best_rmse_so_far - convergence_threshold:
                best_rmse_so_far = rmse
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            return rmse

        # Run Bayesian Optimization
        logger.info(
            f"Starting Bayesian Optimization for alpha (n_calls={n_calls}, lookback={lookback_weeks})"
        )

        try:
            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=n_calls,
                random_state=42,
                acq_func="EI",  # Expected Improvement
                n_initial_points=5,  # Initial random points for exploration
                callback=None,  # Can add custom callback for early stopping
            )

            self.optimal_alpha = result.x[0]
            best_rmse = result.fun

            # Check convergence
            converged = False
            if len(optimization_history) >= patience:
                # Check if last 'patience' iterations showed minimal improvement
                recent_improvements = [
                    optimization_history[i]["rmse"]
                    - optimization_history[i - 1]["rmse"]
                    for i in range(
                        len(optimization_history) - patience + 1,
                        len(optimization_history),
                    )
                    if i > 0
                ]
                if all(abs(imp) < convergence_threshold for imp in recent_improvements):
                    converged = True

            # Store optimization history
            self.optimization_history = optimization_history

            logger.info(
                f"Optimization complete: alpha={self.optimal_alpha:.4f}, "
                f"RMSE={best_rmse:.4f}, converged={converged}, "
                f"iterations={len(optimization_history)}"
            )

            return {
                "optimal_alpha": self.optimal_alpha,
                "best_rmse": best_rmse,
                "converged": converged,
                "iterations": len(optimization_history),
                "optimization_history": optimization_history,
            }

        except Exception as e:
            logger.error(f"Error during Bayesian Optimization: {str(e)}", exc_info=True)
            # Fallback to default
            self.optimal_alpha = 0.5
            default_rmse = self._calculate_rmse(0.5, historical_data, lookback_weeks)
            return {
                "optimal_alpha": 0.5,
                "best_rmse": default_rmse,
                "converged": False,
                "iterations": 0,
                "optimization_history": [],
                "error": str(e),
            }

    def get_form(
        self, historical_points: List[float], lookback_weeks: int = 5
    ) -> float:
        """Get form using optimal alpha (or default if not optimized)"""
        alpha = self.optimal_alpha if self.optimal_alpha is not None else 0.5
        return self.calculate_form(historical_points, alpha, lookback_weeks)

    def load_optimal_alpha_from_db(
        self, db: Session, gameweek: Optional[int] = None
    ) -> Optional[float]:
        """
        Load optimal alpha from database for a specific gameweek.

        Args:
            db: Database session
            gameweek: Gameweek to load alpha for (uses latest if None)

        Returns:
            Optimal alpha value or None if not found
        """
        try:
            from app.models import FormAlpha

            if gameweek:
                entry = (
                    db.query(FormAlpha).filter(FormAlpha.gameweek == gameweek).first()
                )
            else:
                # Get latest entry
                entry = db.query(FormAlpha).order_by(FormAlpha.gameweek.desc()).first()

            if entry:
                self.optimal_alpha = entry.optimal_alpha
                logger.info(
                    f"Loaded optimal alpha {self.optimal_alpha:.4f} from database (GW {entry.gameweek})"
                )
                return self.optimal_alpha

            return None
        except Exception as e:
            logger.warning(f"Failed to load alpha from database: {str(e)}")
            return None

    def compare_with_baseline(
        self,
        historical_data: pd.DataFrame,
        lookback_weeks: int = 5,
        baseline_alpha: float = 0.5,
    ) -> Dict:
        """
        Compare optimized alpha performance with baseline.

        Args:
            historical_data: DataFrame with historical points
            lookback_weeks: Number of weeks for form calculation
            baseline_alpha: Baseline alpha value to compare against

        Returns:
            Dictionary with comparison metrics
        """
        if self.optimal_alpha is None:
            logger.warning("No optimized alpha available, running optimization first")
            self.optimize_alpha(historical_data, lookback_weeks)

        optimized_rmse = self._calculate_rmse(
            self.optimal_alpha, historical_data, lookback_weeks
        )
        baseline_rmse = self._calculate_rmse(
            baseline_alpha, historical_data, lookback_weeks
        )

        improvement = (
            ((baseline_rmse - optimized_rmse) / baseline_rmse) * 100
            if baseline_rmse > 0
            else 0
        )

        return {
            "optimized_alpha": self.optimal_alpha,
            "baseline_alpha": baseline_alpha,
            "optimized_rmse": optimized_rmse,
            "baseline_rmse": baseline_rmse,
            "improvement_percent": improvement,
            "improves_baseline": optimized_rmse < baseline_rmse,
        }


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
            if "goals_h" not in fixture or "goals_a" not in fixture:
                continue

            data.append(
                {
                    "team_h": fixture.get("team_h", fixture.get("team_h_name", "")),
                    "team_a": fixture.get("team_a", fixture.get("team_a_name", "")),
                    "goals_h": int(fixture.get("goals_h", 0)),
                    "goals_a": int(fixture.get("goals_a", 0)),
                    "is_home": 1,
                }
            )

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
        teams = sorted(set(df["team_h"].unique()) | set(df["team_a"].unique()))

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
            home_idx = teams.index(row["team_h"])
            home_features[1 + home_idx] = 1.0

            # Set defense strength for away team
            away_idx = teams.index(row["team_a"])
            home_features[1 + len(teams) + away_idx] = -1.0

            X_home.append(home_features)
            y_home.append(row["goals_h"])

        X_home = np.array(X_home)
        y_home = np.array(y_home)

        # Fit Poisson model for home goals
        try:
            poisson_model_home = Poisson(y_home, X_home)
            result_home = poisson_model_home.fit(method="lbfgs", maxiter=1000)

            params = result_home.params

            # Extract coefficients
            self.home_advantage = params[0]

            # Attack strengths (offset by average)
            attack_params = params[1 : 1 + len(teams)]
            avg_attack = np.mean(attack_params)
            self.attack_strengths = {
                team: float(attack_params[i] - avg_attack)
                for i, team in enumerate(teams)
            }

            # Defense strengths (offset by average)
            defense_params = params[1 + len(teams) :]
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
            home_goals = df[df["team_h"] == team]["goals_h"].mean()
            away_goals = df[df["team_a"] == team]["goals_a"].mean()
            goals_for = (home_goals + away_goals) / 2

            home_conceded = df[df["team_h"] == team]["goals_a"].mean()
            away_conceded = df[df["team_a"] == team]["goals_h"].mean()
            goals_against = (home_conceded + away_conceded) / 2

            self.attack_strengths[team] = float(goals_for - 1.5)  # Normalize
            self.defense_strengths[team] = float(1.5 - goals_against)  # Normalize

        self.home_advantage = 0.2
        self.is_fitted = True

    def get_expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
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

    def get_fixture_difficulty(self, team: str, opponent: str, is_home: bool) -> float:
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

    def get_stochastic_fdr(
        self, team: str, opponent: str, is_home: bool, n_simulations: int = 10000
    ) -> Dict:
        """
        Calculate stochastic fixture difficulty (FDR 2.0) using Poisson distribution.
        Provides probability distribution of outcomes rather than just expected value.

        Args:
            team: Team name
            opponent: Opponent team name
            is_home: Whether team is playing at home
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with:
            - fdr_mean: Mean FDR (1-5 scale)
            - fdr_std: Standard deviation of FDR
            - win_prob: Probability of winning
            - draw_prob: Probability of drawing
            - loss_prob: Probability of losing
            - clean_sheet_prob: Probability of clean sheet
            - expected_goals_for: Expected goals scored
            - expected_goals_against: Expected goals conceded
            - goal_distribution: Probability distribution of goals
        """
        if not self.is_fitted:
            return {
                "fdr_mean": 3.0,
                "fdr_std": 0.0,
                "win_prob": 0.33,
                "draw_prob": 0.34,
                "loss_prob": 0.33,
                "clean_sheet_prob": 0.25,
                "expected_goals_for": 1.5,
                "expected_goals_against": 1.5,
                "goal_distribution": {},
            }

        # Get expected goals
        if is_home:
            exp_goals_for, exp_goals_against = self.get_expected_goals(team, opponent)
        else:
            exp_goals_away, exp_goals_home = self.get_expected_goals(opponent, team)
            exp_goals_for = exp_goals_away
            exp_goals_against = exp_goals_home

        # Use Poisson distribution for goal probabilities
        max_goals = 6  # Consider up to 6 goals
        goal_probs_for = [poisson.pmf(i, exp_goals_for) for i in range(max_goals + 1)]
        goal_probs_against = [
            poisson.pmf(i, exp_goals_against) for i in range(max_goals + 1)
        ]

        # Calculate outcome probabilities
        win_prob = 0.0
        draw_prob = 0.0
        loss_prob = 0.0
        clean_sheet_prob = goal_probs_against[0]

        for goals_for in range(max_goals + 1):
            for goals_against in range(max_goals + 1):
                prob = goal_probs_for[goals_for] * goal_probs_against[goals_against]
                if goals_for > goals_against:
                    win_prob += prob
                elif goals_for == goals_against:
                    draw_prob += prob
                else:
                    loss_prob += prob

        # Normalize probabilities (should sum to ~1, but account for truncation)
        total_prob = win_prob + draw_prob + loss_prob
        if total_prob > 0:
            win_prob /= total_prob
            draw_prob /= total_prob
            loss_prob /= total_prob

        # Calculate FDR based on expected goals against (stochastic)
        # FDR = 1 + (expected_goals_against / 2.0) normalized to 1-5
        fdr_mean = float(np.clip(1 + (exp_goals_against / 2.0), 1.0, 5.0))

        # Calculate FDR variance based on goal distribution variance
        # Higher variance in goals against = higher uncertainty in FDR
        goal_variance = exp_goals_against  # Poisson variance equals mean
        fdr_std = float(np.sqrt(goal_variance) / 2.0)  # Scale to FDR units

        # Goal distribution for analysis
        goal_distribution = {
            "goals_for": {i: float(goal_probs_for[i]) for i in range(max_goals + 1)},
            "goals_against": {
                i: float(goal_probs_against[i]) for i in range(max_goals + 1)
            },
        }

        return {
            "fdr_mean": fdr_mean,
            "fdr_std": fdr_std,
            "win_prob": float(win_prob),
            "draw_prob": float(draw_prob),
            "loss_prob": float(loss_prob),
            "clean_sheet_prob": float(clean_sheet_prob),
            "expected_goals_for": float(exp_goals_for),
            "expected_goals_against": float(exp_goals_against),
            "goal_distribution": goal_distribution,
        }

    def compare_with_fpl_fdr(
        self, fixtures: List[Dict], fpl_fdr_data: Optional[Dict] = None
    ) -> Dict:
        """
        Compare FDR 2.0 ratings with official FPL FDR.

        Args:
            fixtures: List of fixtures with FPL FDR data
            fpl_fdr_data: Optional dict mapping (team, opponent, is_home) -> FPL FDR

        Returns:
            Dictionary with comparison metrics:
            - correlation: Correlation coefficient
            - mean_absolute_error: MAE between FDR 2.0 and FPL FDR
            - r_squared: R-squared value
            - comparison_data: List of comparison points
        """
        if not self.is_fitted:
            return {
                "correlation": 0.0,
                "mean_absolute_error": 0.0,
                "r_squared": 0.0,
                "comparison_data": [],
            }

        comparison_data = []

        for fixture in fixtures:
            team_h = fixture.get("team_h", fixture.get("team_h_name", ""))
            team_a = fixture.get("team_a", fixture.get("team_a_name", ""))

            if not team_h or not team_a:
                continue

            # Get FDR 2.0 for home team
            fdr2_home = self.get_stochastic_fdr(team_h, team_a, is_home=True)

            # Get FDR 2.0 for away team
            fdr2_away = self.get_stochastic_fdr(team_a, team_h, is_home=False)

            # Get FPL FDR if available
            fpl_fdr_home = fixture.get("team_h_difficulty", fixture.get("fpl_fdr_home"))
            fpl_fdr_away = fixture.get("team_a_difficulty", fixture.get("fpl_fdr_away"))

            if fpl_fdr_home is not None:
                comparison_data.append(
                    {
                        "team": team_h,
                        "opponent": team_a,
                        "is_home": True,
                        "fdr2": fdr2_home["fdr_mean"],
                        "fpl_fdr": float(fpl_fdr_home),
                    }
                )

            if fpl_fdr_away is not None:
                comparison_data.append(
                    {
                        "team": team_a,
                        "opponent": team_h,
                        "is_home": False,
                        "fdr2": fdr2_away["fdr_mean"],
                        "fpl_fdr": float(fpl_fdr_away),
                    }
                )

        if len(comparison_data) < 2:
            return {
                "correlation": 0.0,
                "mean_absolute_error": 0.0,
                "r_squared": 0.0,
                "comparison_data": comparison_data,
            }

        # Calculate metrics
        fdr2_values = [d["fdr2"] for d in comparison_data]
        fpl_fdr_values = [d["fpl_fdr"] for d in comparison_data]

        # Correlation
        correlation = float(np.corrcoef(fdr2_values, fpl_fdr_values)[0, 1])

        # Mean Absolute Error
        mae = float(np.mean(np.abs(np.array(fdr2_values) - np.array(fpl_fdr_values))))

        # R-squared
        ss_res = np.sum((np.array(fpl_fdr_values) - np.array(fdr2_values)) ** 2)
        ss_tot = np.sum((np.array(fpl_fdr_values) - np.mean(fpl_fdr_values)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            "correlation": correlation,
            "mean_absolute_error": mae,
            "r_squared": r_squared,
            "comparison_data": comparison_data,
        }

    def verify_with_actual_outcomes(self, fixtures: List[Dict]) -> Dict:
        """
        Verify FDR 2.0 predictions correlate with actual goal outcomes.

        Args:
            fixtures: List of fixtures with actual goals scored

        Returns:
            Dictionary with verification metrics
        """
        if not self.is_fitted:
            return {
                "correlation_goals_for": 0.0,
                "correlation_goals_against": 0.0,
                "prediction_accuracy": 0.0,
            }

        predictions = []
        actuals = []

        for fixture in fixtures:
            team_h = fixture.get("team_h", fixture.get("team_h_name", ""))
            team_a = fixture.get("team_a", fixture.get("team_a_name", ""))
            goals_h = fixture.get("goals_h")
            goals_a = fixture.get("goals_a")

            if not team_h or not team_a or goals_h is None or goals_a is None:
                continue

            # Get predictions
            fdr2_home = self.get_stochastic_fdr(team_h, team_a, is_home=True)
            fdr2_away = self.get_stochastic_fdr(team_a, team_h, is_home=False)

            predictions.append(
                {
                    "home_goals_for": fdr2_home["expected_goals_for"],
                    "home_goals_against": fdr2_home["expected_goals_against"],
                    "away_goals_for": fdr2_away["expected_goals_for"],
                    "away_goals_against": fdr2_away["expected_goals_against"],
                }
            )

            actuals.append(
                {
                    "home_goals_for": float(goals_h),
                    "home_goals_against": float(goals_a),
                    "away_goals_for": float(goals_a),
                    "away_goals_against": float(goals_h),
                }
            )

        if len(predictions) < 2:
            return {
                "correlation_goals_for": 0.0,
                "correlation_goals_against": 0.0,
                "prediction_accuracy": 0.0,
            }

        # Calculate correlations
        pred_goals_for = [p["home_goals_for"] for p in predictions] + [
            p["away_goals_for"] for p in predictions
        ]
        actual_goals_for = [a["home_goals_for"] for a in actuals] + [
            a["away_goals_for"] for a in actuals
        ]

        pred_goals_against = [p["home_goals_against"] for p in predictions] + [
            p["away_goals_against"] for p in predictions
        ]
        actual_goals_against = [a["home_goals_against"] for a in actuals] + [
            a["away_goals_against"] for a in actuals
        ]

        corr_for = (
            float(np.corrcoef(pred_goals_for, actual_goals_for)[0, 1])
            if len(pred_goals_for) > 1
            else 0.0
        )
        corr_against = (
            float(np.corrcoef(pred_goals_against, actual_goals_against)[0, 1])
            if len(pred_goals_against) > 1
            else 0.0
        )

        # Calculate prediction accuracy (within 0.5 goals)
        accuracy = 0.0
        if len(predictions) > 0:
            correct = 0
            total = 0
            for pred, actual in zip(predictions, actuals):
                if abs(pred["home_goals_for"] - actual["home_goals_for"]) < 0.5:
                    correct += 1
                total += 1
                if abs(pred["home_goals_against"] - actual["home_goals_against"]) < 0.5:
                    correct += 1
                total += 1
            accuracy = correct / total if total > 0 else 0.0

        return {
            "correlation_goals_for": corr_for,
            "correlation_goals_against": corr_against,
            "prediction_accuracy": float(accuracy),
            "n_fixtures": len(predictions),
        }


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
            "defender": {
                "block": 1,  # 1 point per block
                "intervention": 1,  # 1 point per intervention
                "pass_bonus": 0.1,  # 0.1 points per 10 successful passes
            },
            "midfielder": {"block": 1, "intervention": 1, "pass_bonus": 0.1},
            "forward": {
                "block": 0,  # Forwards don't get block points
                "intervention": 1,
                "pass_bonus": 0.1,
            },
        }

    def calculate_floor_points(
        self, player_data: Dict, position: str, minutes: int = 90
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
        if "def" in position_lower:
            rules = self.defcon_rules["defender"]
        elif "mid" in position_lower:
            rules = self.defcon_rules["midfielder"]
        else:
            rules = self.defcon_rules["forward"]

        # Extract DefCon stats (if available from API)
        blocks = player_data.get("blocks", player_data.get("total_blocks", 0))
        interventions = player_data.get(
            "interceptions", player_data.get("total_interceptions", 0)
        )
        passes = player_data.get("passes", player_data.get("total_passes", 0))

        # Calculate per-90 averages if we have historical data
        if "minutes_played" in player_data and player_data["minutes_played"] > 0:
            games_played = player_data.get("games_played", 1)
            blocks_per_90 = (blocks / games_played) * (
                90 / max(player_data["minutes_played"] / games_played, 1)
            )
            interventions_per_90 = (interventions / games_played) * (
                90 / max(player_data["minutes_played"] / games_played, 1)
            )
            passes_per_90 = (passes / games_played) * (
                90 / max(player_data["minutes_played"] / games_played, 1)
            )
        else:
            # Use season averages if available
            blocks_per_90 = player_data.get(
                "blocks_per_90", blocks / max(minutes / 90, 1)
            )
            interventions_per_90 = player_data.get(
                "interventions_per_90", interventions / max(minutes / 90, 1)
            )
            passes_per_90 = player_data.get(
                "passes_per_90", passes / max(minutes / 90, 1)
            )

        # Scale to expected minutes
        minutes_factor = minutes / 90.0

        # Calculate floor points
        floor_points = 0.0

        # Block points
        if rules["block"] > 0:
            floor_points += blocks_per_90 * rules["block"] * minutes_factor

        # Intervention points
        floor_points += interventions_per_90 * rules["intervention"] * minutes_factor

        # Pass bonus (per 10 successful passes)
        floor_points += (passes_per_90 / 10.0) * rules["pass_bonus"] * minutes_factor

        return float(max(0.0, floor_points))

    def extract_defcon_features(
        self, player_data: Dict, position: str
    ) -> Dict[str, float]:
        """
        Extract all DefCon-related features for a player.

        Returns:
            Dictionary with DefCon features
        """
        minutes = player_data.get("minutes", player_data.get("expected_minutes", 90))

        return {
            "floor_points": self.calculate_floor_points(player_data, position, minutes),
            "blocks_per_90": player_data.get("blocks_per_90", 0.0),
            "interventions_per_90": player_data.get("interventions_per_90", 0.0),
            "passes_per_90": player_data.get("passes_per_90", 0.0),
            "defcon_score": self.calculate_floor_points(
                player_data, position, 90
            ),  # Full match baseline
        }


class FeatureEngineeringService:
    """
    Main service that orchestrates all feature engineering components.
    Integrates advanced features: xG/xA from Understat, FDR 2.0, optimized alpha.
    """

    def __init__(self, third_party_service=None, db_session: Optional[Session] = None):
        self.form_alpha = DynamicFormAlpha()
        self.fdr_model = DixonColesFDR()
        self.defcon_engine = DefConFeatureEngine()
        self.third_party_service = third_party_service
        self.db_session = db_session

    def calculate_all_features(
        self,
        player_data: Dict,
        historical_points: List[float],
        fixture_data: Optional[Dict] = None,
        position: str = "MID",
        third_party_data: Optional[Dict] = None,
        use_stochastic_fdr: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate all engineered features for a player.
        Integrates advanced features: xG/xA from Understat, FDR 2.0, optimized alpha.

        Args:
            player_data: Current player statistics
            historical_points: Historical points (most recent first)
            fixture_data: Upcoming fixture information
            position: Player position
            third_party_data: Optional dict with Understat/FBref data (from ThirdPartyDataService)
            use_stochastic_fdr: Whether to use stochastic FDR 2.0 (default: True)

        Returns:
            Dictionary of all engineered features (no null values)
        """
        features = {}

        # 1. Dynamic Form Alpha (using optimized alpha)
        if historical_points:
            # Ensure optimal alpha is loaded/optimized
            if self.form_alpha.optimal_alpha is None:
                # Use default if not optimized
                logger.debug("Form alpha not optimized, using default 0.5")

            form = self.form_alpha.get_form(historical_points)
            features["dynamic_form"] = form
            features["form_trend"] = self._calculate_trend(historical_points)
            features["form_alpha_used"] = (
                self.form_alpha.optimal_alpha
                if self.form_alpha.optimal_alpha is not None
                else 0.5
            )
        else:
            features["dynamic_form"] = 0.0
            features["form_trend"] = 0.0
            features["form_alpha_used"] = 0.5

        # 2. FDR 2.0 (Stochastic Fixture Difficulty)
        if fixture_data and self.fdr_model.is_fitted:
            team = player_data.get("team", player_data.get("team_name", ""))
            opponent = fixture_data.get(
                "opponent", fixture_data.get("opponent_team", "")
            )
            is_home = fixture_data.get("is_home", True)

            if use_stochastic_fdr and team and opponent:
                # Use stochastic FDR 2.0
                fdr2 = self.fdr_model.get_stochastic_fdr(team, opponent, is_home)
                features["fdr"] = fdr2["fdr_mean"]
                features["fdr_std"] = fdr2["fdr_std"]
                features["fdr_win_prob"] = fdr2["win_prob"]
                features["fdr_draw_prob"] = fdr2["draw_prob"]
                features["fdr_loss_prob"] = fdr2["loss_prob"]
                features["fdr_clean_sheet_prob"] = fdr2["clean_sheet_prob"]
                features["fdr_expected_goals_for"] = fdr2["expected_goals_for"]
                features["fdr_expected_goals_against"] = fdr2["expected_goals_against"]
            else:
                # Fallback to basic FDR
                fdr = self.fdr_model.get_fixture_difficulty(team, opponent, is_home)
                features["fdr"] = fdr
                features["fdr_std"] = 0.0
                features["fdr_win_prob"] = 0.33
                features["fdr_draw_prob"] = 0.34
                features["fdr_loss_prob"] = 0.33
                features["fdr_clean_sheet_prob"] = 0.25
                features["fdr_expected_goals_for"] = 1.5
                features["fdr_expected_goals_against"] = 1.5

            # Attack/defense strengths
            features["fdr_attack"] = self.fdr_model.attack_strengths.get(opponent, 0.0)
            features["fdr_defense"] = self.fdr_model.defense_strengths.get(
                opponent, 0.0
            )
            features["team_attack"] = self.fdr_model.attack_strengths.get(team, 0.0)
            features["team_defense"] = self.fdr_model.defense_strengths.get(team, 0.0)
        else:
            # Default neutral values
            features["fdr"] = 3.0
            features["fdr_std"] = 0.0
            features["fdr_win_prob"] = 0.33
            features["fdr_draw_prob"] = 0.34
            features["fdr_loss_prob"] = 0.33
            features["fdr_clean_sheet_prob"] = 0.25
            features["fdr_expected_goals_for"] = 1.5
            features["fdr_expected_goals_against"] = 1.5
            features["fdr_attack"] = 0.0
            features["fdr_defense"] = 0.0
            features["team_attack"] = 0.0
            features["team_defense"] = 0.0

        # 3. External Data Integration (xG/xA from Understat)
        if third_party_data:
            # Understat xG/xA metrics
            features["understat_xg"] = third_party_data.get(
                "understat_xg", player_data.get("xg", 0.0)
            )
            features["understat_xa"] = third_party_data.get(
                "understat_xa", player_data.get("xa", 0.0)
            )
            features["understat_npxg"] = third_party_data.get("understat_npxg", 0.0)
            features["understat_xg_per_90"] = third_party_data.get(
                "understat_xg_per_90", player_data.get("xg_per_90", 0.0)
            )
            features["understat_xa_per_90"] = third_party_data.get(
                "understat_xa_per_90", player_data.get("xa_per_90", 0.0)
            )
            features["understat_npxg_per_90"] = third_party_data.get(
                "understat_npxg_per_90", 0.0
            )

            # FBref defensive metrics
            features["fbref_blocks"] = third_party_data.get("fbref_blocks", 0)
            features["fbref_blocks_per_90"] = third_party_data.get(
                "fbref_blocks_per_90", 0.0
            )
            features["fbref_interventions"] = third_party_data.get(
                "fbref_interventions", 0
            )
            features["fbref_interventions_per_90"] = third_party_data.get(
                "fbref_interventions_per_90", 0.0
            )
            features["fbref_passes"] = third_party_data.get("fbref_passes", 0)
            features["fbref_passes_per_90"] = third_party_data.get(
                "fbref_passes_per_90", 0.0
            )
        else:
            # Use player_data as fallback
            features["understat_xg"] = player_data.get("xg", 0.0)
            features["understat_xa"] = player_data.get("xa", 0.0)
            features["understat_npxg"] = player_data.get("npxg", 0.0)
            features["understat_xg_per_90"] = player_data.get("xg_per_90", 0.0)
            features["understat_xa_per_90"] = player_data.get("xa_per_90", 0.0)
            features["understat_npxg_per_90"] = player_data.get("npxg_per_90", 0.0)
            features["fbref_blocks"] = 0
            features["fbref_blocks_per_90"] = 0.0
            features["fbref_interventions"] = 0
            features["fbref_interventions_per_90"] = 0.0
            features["fbref_passes"] = 0
            features["fbref_passes_per_90"] = 0.0

        # 4. DefCon Features
        defcon_features = self.defcon_engine.extract_defcon_features(
            player_data, position
        )
        features.update(defcon_features)

        # 5. Validate and fill null values
        features = self._validate_and_fill_features(features)

        return features

    def _validate_and_fill_features(
        self, features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Validate features and fill any null/NaN values with defaults.
        Ensures no null values in the feature set.

        Args:
            features: Dictionary of features

        Returns:
            Validated features dictionary with no null values
        """
        validated = {}
        defaults = {
            "dynamic_form": 0.0,
            "form_trend": 0.0,
            "form_alpha_used": 0.5,
            "fdr": 3.0,
            "fdr_std": 0.0,
            "fdr_win_prob": 0.33,
            "fdr_draw_prob": 0.34,
            "fdr_loss_prob": 0.33,
            "fdr_clean_sheet_prob": 0.25,
            "fdr_expected_goals_for": 1.5,
            "fdr_expected_goals_against": 1.5,
            "fdr_attack": 0.0,
            "fdr_defense": 0.0,
            "team_attack": 0.0,
            "team_defense": 0.0,
            "understat_xg": 0.0,
            "understat_xa": 0.0,
            "understat_npxg": 0.0,
            "understat_xg_per_90": 0.0,
            "understat_xa_per_90": 0.0,
            "understat_npxg_per_90": 0.0,
            "fbref_blocks": 0,
            "fbref_blocks_per_90": 0.0,
            "fbref_interventions": 0,
            "fbref_interventions_per_90": 0.0,
            "fbref_passes": 0,
            "fbref_passes_per_90": 0.0,
        }

        for key, value in features.items():
            # Check for None, NaN, or invalid values
            if value is None or (
                isinstance(value, float) and (np.isnan(value) or np.isinf(value))
            ):
                validated[key] = defaults.get(key, 0.0)
            else:
                validated[key] = (
                    float(value) if isinstance(value, (int, float, np.number)) else 0.0
                )

        return validated

    def _calculate_trend(self, historical_points: List[float], weeks: int = 3) -> float:
        """Calculate form trend (positive = improving, negative = declining)"""
        if len(historical_points) < weeks:
            return 0.0

        recent = np.mean(historical_points[:weeks])
        previous = (
            np.mean(historical_points[weeks : weeks * 2])
            if len(historical_points) >= weeks * 2
            else recent
        )

        return float(recent - previous)

    def optimize_form_alpha(
        self, historical_data: pd.DataFrame, lookback_weeks: int = 5, n_calls: int = 50
    ) -> Dict:
        """
        Optimize form alpha for all players using Bayesian Optimization.

        Returns:
            Dictionary with optimization results including optimal_alpha, best_rmse, etc.
        """
        return self.form_alpha.optimize_alpha(
            historical_data, lookback_weeks=lookback_weeks, n_calls=n_calls
        )

    def fit_fdr_model(self, fixtures: List[Dict]):
        """Fit Dixon-Coles FDR model"""
        self.fdr_model.fit(fixtures)

    def enrich_training_data(
        self,
        training_data: pd.DataFrame,
        load_optimal_alpha: bool = True,
        gameweek: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Enrich training data with advanced features (xG/xA, FDR 2.0, optimized alpha).
        Ensures all features are properly aligned and have no null values.

        Args:
            training_data: Training DataFrame with player data
            load_optimal_alpha: Whether to load optimal alpha from database (default: True)
            gameweek: Gameweek to load alpha for (uses latest if None)

        Returns:
            Enriched DataFrame with additional feature columns
        """
        # Load optimal alpha if requested
        if load_optimal_alpha and self.db_session:
            try:
                self.form_alpha.load_optimal_alpha_from_db(self.db_session, gameweek)
            except Exception as e:
                logger.warning(f"Failed to load optimal alpha: {str(e)}")

        # Ensure FDR model is fitted (should be done before calling this)
        if not self.fdr_model.is_fitted:
            logger.warning("FDR model not fitted. FDR features will use defaults.")

        # Create enriched data list
        enriched_rows = []

        for idx, row in training_data.iterrows():
            player_data = row.to_dict()

            # Get historical points for form calculation
            # This should be calculated from previous gameweeks in training data
            historical_points = []  # Will be populated from training data if available

            # Get fixture data
            fixture_data = {
                "opponent": player_data.get(
                    "opponent_team", player_data.get("opponent", "")
                ),
                "is_home": bool(
                    player_data.get("was_home", player_data.get("is_home", True))
                ),
                "team": player_data.get("team_name", player_data.get("team", "")),
            }

            # Get third-party data if available
            third_party_data = None
            if self.third_party_service:
                # Try to get enriched data (this would need player ID mapping)
                # For now, extract from player_data if available
                third_party_data = {
                    "understat_xg": player_data.get(
                        "understat_xg", player_data.get("xg", 0.0)
                    ),
                    "understat_xa": player_data.get(
                        "understat_xa", player_data.get("xa", 0.0)
                    ),
                    "understat_npxg": player_data.get("understat_npxg", 0.0),
                    "understat_xg_per_90": player_data.get(
                        "understat_xg_per_90", player_data.get("xg_per_90", 0.0)
                    ),
                    "understat_xa_per_90": player_data.get(
                        "understat_xa_per_90", player_data.get("xa_per_90", 0.0)
                    ),
                    "understat_npxg_per_90": player_data.get(
                        "understat_npxg_per_90", 0.0
                    ),
                    "fbref_blocks": player_data.get("fbref_blocks", 0),
                    "fbref_blocks_per_90": player_data.get("fbref_blocks_per_90", 0.0),
                    "fbref_interventions": player_data.get("fbref_interventions", 0),
                    "fbref_interventions_per_90": player_data.get(
                        "fbref_interventions_per_90", 0.0
                    ),
                    "fbref_passes": player_data.get("fbref_passes", 0),
                    "fbref_passes_per_90": player_data.get("fbref_passes_per_90", 0.0),
                }

            # Calculate all features
            position = player_data.get("position", "MID")
            features = self.calculate_all_features(
                player_data=player_data,
                historical_points=historical_points,
                fixture_data=fixture_data,
                position=position,
                third_party_data=third_party_data,
                use_stochastic_fdr=True,
            )

            # Merge features into player data
            enriched_row = {**player_data, **features}
            enriched_rows.append(enriched_row)

        # Create enriched DataFrame
        enriched_df = pd.DataFrame(enriched_rows)

        # Validate no null values
        null_counts = enriched_df.isnull().sum()
        if null_counts.any():
            logger.warning(
                f"Found null values in enriched data: {null_counts[null_counts > 0].to_dict()}"
            )
            # Fill remaining nulls
            enriched_df = enriched_df.fillna(0.0)

        logger.info(
            f"Enriched training data: {len(enriched_df)} rows, {len(enriched_df.columns)} columns"
        )

        return enriched_df
