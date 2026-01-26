"""
Pure calculation functions for feature engineering.
These functions are pure (no side effects, no database dependencies) and can be unit tested independently.

All functions accept standard Python types (lists, dicts, floats) and return floats or lists.
No SQLAlchemy ORM objects or database sessions are used.
"""

from typing import List, Optional
import numpy as np


def calculate_rolling_average(values: List[float], window: int) -> float:
    """
    Calculate rolling average over a specified window.
    
    Args:
        values: List of values (most recent first)
        window: Number of values to include in the average
        
    Returns:
        Rolling average, or 0.0 if insufficient data
    """
    if not values or len(values) == 0:
        return 0.0
    
    if len(values) < window:
        return 0.0
    
    window_values = values[:window]
    return float(np.mean(window_values))


def calculate_lag_feature(values: List[float], lag: int) -> float:
    """
    Extract a lag feature (value from N periods ago).
    
    Args:
        values: List of values (most recent first, index 0 = most recent)
        lag: Lag period (1 = most recent, 2 = one period ago, etc.)
        
    Returns:
        Lag value, or 0.0 if insufficient data
    """
    if not values or len(values) == 0:
        return 0.0
    
    # lag=1 means most recent (index 0), lag=2 means one period ago (index 1)
    index = lag - 1
    if index < 0 or index >= len(values):
        return 0.0
    
    value = values[index]
    return float(value) if value is not None and not np.isnan(value) else 0.0


def calculate_per_90(total_value: float, total_minutes: float) -> float:
    """
    Calculate per-90 statistic from total value and minutes.
    Handles division by zero by returning 0.0.
    
    Args:
        total_value: Total value (e.g., total goals, total xG)
        total_minutes: Total minutes played
        
    Returns:
        Per-90 value, or 0.0 if minutes is 0 or invalid
    """
    if total_minutes is None or total_minutes <= 0:
        return 0.0
    
    if total_value is None or np.isnan(total_value):
        return 0.0
    
    per_90 = (total_value / total_minutes) * 90.0
    return float(per_90) if not np.isnan(per_90) and not np.isinf(per_90) else 0.0


def calculate_clean_sheet_rate(
    clean_sheets: List[float], minutes: List[float]
) -> float:
    """
    Calculate clean sheet rate (percentage of games with clean sheet).
    
    Args:
        clean_sheets: List of clean sheet values (1.0 = clean sheet, 0.0 = no clean sheet)
        minutes: List of minutes played for each game
        
    Returns:
        Clean sheet rate (0.0 to 1.0), or 0.0 if no valid games
    """
    if not clean_sheets or not minutes:
        return 0.0
    
    if len(clean_sheets) != len(minutes):
        return 0.0
    
    # Only count games where player actually played (minutes > 0)
    games_with_minutes = [
        (cs, m) for cs, m in zip(clean_sheets, minutes) if m > 0
    ]
    
    if len(games_with_minutes) == 0:
        return 0.0
    
    cs_count = sum(1 for cs, _ in games_with_minutes if cs > 0)
    return float(cs_count / len(games_with_minutes))


def calculate_form(
    historical_points: List[float], alpha: float, lookback_weeks: int = 5
) -> float:
    """
    Calculate weighted form using exponential decay.
    
    Args:
        historical_points: List of points from most recent to oldest
        alpha: Decay coefficient (higher = more weight on recent, typically 0.1-1.0)
        lookback_weeks: Number of weeks to consider
        
    Returns:
        Weighted form score, or 0.0 if no data
    """
    if not historical_points or len(historical_points) == 0:
        return 0.0
    
    # Take only recent weeks
    recent_points = historical_points[:lookback_weeks]
    n = len(recent_points)
    
    if n == 0:
        return 0.0
    
    # Exponential decay weights: w0 = 1 (most recent), w1 = alpha, w2 = alpha^2, ...
    # where 0 < alpha < 1 means faster decay (more emphasis on recent)
    weights = [alpha**i for i in range(n)]
    total_weight = sum(weights)
    
    if total_weight == 0:
        return float(np.mean(recent_points))
    
    weighted_sum = sum(p * w for p, w in zip(recent_points, weights))
    return float(weighted_sum / total_weight)


def calculate_trend(historical_points: List[float], weeks: int = 3) -> float:
    """
    Calculate form trend (positive = improving, negative = declining).
    
    Args:
        historical_points: List of points from most recent to oldest
        weeks: Number of weeks to compare (recent vs previous period)
        
    Returns:
        Trend value (recent average - previous average), or 0.0 if insufficient data
    """
    if not historical_points or len(historical_points) < weeks:
        return 0.0
    
    recent = np.mean(historical_points[:weeks])
    
    if len(historical_points) >= weeks * 2:
        previous = np.mean(historical_points[weeks : weeks * 2])
    else:
        previous = recent
    
    return float(recent - previous)


def pad_list(values: List[float], target_length: int, pad_value: float = 0.0) -> List[float]:
    """
    Pad a list to a target length with a specified value.
    
    Args:
        values: List to pad
        target_length: Target length
        pad_value: Value to use for padding (default: 0.0)
        
    Returns:
        Padded list
    """
    if not values:
        return [pad_value] * target_length
    
    if len(values) >= target_length:
        return values[:target_length]
    
    return list(values) + [pad_value] * (target_length - len(values))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero and invalid values.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division is invalid (default: 0.0)
        
    Returns:
        Division result, or default if invalid
    """
    if denominator is None or denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    
    if numerator is None or np.isnan(numerator) or np.isinf(numerator):
        return default
    
    result = numerator / denominator
    if np.isnan(result) or np.isinf(result):
        return default
    
    return float(result)


def calculate_ratio(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """
    Calculate a ratio with safe division by zero handling.
    Alias for safe_divide for semantic clarity.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division is invalid (default: 0.0)
        
    Returns:
        Ratio result, or default if invalid
    """
    return safe_divide(numerator, denominator, default)


def calculate_goals_per_90(goals: float, minutes: float) -> float:
    """
    Calculate goals per 90 minutes.
    
    Args:
        goals: Total goals
        minutes: Total minutes played
        
    Returns:
        Goals per 90, or 0.0 if invalid
    """
    return calculate_per_90(goals, minutes)


def calculate_assists_per_90(assists: float, minutes: float) -> float:
    """
    Calculate assists per 90 minutes.
    
    Args:
        assists: Total assists
        minutes: Total minutes played
        
    Returns:
        Assists per 90, or 0.0 if invalid
    """
    return calculate_per_90(assists, minutes)


def calculate_xg_per_90(xg: float, minutes: float) -> float:
    """
    Calculate expected goals (xG) per 90 minutes.
    
    Args:
        xg: Total expected goals
        minutes: Total minutes played
        
    Returns:
        xG per 90, or 0.0 if invalid
    """
    return calculate_per_90(xg, minutes)


def calculate_xa_per_90(xa: float, minutes: float) -> float:
    """
    Calculate expected assists (xA) per 90 minutes.
    
    Args:
        xa: Total expected assists
        minutes: Total minutes played
        
    Returns:
        xA per 90, or 0.0 if invalid
    """
    return calculate_per_90(xa, minutes)


def calculate_xgc_per_90(xgc: float, minutes: float) -> float:
    """
    Calculate expected goals conceded (xGC) per 90 minutes.
    
    Args:
        xgc: Total expected goals conceded
        minutes: Total minutes played
        
    Returns:
        xGC per 90, or 0.0 if invalid
    """
    return calculate_per_90(xgc, minutes)


def calculate_goals_conceded_per_90(goals_conceded: float, minutes: float) -> float:
    """
    Calculate goals conceded per 90 minutes.
    
    Args:
        goals_conceded: Total goals conceded
        minutes: Total minutes played
        
    Returns:
        Goals conceded per 90, or 0.0 if invalid
    """
    return calculate_per_90(goals_conceded, minutes)
