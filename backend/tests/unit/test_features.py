"""
Unit tests for pure calculation functions in app.services.ml.calculations.

These tests verify that calculation logic works correctly without requiring
database connections or external services.
"""

import pytest
import numpy as np
from app.services.ml.calculations import (
    calculate_rolling_average,
    calculate_lag_feature,
    calculate_per_90,
    calculate_clean_sheet_rate,
    calculate_form,
    calculate_trend,
    pad_list,
    safe_divide,
    calculate_ratio,
    calculate_goals_per_90,
    calculate_assists_per_90,
    calculate_xg_per_90,
    calculate_xa_per_90,
    calculate_xgc_per_90,
    calculate_goals_conceded_per_90,
)


class TestRollingAverage:
    """Tests for calculate_rolling_average function."""

    def test_normal_case(self):
        """Test rolling average with sufficient data."""
        values = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = calculate_rolling_average(values, 3)
        assert result == pytest.approx(8.0)  # (10 + 8 + 6) / 3

    def test_window_5(self):
        """Test rolling average with window of 5."""
        values = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = calculate_rolling_average(values, 5)
        assert result == pytest.approx(6.0)  # (10 + 8 + 6 + 4 + 2) / 5

    def test_insufficient_data(self):
        """Test rolling average with insufficient data."""
        values = [10.0, 8.0]
        result = calculate_rolling_average(values, 3)
        assert result == 0.0

    def test_empty_list(self):
        """Test rolling average with empty list."""
        values = []
        result = calculate_rolling_average(values, 3)
        assert result == 0.0

    def test_none_values(self):
        """Test rolling average with None input."""
        result = calculate_rolling_average(None, 3)
        assert result == 0.0

    def test_single_value(self):
        """Test rolling average with single value."""
        values = [5.0]
        result = calculate_rolling_average(values, 1)
        assert result == pytest.approx(5.0)


class TestLagFeature:
    """Tests for calculate_lag_feature function."""

    def test_lag_1_most_recent(self):
        """Test lag 1 returns most recent value."""
        values = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = calculate_lag_feature(values, 1)
        assert result == 10.0

    def test_lag_3(self):
        """Test lag 3 returns value from 3 periods ago."""
        values = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = calculate_lag_feature(values, 3)
        assert result == 6.0  # index 2

    def test_lag_5(self):
        """Test lag 5 returns value from 5 periods ago."""
        values = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = calculate_lag_feature(values, 5)
        assert result == 2.0  # index 4

    def test_insufficient_data(self):
        """Test lag feature with insufficient data."""
        values = [10.0, 8.0]
        result = calculate_lag_feature(values, 5)
        assert result == 0.0

    def test_empty_list(self):
        """Test lag feature with empty list."""
        values = []
        result = calculate_lag_feature(values, 1)
        assert result == 0.0

    def test_nan_value(self):
        """Test lag feature handles NaN values."""
        values = [10.0, np.nan, 6.0]
        result = calculate_lag_feature(values, 2)
        assert result == 0.0

    def test_none_value(self):
        """Test lag feature handles None values."""
        values = [10.0, None, 6.0]
        result = calculate_lag_feature(values, 2)
        assert result == 0.0


class TestPer90:
    """Tests for per-90 calculation functions."""

    def test_normal_per_90(self):
        """Test normal per-90 calculation."""
        result = calculate_per_90(10.0, 180.0)  # 10 goals in 180 minutes
        assert result == pytest.approx(5.0)  # 10 / 180 * 90 = 5.0

    def test_zero_minutes(self):
        """Test per-90 with zero minutes (division by zero)."""
        result = calculate_per_90(10.0, 0.0)
        assert result == 0.0

    def test_negative_minutes(self):
        """Test per-90 with negative minutes."""
        result = calculate_per_90(10.0, -10.0)
        assert result == 0.0

    def test_none_minutes(self):
        """Test per-90 with None minutes."""
        result = calculate_per_90(10.0, None)
        assert result == 0.0

    def test_nan_values(self):
        """Test per-90 with NaN values."""
        result = calculate_per_90(np.nan, 180.0)
        assert result == 0.0
        result = calculate_per_90(10.0, np.nan)
        assert result == 0.0

    def test_inf_values(self):
        """Test per-90 with infinity values."""
        result = calculate_per_90(10.0, np.inf)
        assert result == 0.0
        result = calculate_per_90(np.inf, 180.0)
        assert result == 0.0

    def test_goals_per_90(self):
        """Test goals_per_90 wrapper."""
        result = calculate_goals_per_90(20.0, 360.0)
        assert result == pytest.approx(5.0)

    def test_assists_per_90(self):
        """Test assists_per_90 wrapper."""
        result = calculate_assists_per_90(10.0, 180.0)
        assert result == pytest.approx(5.0)

    def test_xg_per_90(self):
        """Test xg_per_90 wrapper."""
        result = calculate_xg_per_90(5.0, 90.0)
        assert result == pytest.approx(5.0)

    def test_xa_per_90(self):
        """Test xa_per_90 wrapper."""
        result = calculate_xa_per_90(3.0, 90.0)
        assert result == pytest.approx(3.0)

    def test_xgc_per_90(self):
        """Test xgc_per_90 wrapper."""
        result = calculate_xgc_per_90(2.0, 180.0)
        assert result == pytest.approx(1.0)

    def test_goals_conceded_per_90(self):
        """Test goals_conceded_per_90 wrapper."""
        result = calculate_goals_conceded_per_90(4.0, 180.0)
        assert result == pytest.approx(2.0)


class TestCleanSheetRate:
    """Tests for calculate_clean_sheet_rate function."""

    def test_normal_case(self):
        """Test clean sheet rate with normal data."""
        clean_sheets = [1.0, 0.0, 1.0, 0.0, 1.0]
        minutes = [90.0, 90.0, 90.0, 90.0, 90.0]
        result = calculate_clean_sheet_rate(clean_sheets, minutes)
        assert result == pytest.approx(0.6)  # 3 clean sheets out of 5 games

    def test_no_clean_sheets(self):
        """Test clean sheet rate with no clean sheets."""
        clean_sheets = [0.0, 0.0, 0.0]
        minutes = [90.0, 90.0, 90.0]
        result = calculate_clean_sheet_rate(clean_sheets, minutes)
        assert result == 0.0

    def test_all_clean_sheets(self):
        """Test clean sheet rate with all clean sheets."""
        clean_sheets = [1.0, 1.0, 1.0]
        minutes = [90.0, 90.0, 90.0]
        result = calculate_clean_sheet_rate(clean_sheets, minutes)
        assert result == pytest.approx(1.0)

    def test_exclude_zero_minutes(self):
        """Test that games with 0 minutes are excluded."""
        clean_sheets = [1.0, 0.0, 1.0, 0.0]
        minutes = [90.0, 0.0, 90.0, 90.0]  # Second game has 0 minutes
        result = calculate_clean_sheet_rate(clean_sheets, minutes)
        assert result == pytest.approx(2.0 / 3.0)  # 2 clean sheets out of 3 games with minutes

    def test_empty_lists(self):
        """Test clean sheet rate with empty lists."""
        result = calculate_clean_sheet_rate([], [])
        assert result == 0.0

    def test_mismatched_lengths(self):
        """Test clean sheet rate with mismatched list lengths."""
        clean_sheets = [1.0, 0.0]
        minutes = [90.0, 90.0, 90.0]
        result = calculate_clean_sheet_rate(clean_sheets, minutes)
        assert result == 0.0

    def test_all_zero_minutes(self):
        """Test clean sheet rate when all games have 0 minutes."""
        clean_sheets = [1.0, 1.0]
        minutes = [0.0, 0.0]
        result = calculate_clean_sheet_rate(clean_sheets, minutes)
        assert result == 0.0


class TestForm:
    """Tests for calculate_form function."""

    def test_normal_case(self):
        """Test form calculation with normal data."""
        historical_points = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = calculate_form(historical_points, alpha=0.5, lookback_weeks=5)
        # With alpha=0.5, weights are [1, 0.5, 0.25, 0.125, 0.0625]
        # Weighted sum: 10*1 + 8*0.5 + 6*0.25 + 4*0.125 + 2*0.0625 = 10 + 4 + 1.5 + 0.5 + 0.125 = 16.125
        # Total weight: 1 + 0.5 + 0.25 + 0.125 + 0.0625 = 1.9375
        # Result: 16.125 / 1.9375 ≈ 8.322
        assert result > 0.0
        assert result < 10.0  # Should be weighted toward recent (higher) values

    def test_alpha_1_0(self):
        """Test form with alpha=1.0 (equal weights)."""
        historical_points = [10.0, 8.0, 6.0]
        result = calculate_form(historical_points, alpha=1.0, lookback_weeks=3)
        assert result == pytest.approx(8.0)  # Simple average: (10 + 8 + 6) / 3

    def test_insufficient_data(self):
        """Test form with insufficient data."""
        historical_points = [10.0, 8.0]
        result = calculate_form(historical_points, alpha=0.5, lookback_weeks=5)
        # Should use available data
        assert result > 0.0

    def test_empty_list(self):
        """Test form with empty list."""
        historical_points = []
        result = calculate_form(historical_points, alpha=0.5, lookback_weeks=5)
        assert result == 0.0

    def test_none_input(self):
        """Test form with None input."""
        result = calculate_form(None, alpha=0.5, lookback_weeks=5)
        assert result == 0.0

    def test_single_value(self):
        """Test form with single value."""
        historical_points = [5.0]
        result = calculate_form(historical_points, alpha=0.5, lookback_weeks=1)
        assert result == pytest.approx(5.0)


class TestTrend:
    """Tests for calculate_trend function."""

    def test_improving_trend(self):
        """Test trend calculation with improving form."""
        historical_points = [10.0, 9.0, 8.0, 5.0, 4.0, 3.0]  # Recent is better
        result = calculate_trend(historical_points, weeks=3)
        # Recent (first 3): (10 + 9 + 8) / 3 = 9.0
        # Previous (next 3): (5 + 4 + 3) / 3 = 4.0
        # Trend: 9.0 - 4.0 = 5.0
        assert result == pytest.approx(5.0)

    def test_declining_trend(self):
        """Test trend calculation with declining form."""
        historical_points = [3.0, 4.0, 5.0, 8.0, 9.0, 10.0]  # Recent is worse
        result = calculate_trend(historical_points, weeks=3)
        # Recent (first 3): (3 + 4 + 5) / 3 = 4.0
        # Previous (next 3): (8 + 9 + 10) / 3 = 9.0
        # Trend: 4.0 - 9.0 = -5.0
        assert result == pytest.approx(-5.0)

    def test_insufficient_data(self):
        """Test trend with insufficient data."""
        historical_points = [10.0, 8.0]
        result = calculate_trend(historical_points, weeks=3)
        assert result == 0.0

    def test_empty_list(self):
        """Test trend with empty list."""
        historical_points = []
        result = calculate_trend(historical_points, weeks=3)
        assert result == 0.0

    def test_none_input(self):
        """Test trend with None input."""
        result = calculate_trend(None, weeks=3)
        assert result == 0.0


class TestPadList:
    """Tests for pad_list function."""

    def test_pad_to_target(self):
        """Test padding list to target length."""
        values = [1.0, 2.0]
        result = pad_list(values, 5)
        assert len(result) == 5
        assert result == [1.0, 2.0, 0.0, 0.0, 0.0]

    def test_no_padding_needed(self):
        """Test when list is already long enough."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = pad_list(values, 3)
        assert len(result) == 3
        assert result == [1.0, 2.0, 3.0]

    def test_custom_pad_value(self):
        """Test padding with custom pad value."""
        values = [1.0, 2.0]
        result = pad_list(values, 5, pad_value=-1.0)
        assert result == [1.0, 2.0, -1.0, -1.0, -1.0]

    def test_empty_list(self):
        """Test padding empty list."""
        values = []
        result = pad_list(values, 3)
        assert result == [0.0, 0.0, 0.0]

    def test_none_input(self):
        """Test padding None input."""
        result = pad_list(None, 3)
        assert result == [0.0, 0.0, 0.0]


class TestSafeDivide:
    """Tests for safe_divide and calculate_ratio functions."""

    def test_normal_division(self):
        """Test normal division."""
        result = safe_divide(10.0, 2.0)
        assert result == pytest.approx(5.0)

    def test_division_by_zero(self):
        """Test division by zero returns default."""
        result = safe_divide(10.0, 0.0)
        assert result == 0.0

    def test_custom_default(self):
        """Test division by zero with custom default."""
        result = safe_divide(10.0, 0.0, default=-1.0)
        assert result == -1.0

    def test_nan_numerator(self):
        """Test division with NaN numerator."""
        result = safe_divide(np.nan, 2.0)
        assert result == 0.0

    def test_nan_denominator(self):
        """Test division with NaN denominator."""
        result = safe_divide(10.0, np.nan)
        assert result == 0.0

    def test_inf_values(self):
        """Test division with infinity values."""
        result = safe_divide(10.0, np.inf)
        assert result == 0.0
        result = safe_divide(np.inf, 10.0)
        assert result == 0.0

    def test_none_values(self):
        """Test division with None values."""
        result = safe_divide(10.0, None)
        assert result == 0.0
        result = safe_divide(None, 2.0)
        assert result == 0.0

    def test_calculate_ratio_alias(self):
        """Test that calculate_ratio is an alias for safe_divide."""
        result1 = safe_divide(10.0, 2.0)
        result2 = calculate_ratio(10.0, 2.0)
        assert result1 == result2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_numbers(self):
        """Test calculations with very large numbers."""
        result = calculate_per_90(1e10, 90.0)
        assert result > 0.0
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_very_small_numbers(self):
        """Test calculations with very small numbers."""
        result = calculate_per_90(1e-10, 90.0)
        assert result >= 0.0
        assert not np.isnan(result)

    def test_mixed_types(self):
        """Test that functions handle mixed numeric types."""
        result = calculate_per_90(10, 180)  # Integers
        assert result == pytest.approx(5.0)

    def test_zero_values(self):
        """Test calculations with zero values."""
        result = calculate_per_90(0.0, 90.0)
        assert result == 0.0
        
        result = calculate_rolling_average([0.0, 0.0, 0.0], 3)
        assert result == 0.0
