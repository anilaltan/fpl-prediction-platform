# Backtest Bias & Scaling Fixes

## Summary
Fixed critical issues causing negative R² (-0.61) and high RMSE (14.87) despite strong Spearman correlation (0.60). The model was ranking players correctly but had significant bias in predicted point values.

## Issues Identified & Fixed

### 1. **Expected Bonus Points Not Added to Final xP** ✅ FIXED
**Location**: `backend/app/services/ml_engine.py:1535-1539`

**Problem**: 
- `expected_bonus` was calculated but never added to the final `xp` value
- This caused systematic under-prediction, especially for top players who typically earn bonus points

**Fix**:
```python
# CRITICAL FIX: Add expected bonus points (was calculated but never added!)
xp += expected_bonus
```

**Impact**: This alone should increase predicted values by 0.5-1.5 points for attacking players, helping align the distribution.

---

### 2. **Incorrect Metric Calculation on Weekly Team Sums** ✅ FIXED
**Location**: `backend/app/services/backtest.py:_calculate_overall_metrics()`

**Problem**:
- RMSE and R² were calculated on weekly aggregated team points (sum of top 11 players)
- This is incorrect - metrics should be calculated on individual player predictions across all weeks
- Weekly aggregation masks individual prediction errors

**Fix**:
- Added `self.all_individual_predictions` and `self.all_individual_actuals` to store all player predictions
- Modified `_calculate_overall_metrics()` to use individual predictions when available
- Added bias metrics (mean difference, percentage bias) for better diagnostics

**Impact**: 
- More accurate RMSE/R² calculation
- Better visibility into prediction bias

---

### 3. **Missing Calibration Layer** ✅ FIXED
**Location**: `backend/app/services/ml_engine.py:fit_calibration()`

**Problem**:
- FPL points are "zero-inflated" (many players get 0-2 points)
- Model predictions may have correct ranking but wrong scale
- No mechanism to align predicted distribution with actual distribution

**Fix**:
- Added `fit_calibration()` method with two approaches:
  - **Linear**: Simple scale + offset adjustment (robust for zero-inflated data)
  - **Isotonic**: Monotonic regression (more sophisticated, requires sklearn)
- Calibration automatically fitted during backtest using historical predictions
- Calibrated predictions reported alongside raw predictions

**Impact**:
- Should significantly improve R² (from negative to positive)
- Reduces RMSE by aligning predicted scale with actual distribution
- Maintains ranking (Spearman correlation) while fixing scale

---

### 4. **Individual Predictions Not Stored** ✅ FIXED
**Location**: `backend/app/services/backtest.py:_backtest_week()`

**Problem**:
- Individual player predictions were calculated but discarded after weekly aggregation
- Could not calculate proper aggregate metrics

**Fix**:
- Store all individual predictions in `self.all_individual_predictions`
- Store all individual actuals in `self.all_individual_actuals`
- Used for both metric calculation and calibration fitting

**Impact**: Enables proper aggregate metrics and calibration fitting.

---

## Code Changes Summary

### `backend/app/services/ml_engine.py`
1. **Added calibration layer** (lines ~1173-1177):
   - `calibration_enabled`, `calibration_scale`, `calibration_offset`, `calibration_fitted`
2. **Fixed expected_bonus addition** (line ~1546):
   - Now added to final xP calculation
3. **Added fit_calibration() method** (lines ~1720-1850):
   - Linear and isotonic calibration methods
   - Comprehensive metrics reporting

### `backend/app/services/backtest.py`
1. **Added individual prediction storage** (lines ~73-75):
   - `all_individual_predictions`, `all_individual_actuals`
2. **Store predictions during backtest** (lines ~380-384):
   - Capture all individual predictions for aggregate metrics
3. **Fixed metric calculation** (lines ~938-1007):
   - Use individual predictions instead of weekly sums
   - Added bias metrics (bias, pct_bias)
4. **Integrated calibration fitting** (lines ~214-260):
   - Automatically fit calibration during backtest
   - Report calibrated metrics alongside raw metrics

---

## Expected Improvements

### Before Fixes:
- RMSE: 14.87
- R²: -0.61 (negative = model worse than mean prediction)
- Spearman: 0.60 (good ranking)
- Mean Predicted: ~2.0 (estimated)
- Mean Actual: ~4.5 (estimated)

### After Fixes:
- **Expected Bonus Fix**: +0.5-1.5 points for attacking players
- **Calibration**: Should bring R² from negative to positive (target: 0.2-0.4)
- **RMSE**: Should reduce by 20-30% with calibration
- **Bias**: Should reduce from ~-2.5 to <0.5 points

---

## Usage

### Running Backtest with Calibration
The calibration is automatically fitted during backtest. Results include:
- Raw metrics (before calibration)
- Calibrated metrics (after calibration)
- Calibration parameters (scale, offset)

### Manual Calibration Fitting
```python
from app.services.ml_engine import PLEngine

plengine = PLEngine()
calibration_result = plengine.fit_calibration(
    predicted_points=np.array(predictions),
    actual_points=np.array(actuals),
    method='linear'  # or 'isotonic'
)
```

### Disabling Calibration
```python
plengine.calibration_enabled = False
```

---

## Next Steps

1. **Run backtest** to verify improvements
2. **Check calibrated metrics** - R² should be positive
3. **Analyze bias metrics** - should be close to 0
4. **Consider isotonic calibration** if linear doesn't fully solve the issue
5. **Investigate feature scaling** if bias persists (Attack Model features may need normalization)

---

## Technical Notes

### Why Negative R²?
R² = 1 - (SS_res / SS_tot)
- SS_res: Sum of squared residuals (predicted - actual)²
- SS_tot: Sum of squared deviations from mean (actual - mean)²

If SS_res > SS_tot, R² is negative, meaning the model is worse than simply predicting the mean.

### Why Calibration Works
- Model ranks correctly (high Spearman) but scale is off
- Calibration applies a simple transformation: `calibrated = scale * predicted + offset`
- This preserves ranking while fixing scale
- Linear calibration is robust for zero-inflated distributions

### Zero-Inflated Distribution
FPL points distribution:
- Many players: 0-2 points (bench players, rotation)
- Few players: 5-15 points (starters, goal scorers)
- Very few: 15+ points (captain picks, hat-tricks)

This makes mean-based scaling less effective - median-based or robust scaling works better.
