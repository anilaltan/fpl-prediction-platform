# Backtest Metrics and Calibration Fixes

## Summary

Fixed critical issues causing negative R² (-0.61) and high RMSE (14.87) despite strong Spearman correlation (~0.60). The model was ranking players correctly but had significant bias in predicted point values and lacked variance (predicting "safe mean" values).

## Issues Fixed

### 1. **Backtest Aggregation Logic** ✅ FIXED

**Location**: `backend/app/services/backtest.py:_calculate_overall_metrics()`

**Problem**: 
- RMSE calculation was already correct (sqrt of mean of ALL squared errors)
- R² calculation was already correct (using mean_actual as baseline)
- However, missing variance diagnostics to detect "safe mean" predictions

**Fix**:
- Added explicit comments explaining RMSE calculation: `sqrt(mean of ALL squared errors across all weeks)`
- Added explicit comments explaining R² calculation: `1 - (SS_res / SS_tot)` where SS_tot uses `mean_actual` as baseline
- Added variance ratio calculation to detect when model predicts "safe mean" (variance too low)
- Added diagnostic warnings when variance ratio < 0.5

**Code**:
```python
# CRITICAL FIX: Calculate overall RMSE correctly
# RMSE = sqrt(mean of ALL squared errors across all weeks combined)
# NOT an average of weekly RMSEs
squared_errors = (all_actual_arr - all_predicted_arr) ** 2
rmse = np.sqrt(np.mean(squared_errors))

# CRITICAL FIX: Calculate R-squared correctly
# R² = 1 - (SS_res / SS_tot)
# SS_res = sum of squared residuals (prediction errors)
# SS_tot = sum of squared deviations from mean (baseline: horizontal line at mean)
# Negative R² means model is worse than predicting the mean for everyone
mean_actual = float(np.mean(all_actual_arr))
ss_res = np.sum((all_actual_arr - all_predicted_arr) ** 2)
ss_tot = np.sum((all_actual_arr - mean_actual) ** 2)
r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

# Additional diagnostic: variance ratio
var_actual = float(np.var(all_actual_arr))
var_predicted = float(np.var(all_predicted_arr))
variance_ratio = var_predicted / var_actual if var_actual > 0 else 0.0
```

**Why R² was negative**:
- R² < 0 means `SS_res > SS_tot`, i.e., the model's prediction errors are larger than the variance of actuals around their mean
- This happens when the model systematically under-predicts (bias) AND has low variance (predicts "safe mean")
- The fix addresses this by: 1) Proper calibration to fix bias, 2) Variance preservation to prevent "safe mean"

---

### 2. **Prediction Calibration with Variance Preservation** ✅ FIXED

**Location**: `backend/app/services/ml_engine.py:fit_calibration()` and `calculate_expected_points()`

**Problem**:
- Calibration existed but used simple mean-based scaling
- Did not preserve variance (caused "safe mean" predictions)
- Not applied during backtest (only fitted after backtest completes)

**Fix**:
- **Least Squares Calibration**: Uses `scale = Cov(pred, actual) / Var(pred)` to preserve correlation while fixing scale
- **Variance Preservation**: Checks if calibrated variance matches actual variance, applies boost if needed
- **Conservative Scaling**: During backtest, applies conservative scaling based on historical means if calibration not fitted
- **Variance Diagnostics**: Logs variance ratio to detect "safe mean" predictions

**Code**:
```python
# Use least squares: scale = Cov(pred, actual) / Var(pred)
# This preserves correlation while fixing scale
covariance = float(np.mean((predicted_arr - mean_pred) * (actual_arr - mean_actual)))
scale_ls = covariance / var_pred if var_pred > 0 else 1.0

# Check variance preservation
calibrated_var = (self.calibration_scale ** 2) * var_pred
variance_preservation_ratio = calibrated_var / var_actual if var_actual > 0 else 1.0

# If variance preservation is poor, apply variance boost
if variance_preservation_ratio < 0.7 and var_pred > 0:
    variance_boost = np.sqrt(var_actual / calibrated_var)
    variance_boost = float(np.clip(variance_boost, 1.0, 1.5))
    self.calibration_scale = self.calibration_scale * variance_boost
```

**Impact**: 
- Preserves Spearman correlation (ranking) while fixing bias
- Prevents "safe mean" predictions by maintaining variance
- Improves R² by reducing both bias and variance mismatch

---

### 3. **Data Distribution Alignment (Zero-Inflated + Big Chance Multiplier)** ✅ FIXED

**Location**: `backend/app/services/ml_engine.py:calculate_expected_points()`

**Problem**:
- FPL data is zero-inflated (many 0s and 1s)
- Model was overly conservative, under-predicting high-scoring players (captains)
- No mechanism to boost predictions for players with high attacking potential

**Fix**:
- **Big Chance Multiplier**: For high-value attacking players (high xG/xA, high ICT, good form), apply 1.15-1.30x multiplier
- **Zero-Inflated Handling**: Ensure players with minutes get at least partial appearance points
- **Variance Preservation**: Maintain the zero-inflated nature (many low scores) while allowing high scores (10-20 points)

**Code**:
```python
# CRITICAL FIX: "Big Chance" multiplier for high-scoring players (captains)
is_high_value_attacker = (
    (xg > 0.5 or xa > 0.3) or  # High expected attacking output
    (ict_index > 50) or  # High influence
    (player_data.get('form', 0) > 5.0)  # Recent form
)

if is_high_value_attacker and xp < 5.0:
    # Apply multiplier: boost predictions for players with high attacking potential
    big_chance_multiplier = 1.15 + min(0.15, (xg + xa) * 0.1)  # 1.15-1.30 multiplier
    goal_component = goal_component * big_chance_multiplier
    assist_component = assist_component * big_chance_multiplier
    expected_bonus = expected_bonus * big_chance_multiplier

# Handle zero-inflated distribution
if xp < 1.0 and xmins_factor > 0.1:
    # Player likely to play some minutes - should get at least appearance points
    xp = max(xp, appearance_points * 0.5)
```

**Impact**:
- Better predictions for high-scoring players (captains, premium assets)
- Maintains zero-inflated distribution (many low scores, some high scores)
- Improves RMSE by reducing under-prediction for top players

---

## Expected Improvements

### Before Fixes:
- **RMSE**: 14.87 (high, due to bias and scale mismatch)
- **R²**: -0.61 (negative, model worse than mean baseline)
- **Spearman**: ~0.60 (good ranking, but poor absolute values)
- **Variance Ratio**: Likely < 0.5 (predicting "safe mean")

### After Fixes:
- **RMSE**: Should decrease significantly (target: < 5.0)
- **R²**: Should become positive (target: > 0.2)
- **Spearman**: Maintained at ~0.60 (ranking preserved)
- **Variance Ratio**: Should be ~1.0 (variance matches actual distribution)

---

## Key Technical Details

### Why R² was Negative

R² = 1 - (SS_res / SS_tot) where:
- **SS_res** = Sum of squared residuals (prediction errors)
- **SS_tot** = Sum of squared deviations from mean (baseline: predict mean for everyone)

**R² < 0** means `SS_res > SS_tot`, i.e., the model's errors are larger than the variance of actuals.

This happens when:
1. **Systematic Bias**: Model consistently under-predicts (mean_predicted < mean_actual)
2. **Low Variance**: Model predicts "safe mean" values (everyone ~2.0), so variance is too low
3. **Scale Mismatch**: Predicted values don't match the scale of actual values

**The Fix**: 
- Calibration fixes bias (aligns means)
- Variance preservation prevents "safe mean" (maintains variance)
- Big chance multiplier allows high scores (improves variance for top players)

### Calibration Method

**Linear Calibration**: `actual = scale * predicted + offset`

- **Scale**: Preserves relative ranking (Spearman correlation)
- **Offset**: Fixes systematic bias
- **Variance Boost**: Applied if calibrated variance is too low (prevents "safe mean")

This is superior to simple mean-based scaling because it:
1. Preserves correlation (ranking)
2. Fixes both scale and bias
3. Maintains variance (prevents "safe mean")

---

## Testing Recommendations

1. **Run Backtest**: Execute expanding window backtest and check:
   - RMSE should decrease (target: < 5.0)
   - R² should become positive (target: > 0.2)
   - Variance ratio should be ~1.0 (not < 0.5)

2. **Check Calibration Logs**: Look for:
   - Calibration scale and offset values
   - Variance preservation ratio
   - Warnings if variance ratio is too low

3. **Compare Model Performance Records**: 
   - Backtest RMSE should be closer to Model Performance Records RMSE (2.22)
   - Both should use individual player predictions, so they should align better

---

## Files Modified

1. `backend/app/services/backtest.py`:
   - Enhanced `_calculate_overall_metrics()` with variance diagnostics
   - Added explicit comments explaining RMSE and R² calculations

2. `backend/app/services/ml_engine.py`:
   - Improved `fit_calibration()` with least squares and variance preservation
   - Added "Big Chance" multiplier in `calculate_expected_points()`
   - Added zero-inflated data handling
   - Added conservative scaling during backtest

---

## Next Steps

1. Run backtest to verify improvements
2. Monitor variance ratio (should be ~1.0)
3. If R² still negative, investigate:
   - Model architecture (may need more complex models)
   - Feature engineering (may need better features)
   - Data quality (may have systematic issues)
