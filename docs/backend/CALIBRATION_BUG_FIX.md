# Critical Bug Fix: Calibration Not Applied During Backtest

## Problem

The backtest was still showing negative R² and high RMSE even after implementing calibration. The root cause was a **critical bug**: calibration was fitted but **not applied** to predictions during the backtest.

## Root Cause

In `backend/app/services/backtest.py`, the `_backtest_week()` method was creating a **new `PLEngine()` instance every week**:

```python
# OLD CODE (BUGGY)
def _backtest_week(...):
    # ...
    self.plengine = PLEngine()  # ❌ Creates new instance, loses calibration!
```

This meant:
1. Calibration was fitted early (after ~5 weeks) on one `PLEngine` instance
2. Next week, a **new** `PLEngine` instance was created
3. The new instance had `calibration_fitted = False`, so calibration was **not applied**
4. All predictions were uncalibrated, leading to poor metrics

## Fix

Changed to **reuse the existing `PLEngine` instance** if it exists:

```python
# NEW CODE (FIXED)
def _backtest_week(...):
    # ...
    # CRITICAL FIX: Reuse existing PLEngine instance to preserve calibration
    if self.plengine is None:
        self.plengine = PLEngine()
    # Models are retrained each week, but calibration parameters persist
```

## How It Works Now

1. **Week 1-5**: Make uncalibrated predictions, collect data
2. **Week 6**: Fit calibration on collected predictions (≥5 weeks of data)
3. **Week 6+**: Reuse the same `PLEngine` instance, so calibration is **applied** to all subsequent predictions
4. **End**: Fit final calibration on all data for reporting

## Expected Improvements

With this fix:
- **RMSE**: Should decrease significantly (calibrated predictions)
- **R²**: Should become positive (calibration fixes bias and scale)
- **Variance**: Should match actual distribution (calibration preserves variance)

## Testing

Run the backtest again:
```bash
docker compose exec backend python3 populate_all_tables.py
```

Check the logs for:
- "Fitting Early Calibration" message
- "Early calibration fitted: scale=X.XXX, offset=X.XXX"
- Improved RMSE and R² in final metrics

## Files Modified

- `backend/app/services/backtest.py`: Fixed `_backtest_week()` to reuse PLEngine instance
- `backend/app/services/backtest.py`: Added early calibration fitting in main loop
- `backend/app/services/ml_engine.py`: Enhanced calibration with variance preservation
