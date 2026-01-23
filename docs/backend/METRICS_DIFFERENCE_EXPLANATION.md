# Why Backtest Summary and Model Performance Records Are Different

## Key Differences

### 1. **Data Source & Calculation Method**

#### Model Performance Records (RMSE: 2.228, MAE: 1.037)
- **Source**: Pre-calculated predictions stored in `Prediction` table
- **Method**: Direct comparison of stored predictions vs actual points
- **Code**: `backend/populate_all_tables.py:track_model_performance()`
- **Calculation**:
  ```python
  # Simple comparison: Prediction.xp vs PlayerGameweekStats.total_points
  errors = [abs(pred_map[fpl_id] - actual_map[fpl_id]) for fpl_id in matching_players]
  mae = np.mean(errors)
  rmse = np.sqrt(np.mean([e**2 for e in errors]))
  ```
- **Scope**: Per gameweek, individual player predictions
- **When calculated**: After predictions are generated and stored

#### Backtest Summary (RMSE: 14.87, R²: -0.61)
- **Source**: Predictions generated during expanding window backtest
- **Method**: Train model on historical data, predict current week, compare
- **Code**: `backend/app/services/backtest.py:BacktestEngine`
- **Calculation**:
  ```python
  # Expanding window: Train on weeks 1-N, predict week N+1
  # Compare all individual predictions across all weeks
  all_predicted_arr = np.array(self.all_individual_predictions)
  all_actual_arr = np.array(self.all_individual_actuals)
  rmse = np.sqrt(mean_squared_error(all_actual_arr, all_predicted_arr))
  ```
- **Scope**: Aggregated across all gameweeks in backtest
- **When calculated**: During backtest execution

---

### 2. **What They Measure**

#### Model Performance Records
- **Measures**: Accuracy of **stored predictions** (what users see in the app)
- **Use case**: Track how well predictions perform after they're made
- **Granularity**: Per gameweek
- **Players included**: Only players with both prediction AND actual stats

#### Backtest Summary
- **Measures**: Model's **predictive capability** using historical data
- **Use case**: Validate model performance before deployment
- **Granularity**: Aggregated across all tested gameweeks
- **Players included**: All players in the backtest dataset

---

### 3. **Why RMSE is So Different**

#### Model Performance: RMSE = 2.228 ✅
- **Why lower**: 
  - Uses **calibrated/stored predictions** (may have post-processing)
  - Only includes players with complete data (no missing values)
  - Predictions were made with full context (all features available)
  - May use cached/optimized predictions

#### Backtest: RMSE = 14.87 ⚠️
- **Why higher**:
  - **CRITICAL**: The old backtest code calculated RMSE on **weekly team sums** (sum of top 11 players)
  - This aggregates errors, making RMSE much larger
  - Example: If each player is off by 2 points, team sum is off by 22 points → RMSE = 22
  - **After our fix**: Now calculates on individual players, should be much lower
  - Uses **expanding window** methodology (train on past, predict future)
  - Predictions made with limited historical context (only past weeks)
  - No calibration applied during backtest (raw model predictions)

---

### 4. **Calculation Level**

#### Model Performance Records
```
Individual Player Level:
Player A: Predicted 5.2, Actual 6.0 → Error = 0.8
Player B: Predicted 3.1, Actual 2.5 → Error = 0.6
Player C: Predicted 4.0, Actual 4.0 → Error = 0.0
...
RMSE = sqrt(mean([0.8², 0.6², 0.0², ...])) = 2.228
```

#### Backtest Summary (OLD - Before Fix)
```
Weekly Team Sum Level (WRONG):
GW1: Predicted Team Sum = 55, Actual Team Sum = 70 → Error = 15
GW2: Predicted Team Sum = 48, Actual Team Sum = 65 → Error = 17
...
RMSE = sqrt(mean([15², 17², ...])) = 14.87
```

#### Backtest Summary (NEW - After Fix)
```
Individual Player Level (CORRECT):
Same as Model Performance, but across all weeks
Should now be similar to Model Performance (~2-3 RMSE)
```

---

### 5. **Accuracy Metric**

#### Model Performance: Accuracy = 76.0%
- **Definition**: Percentage of predictions within 1 point of actual
- **Formula**: `sum(1 for e in errors if e <= 1.0) / len(errors)`
- **Interpretation**: 76% of predictions are within ±1 point

#### Backtest: No Accuracy Metric
- Backtest focuses on RMSE, MAE, and R²
- Accuracy is not calculated in backtest (different focus)

---

## Summary Table

| Aspect | Model Performance Records | Backtest Summary |
|--------|---------------------------|------------------|
| **RMSE** | 2.228 | 14.87 (old) / ~2-3 (new) |
| **Data Source** | Stored predictions | Generated during backtest |
| **Calculation Level** | Individual players | Individual players (new) / Team sums (old) |
| **Scope** | Per gameweek | Aggregated across weeks |
| **Calibration** | May include calibration | Raw predictions |
| **Context** | Full feature set | Limited historical context |
| **Use Case** | Track production performance | Validate model before deployment |

---

## Why This Matters

1. **Model Performance Records** tell you: "How accurate are the predictions users see?"
2. **Backtest Summary** tells you: "How well does the model generalize to unseen data?"

Both are important, but they measure different things:
- **Model Performance**: Production accuracy (what users experience)
- **Backtest**: Model validation (how well it should work)

---

## Recommendations

1. **After our backtest fix**, re-run the backtest to get accurate individual-player metrics
2. **Compare both metrics** to ensure they're in the same ballpark (~2-3 RMSE)
3. **If backtest RMSE is still high**, investigate:
   - Are predictions being calibrated before storage?
   - Are there data quality issues in backtest?
   - Is the expanding window methodology appropriate?

4. **Use Model Performance Records** for:
   - Monitoring production accuracy
   - Tracking improvements over time
   - Identifying problematic gameweeks

5. **Use Backtest Summary** for:
   - Model validation before deployment
   - Understanding model generalization
   - Comparing different model versions
