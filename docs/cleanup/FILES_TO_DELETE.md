# Files Identified for Deletion - Code Cleanup Task 1

## Analysis Summary

Scanned `backend/app` directory and cross-referenced with:
- Imports in `main.py`
- Core services in `ARCHITECTURE_MAP.md`
- Usage across entire `backend/` directory

---

## ✅ Files Safe to Delete (Not Used)

### 1. `backend/app/services/single_gw_solver.py`
**Status**: ❌ **UNUSED**  
**Reason**: 
- No imports found anywhere in the codebase
- Functionality appears to be covered by `team_solver.py` (used in main.py)
- `solver.py` (FPLSolver) is used by backtest.py, but single_gw_solver is not

**Size**: ~321 lines  
**Risk**: Low (completely unused)

---

## ⚠️ Files to Remove After Migration (Currently Used)

### 2. `backend/app/services/predictive_engine.py`
**Status**: ⚠️ **DUPLICATE - Needs Migration**  
**Reason**: 
- Duplicate of `ml_engine.py` (PLEngine)
- Currently used in `main.py` endpoints:
  - `/api/predictive/xmins` (line 612)
  - `/api/predictive/attack` (line 635)
  - `/api/predictive/defense` (line 656)
  - `/api/predictive/momentum` (line 682)
  - `/api/predictive/comprehensive` (line 706)
- According to ARCHITECTURE_MAP.md, `PLEngine` from `ml_engine.py` is the active engine
- Both `PredictiveEngine` and `PLEngine` are initialized in main.py (lines 78, 82)

**Size**: ~695 lines  
**Risk**: Medium (requires endpoint migration to use `ml_engine.PLEngine` instead)

**Action Required**: 
1. Migrate endpoints in `main.py` to use `ml_engine` instead of `predictive_engine`
2. Remove import and initialization of `PredictiveEngine`
3. Delete file

---

## ✅ Files That Are Used (Keep These)

### Services Used by main.py:
- ✅ `fpl_api.py` - FPL API integration
- ✅ `etl_service.py` - ETL operations
- ✅ `ml_engine.py` - ML engine (PLEngine) - **KEEP THIS**
- ✅ `feature_engineering.py` - Feature engineering
- ✅ `team_solver.py` - Team optimization solver
- ✅ `market_intelligence.py` - Market intelligence
- ✅ `risk_management.py` - Risk management
- ✅ `backtesting.py` - Backtest engine (API)
- ✅ `entity_resolution.py` - Entity resolution
- ✅ `data_cleaning.py` - Data cleaning
- ✅ `third_party_data.py` - Third-party data

### Services Used by Other Files:
- ✅ `backtest.py` - Used by training scripts, populate_all_tables.py
- ✅ `solver.py` - Used by backtest.py, test_multi_period_solver.py, smoke_test.py
- ✅ `strategy.py` - Used by backtest.py, smoke_test.py
- ✅ `component_feature_engineering.py` - Used by train_*.py scripts, validate_component_model.py
- ✅ `fuzzy_matching.py` - Used by entity_resolution.py (internal import)

### Scripts Used:
- ✅ `scripts/update_predictions.py` - Used for batch prediction updates
- ✅ `scripts/Debug_logic.py` - Debug script (imports ml_engine)
- ✅ `scripts/debug_predictions.py` - Debug script (imports ml_engine)

---

## Summary

### Immediate Deletion (No Migration Needed):
1. ✅ `backend/app/services/single_gw_solver.py` - **SAFE TO DELETE NOW**

### Deletion After Migration:
2. ⚠️ `backend/app/services/predictive_engine.py` - **DELETE AFTER MIGRATING ENDPOINTS**

### Total Lines to Remove:
- Immediate: ~321 lines
- After migration: ~695 lines
- **Total**: ~1,016 lines

---

## Recommended Action Plan

### Step 1: Delete Unused File (Low Risk)
```bash
# Safe to delete immediately
rm backend/app/services/single_gw_solver.py
```

### Step 2: Migrate Endpoints (Medium Risk)
1. Update `main.py` endpoints to use `ml_engine` instead of `predictive_engine`
2. Verify all endpoints work correctly
3. Remove `predictive_engine` import and initialization
4. Delete `predictive_engine.py`

---

## Verification Checklist

Before deleting `predictive_engine.py`:
- [ ] All `/api/predictive/*` endpoints migrated to use `ml_engine.PLEngine`
- [ ] Test all predictive endpoints:
  - [ ] `/api/predictive/xmins`
  - [ ] `/api/predictive/attack`
  - [ ] `/api/predictive/defense`
  - [ ] `/api/predictive/momentum`
  - [ ] `/api/predictive/comprehensive`
- [ ] Remove `from app.services.predictive_engine import PredictiveEngine` from main.py
- [ ] Remove `predictive_engine = PredictiveEngine()` initialization
- [ ] Verify no other files import `predictive_engine`

---

## Notes

- `fuzzy_matching.py` is used internally by `entity_resolution.py`, so it should be kept
- `backtest.py` and `backtesting.py` serve different purposes (scripts vs API), both are needed
- `solver.py` and `team_solver.py` serve different purposes (backtest vs API), both are needed
- All debug scripts are kept as they may be useful for troubleshooting
