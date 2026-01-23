# Test Data and Scripts Cleanup

## Files to Delete

### Test Files (10 files)
- `backend/test_bulk_resolution.py`
- `backend/test_calibration.py`
- `backend/test_canonical_name_generation.py`
- `backend/test_entity_mapping_persistence.py`
- `backend/test_fuzzy_matching.py`
- `backend/test_manual_override.py`
- `backend/test_market_intelligence_ranking.py`
- `backend/test_multi_period_solver.py`
- `backend/test_multi_source_fuzzy_matching.py`
- `backend/test_third_party_scrapers.py`
- `backend/smoke_test.py`

### Debug Scripts (2 files)
- `backend/app/scripts/Debug_logic.py`
- `backend/app/scripts/debug_predictions.py`

### Training/Validation Scripts (5 files)
- `backend/train_attack_model.py`
- `backend/train_defense_model.py`
- `backend/train_xmins_model.py`
- `backend/train_models.py`
- `backend/validate_component_model.py`
- `backend/validate_market_intelligence.py`

### Data Population Scripts (4 files)
- `backend/populate_all_tables.py`
- `backend/load_data.py`
- `backend/refresh_players.py`
- `backend/refresh_teams.py`

### Baseline/Testing Scripts (2 files)
- `backend/phase0_baseline.py`
- `backend/run_backtest.py`

### Test Reports (18 files)
- All files in `backend/reports/` directory

## Files to Keep

### Production Scripts
- `backend/app/scripts/update_predictions.py` - Used in production API

### Database Scripts
- `backend/scripts/` directory - Database migrations and setup

### Documentation
- Keep all `.md` files for reference
