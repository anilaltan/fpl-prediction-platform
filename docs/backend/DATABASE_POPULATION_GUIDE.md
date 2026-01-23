# Database Table Population Guide

This document explains why certain database tables are empty and how to populate them.

## Current Status

Based on the database check:
- ✅ **PlayerGameweekStats**: 16,559 records (populated by ETL)
- ❌ **BacktestResult**: 0 records
- ❌ **BacktestSummary**: 0 records  
- ❌ **EntityMapping**: 0 records
- ❌ **Fixture**: 0 records
- ❌ **FormAlpha**: 0 records
- ❌ **ModelPerformance**: 0 records
- ❌ **TeamFDR**: 0 records
- ❌ **TeamStats**: 0 records

## Why Tables Are Empty

### 1. **Fixtures Table**
**Why empty**: The ETL service (`load_data.py`) only loads players and player_gameweek_stats. Fixtures are not automatically loaded.

**How to populate**:
```bash
# Option 1: Use the API endpoint (fetches from FPL API, doesn't save to DB)
curl http://localhost:8000/api/fpl/fixtures

# Option 2: Create a script to load fixtures (see below)
```

### 2. **BacktestResult & BacktestSummary**
**Why empty**: These are populated when backtests are run. No backtests have been executed yet.

**How to populate**:
```bash
# Run backtest via API endpoint
curl -X POST http://localhost:8000/api/backtesting/run \
  -H "Content-Type: application/json" \
  -d '{
    "historical_data": [...],
    "methodology": "expanding_window",
    "min_train_weeks": 5
  }'

# Or use the BacktestEngine.save_report_to_database() method
```

### 3. **EntityMapping**
**Why empty**: Used for entity resolution across FPL, Understat, and FBref. Not populated by default ETL.

**How to populate**:
- Use the entity resolution endpoints:
  - `/api/entity-resolution/resolve`
  - `/api/entity-resolution/resolve-bulk`
  - `/api/entity-resolution/resolve-all`

### 4. **FormAlpha**
**Why empty**: Populated when form alpha optimization is run.

**How to populate**:
```bash
curl -X POST http://localhost:8000/api/features/optimize-form-alpha \
  -H "Content-Type: application/json" \
  -d '{
    "historical_data": [...],
    "lookback_weeks": 5
  }'
```

### 5. **ModelPerformance**
**Why empty**: Populated when model performance tracking is enabled (separate from backtests).

**How to populate**:
- This table is typically populated by monitoring scripts or when model evaluation is run.
- Currently, there's no automatic population - would need to add tracking code.

### 6. **TeamFDR**
**Why empty**: Populated when FDR (Fixture Difficulty Rating) model is fitted.

**How to populate**:
```bash
curl -X POST http://localhost:8000/api/features/fit-fdr \
  -H "Content-Type: application/json" \
  -d '{
    "historical_data": [...]
  }'
```

### 7. **TeamStats**
**Why empty**: Not populated by the ETL service. This requires separate loading of team-level statistics.

**How to populate**:
- Would need to create a script to aggregate team stats from player_gameweek_stats or fetch from FPL API.

## Quick Population Scripts

### Load Fixtures
```python
# backend/load_fixtures.py (create this)
import asyncio
from app.services.fpl_api import FPLAPIService
from app.database import SessionLocal
from app.models import Fixture, Team

async def load_fixtures(season="2025-26"):
    fpl_api = FPLAPIService()
    db = SessionLocal()
    try:
        # Fetch fixtures from FPL API
        fixtures = await fpl_api.get_fixtures()
        
        # Get team mapping
        teams = {t.id: t for t in db.query(Team).all()}
        
        # Save fixtures
        for fixture_data in fixtures:
            # Map and save fixture
            # ... implementation needed
            pass
        
        db.commit()
    finally:
        db.close()
        await fpl_api.close()
```

### Run Backtest
```python
# backend/run_backtest.py (create this)
from app.services.backtest import BacktestEngine
from app.database import SessionLocal
from app.models import PlayerGameweekStats

def run_backtest():
    db = SessionLocal()
    try:
        # Load training data
        # Run backtest
        # Save results
        pass
    finally:
        db.close()
```

## Automated Population Script

A comprehensive script has been created to populate all tables automatically:

### Usage

```bash
# Run the script to populate all tables
cd /root/fpl-prediction-platform
docker compose exec backend python3 populate_all_tables.py
```

The script will:
1. Load all fixtures from FPL API
2. Aggregate team statistics from player gameweek stats
3. Optimize form alpha coefficient
4. Fit team FDR (Fixture Difficulty Rating) model
5. Create entity mappings for all players
6. Track model performance metrics
7. Run backtest (optional, can be skipped with Ctrl+C)

### What the Script Does

1. **Loads Fixtures** - Fetches all fixtures from FPL API and saves to database
2. **Aggregates TeamStats** - Calculates team-level statistics from PlayerGameweekStats
3. **Optimizes FormAlpha** - Runs Bayesian optimization to find optimal form decay coefficient
4. **Fits TeamFDR** - Calculates team attack/defense strengths from historical data
5. **Runs Backtest** - Executes expanding window backtest and saves results
6. **Creates EntityMappings** - Creates basic entity mappings for all players
7. **Tracks ModelPerformance** - Records model performance metrics per gameweek

### Execution Time

- **Quick operations** (Fixtures, TeamStats, FormAlpha, TeamFDR, EntityMapping): ~30 seconds
- **Backtest**: ~10-30 minutes (depends on number of gameweeks and data volume)
- **Total**: ~15-30 minutes for full population

### Notes

- The backtest is the slowest operation as it trains models for each gameweek
- You can interrupt the script (Ctrl+C) and it will save what it has completed
- Re-running the script will update existing records rather than duplicate them

## Summary

Most of these tables are empty because:
1. **ETL only loads core data**: Players and PlayerGameweekStats
2. **Feature tables require computation**: FormAlpha, TeamFDR need model fitting
3. **Analytics tables require analysis**: BacktestResult/Summary need backtests to run
4. **Supporting tables need separate scripts**: Fixtures, TeamStats, EntityMapping

The platform is functional with just Players and PlayerGameweekStats. The other tables provide additional features and analytics that can be populated as needed.

**Use `populate_all_tables.py` to automatically fill all empty tables.**
