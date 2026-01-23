# FPL Prediction Platform - Refactoring Plan
## From Monolithic to Modular Architecture (4GB RAM Constraint)

**Date**: 2025-01-21  
**Objective**: Transform codebase from monolithic "spaghetti" state into modular, resource-efficient architecture suitable for 2vCPU/4GB RAM environment.

---

## Executive Summary

### Current State Analysis

**Critical Issues Identified**:
1. **Monolithic Files**:
   - `fpl_api.py`: 1,094 lines (RateLimiter, Cache, FPLAPIService all in one file)
   - `ml_engine.py`: 1,988 lines (4 model classes + PLEngine orchestrator)
   - `predictive_engine.py`: 695 lines (duplicate XMinsModel, AttackModel, DefenseModel)

2. **Duplicate Logic**:
   - `PredictiveEngine` vs `PLEngine`: Both initialized in `main.py`, duplicate XMinsModel classes
   - `backtest.py` vs `backtesting.py`: Two different backtest implementations
   - `solver.py` vs `team_solver.py`: Two solver implementations

3. **Memory Issues**:
   - Models loaded into memory permanently (no lazy loading/unloading)
   - No strict data type optimization (int64, float64 everywhere)
   - List comprehensions creating large in-memory lists

4. **Code Quality**:
   - Missing type hints in many functions
   - Inconsistent error handling
   - No centralized exception handler

---

## Task 1: Code Cleanup & Dead Code Elimination

### 1.1 Audit Results

**Files to Remove/Consolidate**:
- ✅ `predictive_engine.py` → **REMOVE** (duplicate of `ml_engine.py`, `PLEngine` is the active one per ARCHITECTURE_MAP.md)
- ⚠️ `backtest.py` vs `backtesting.py` → **KEEP BOTH** (different use cases: `backtest.py` for scripts, `backtesting.py` for API)
- ⚠️ `solver.py` vs `team_solver.py` → **KEEP BOTH** (different use cases: `solver.py` for backtests, `team_solver.py` for API)

**Unused Imports to Remove**:
- Scan all files for unused imports using `pylint` or `ruff`

**Deprecated Features**:
- No "personas" or "experiments" found in code (only in documentation)
- All features in ARCHITECTURE_MAP.md are active

### 1.2 Action Items

1. **Remove `predictive_engine.py`**:
   - Update `main.py` to remove `PredictiveEngine` import and initialization
   - Ensure all endpoints use `PLEngine` from `ml_engine.py`
   - Verify no other files import `PredictiveEngine`

2. **Clean Unused Imports**:
   - Run `ruff check --select F401` to find unused imports
   - Remove all unused imports

3. **Remove Commented Code**:
   - Search for large commented blocks
   - Remove or convert to proper documentation

---

## Task 2: Modularization (Breaking the Monoliths)

### 2.1 Refactor `fpl_api.py` → `app/services/fpl/`

**Current Structure** (1,094 lines):
```
fpl_api.py
├── RateLimiter (96 lines)
├── InMemoryCache (45 lines)
└── FPLAPIService (953 lines)
    ├── HTTP client logic
    ├── Data transformation
    ├── Cache management
    └── Database operations (mixed with API logic)
```

**Target Structure**:
```
app/services/fpl/
├── __init__.py              # Export public API
├── client.py                # HTTP client, rate limiting (200 lines)
├── cache.py                 # LRU cache with TTL (100 lines)
├── processors.py             # Data transformation, normalization (300 lines)
├── repository.py             # Database operations (Repository Pattern) (250 lines)
└── service.py                # FPLAPIService orchestrator (300 lines)
```

**Detailed Breakdown**:

#### `client.py` (HTTP & Rate Limiting)
```python
class RateLimiter:
    """DefCon rate limiter with exponential backoff"""
    
class FPLHTTPClient:
    """Async HTTP client with rate limiting"""
    async def get_bootstrap_data() -> Dict
    async def get_player_data(player_id: int) -> Dict
    async def get_fixtures() -> List[Dict]
```

#### `cache.py` (Strict LRU Caching)
```python
class LRUCache:
    """Memory-efficient LRU cache with TTL"""
    async def get(key: str) -> Optional[Any]
    async def set(key: str, value: Any, ttl: int)
    async def clear(key: Optional[str] = None)
```

#### `processors.py` (Data Transformation)
```python
class FPLDataProcessor:
    """Transform and normalize FPL API responses"""
    def extract_players_from_bootstrap(bootstrap_data: Dict) -> List[Dict]
    def extract_player_history(player_summary: Dict) -> List[Dict]
    def normalize_dgw_points(points: int, matches: int) -> float
```

#### `repository.py` (Repository Pattern)
```python
class FPLRepository:
    """Database operations for FPL data"""
    async def save_player(player_data: Dict) -> Player
    async def save_gameweek_stats(stats_data: Dict) -> PlayerGameweekStats
    async def get_player_by_fpl_id(fpl_id: int) -> Optional[Player]
```

#### `service.py` (Orchestrator)
```python
class FPLAPIService:
    """Main service orchestrating client, cache, processors, repository"""
    def __init__(client, cache, processor, repository)
    async def get_bootstrap_data() -> Dict
    async def fetch_comprehensive_player_data() -> Dict
```

### 2.2 Refactor `ml_engine.py` → `app/services/ml/`

**Current Structure** (1,988 lines):
```
ml_engine.py
├── XMinsModel (324 lines)
├── AttackModel (438 lines)
├── DefenseModel (360 lines)
└── PLEngine (859 lines)
```

**Target Structure**:
```
app/services/ml/
├── __init__.py              # Export public API
├── interfaces.py            # Abstract ModelInterface (50 lines)
├── strategies/
│   ├── __init__.py
│   ├── xmins_strategy.py    # XMinsModel (300 lines)
│   ├── attack_strategy.py   # AttackModel (400 lines)
│   ├── defense_strategy.py  # DefenseModel (350 lines)
│   └── poisson_strategy.py  # Poisson regression (100 lines)
├── model_loader.py          # Lazy loading/unloading with gc.collect() (150 lines)
└── engine.py                # PLEngine orchestrator (400 lines)
```

**Detailed Breakdown**:

#### `interfaces.py` (Abstract Interface)
```python
from abc import ABC, abstractmethod
from typing import Dict, Optional, List

class ModelInterface(ABC):
    """Abstract interface for all ML models"""
    
    @abstractmethod
    async def load(self) -> None:
        """Load model into memory"""
        pass
    
    @abstractmethod
    async def unload(self) -> None:
        """Unload model from memory, call gc.collect()"""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Dict[str, float]:
        """Make prediction"""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass
```

#### `strategies/xmins_strategy.py`
```python
class XMinsStrategy(ModelInterface):
    """XGBoost-based xMins prediction"""
    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self._loaded = False
    
    async def load(self) -> None:
        if not self._loaded:
            # Load from pickle
            self._loaded = True
    
    async def unload(self) -> None:
        if self._loaded:
            self.model = None
            gc.collect()
            self._loaded = False
```

#### `model_loader.py` (Resource Management)
```python
class ModelLoader:
    """Manages model lifecycle: load on demand, unload immediately"""
    
    async def load_model(self, strategy: ModelInterface) -> None:
        """Load model, ensure previous model is unloaded"""
        await strategy.load()
    
    async def unload_model(self, strategy: ModelInterface) -> None:
        """Unload model and force garbage collection"""
        await strategy.unload()
        gc.collect()
```

#### `engine.py` (PLEngine)
```python
class PLEngine:
    """Orchestrates all ML strategies with lazy loading"""
    
    def __init__(self):
        self.xmins = XMinsStrategy()
        self.attack = AttackStrategy()
        self.defense = DefenseStrategy()
        self.loader = ModelLoader()
    
    async def predict(self, player_data: Dict) -> Dict[str, float]:
        """Load models, predict, unload immediately"""
        await self.loader.load_model(self.xmins)
        try:
            result = await self._calculate_expected_points(player_data)
            return result
        finally:
            await self.loader.unload_model(self.xmins)
            # Unload other models too
```

---

## Task 3: Resource Optimization (The 4GB Limit)

### 3.1 Data Type Optimization

**Create `app/utils/data_types.py`**:
```python
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage"""
    for col in df.columns:
        if df[col].dtype == 'int64':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype(np.float32)
        
        elif df[col].dtype == 'object':
            # Convert repeated strings to category
            if df[col].nunique() / len(df) < 0.5:  # < 50% unique
                df[col] = df[col].astype('category')
    
    return df
```

**Apply to all ETL operations**:
- `etl_service.py`: Optimize DataFrames before bulk insert
- `feature_engineering.py`: Optimize feature DataFrames
- `ml_engine.py`: Optimize training data

### 3.2 Generators for Large Data Processing

**Replace list comprehensions with generators**:

**Before** (ETL):
```python
players = [process_player(p) for p in all_players]  # Creates large list
```

**After**:
```python
def process_players_generator(all_players):
    for player in all_players:
        yield process_player(player)

# Use in batches
async def bulk_upsert_players_generator(players_gen, batch_size=100):
    batch = []
    for player in players_gen:
        batch.append(player)
        if len(batch) >= batch_size:
            await upsert_batch(batch)
            batch = []
    if batch:
        await upsert_batch(batch)
```

### 3.3 Async I/O Verification

**Audit all I/O operations**:
- ✅ Database queries: Already async (SQLAlchemy async)
- ✅ HTTP requests: Already async (httpx.AsyncClient)
- ⚠️ File I/O: Check for sync file operations (model loading)
- ⚠️ External API calls: Ensure all are async

**Fix sync operations**:
```python
# Before
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# After
async def load_model_async(path: str):
    loop = asyncio.get_event_loop()
    with open(path, 'rb') as f:
        model = await loop.run_in_executor(None, pickle.load, f)
    return model
```

---

## Task 4: GitHub & Development Standards

### 4.1 Type Hinting

**Enforce strict type hints**:
```python
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

def process_player(
    player_data: Dict[str, Union[int, float, str]],
    gameweek: int,
    season: str = "2025-26"
) -> Optional[Dict[str, float]]:
    """Process player data with strict typing"""
    pass
```

**Add to all public methods**:
- All service classes
- All API endpoints
- All repository methods

### 4.2 Google-Style Docstrings

**Template**:
```python
def calculate_expected_points(
    self,
    player_data: Dict[str, Any],
    fixture_data: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Calculate expected points for a player using component models.
    
    Args:
        player_data: Dictionary containing player statistics and metadata.
            Required keys: 'id', 'position', 'price', 'form'.
        fixture_data: Optional dictionary with fixture information.
            Keys: 'opponent_id', 'is_home', 'fdr'.
    
    Returns:
        Dictionary with prediction components:
        - 'xp': Expected points (float)
        - 'xmins': Expected minutes (float)
        - 'xg': Expected goals (float)
        - 'xa': Expected assists (float)
        - 'xcs': Expected clean sheet probability (float)
    
    Raises:
        ValueError: If player_data is missing required keys.
        ModelNotLoadedError: If ML models are not loaded.
    
    Example:
        >>> player = {'id': 1, 'position': 'MID', 'price': 8.5, 'form': 5.2}
        >>> result = engine.calculate_expected_points(player)
        >>> print(result['xp'])
        5.8
    """
    pass
```

### 4.3 Centralized Error Handling

**Create `app/exceptions.py`**:
```python
from fastapi import HTTPException
from typing import Optional

class AppException(HTTPException):
    """Base exception for application errors"""
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

class FPLAPIError(AppException):
    """FPL API related errors"""
    def __init__(self, detail: str):
        super().__init__(status_code=503, detail=detail, error_code="FPL_API_ERROR")

class ModelNotLoadedError(AppException):
    """ML model not loaded error"""
    def __init__(self, model_name: str):
        super().__init__(
            status_code=503,
            detail=f"Model {model_name} is not loaded",
            error_code="MODEL_NOT_LOADED"
        )

class RateLimitExceededError(AppException):
    """Rate limit exceeded"""
    def __init__(self):
        super().__init__(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            error_code="RATE_LIMIT_EXCEEDED"
        )
```

**Add to `main.py`**:
```python
from app.exceptions import AppException

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

---

## Execution Plan

### Phase 1: Code Cleanup (Low Risk)
1. ✅ Remove `predictive_engine.py`
2. ✅ Clean unused imports
3. ✅ Remove commented code

**Estimated Time**: 2-3 hours  
**Risk**: Low (removing dead code)

### Phase 2: fpl_api.py Refactoring (Medium Risk)
1. Create `app/services/fpl/` directory structure
2. Extract `RateLimiter` → `client.py`
3. Extract `InMemoryCache` → `cache.py`
4. Extract data processors → `processors.py`
5. Create repository pattern → `repository.py`
6. Refactor `FPLAPIService` → `service.py`
7. Update all imports in `main.py` and other files
8. Test all FPL API endpoints

**Estimated Time**: 6-8 hours  
**Risk**: Medium (many dependencies)

### Phase 3: ml_engine.py Refactoring (High Risk)
1. Create `app/services/ml/` directory structure
2. Create `ModelInterface` abstract class
3. Extract `XMinsModel` → `strategies/xmins_strategy.py`
4. Extract `AttackModel` → `strategies/attack_strategy.py`
5. Extract `DefenseModel` → `strategies/defense_strategy.py`
6. Create `model_loader.py` with lazy loading/unloading
7. Refactor `PLEngine` → `engine.py`
8. Update all imports
9. Test ML predictions and memory usage

**Estimated Time**: 8-10 hours  
**Risk**: High (core ML logic, must preserve functionality)

### Phase 4: Resource Optimization (Medium Risk)
1. Create `app/utils/data_types.py` with optimization functions
2. Apply to all ETL operations
3. Replace list comprehensions with generators
4. Audit and fix sync I/O operations
5. Test memory usage improvements

**Estimated Time**: 4-6 hours  
**Risk**: Medium (data processing changes)

### Phase 5: Code Quality (Low Risk)
1. Add type hints to all public methods
2. Add Google-style docstrings
3. Create centralized exception handler
4. Run linters and fix issues

**Estimated Time**: 4-6 hours  
**Risk**: Low (additive changes)

---

## Testing Strategy

### Unit Tests
- Test each extracted module independently
- Mock dependencies for isolated testing

### Integration Tests
- Test FPL API service end-to-end
- Test ML engine predictions match original
- Test memory usage with `memory_profiler`

### Performance Tests
- Measure memory usage before/after
- Measure API response times
- Verify models unload correctly

---

## Success Metrics

1. **Code Organization**:
   - ✅ No file > 500 lines
   - ✅ Clear separation of concerns
   - ✅ Repository pattern implemented

2. **Memory Usage**:
   - ✅ Models unload after inference (gc.collect() called)
   - ✅ DataFrames use optimized types (int16/int32, float32, category)
   - ✅ Generators used for large data processing

3. **Code Quality**:
   - ✅ 100% type hint coverage on public methods
   - ✅ All public methods have docstrings
   - ✅ Centralized error handling

---

## Next Steps

**Please review this plan and approve before I begin implementation.**

I recommend starting with **Phase 1 (Code Cleanup)** as it's low-risk and will immediately reduce codebase size. Then proceed with **Phase 2 (fpl_api.py)** as it's the primary data ingest point and will have the most immediate impact on maintainability.

**Questions for you**:
1. Should I proceed with removing `predictive_engine.py` immediately?
2. Do you want me to start with Phase 1, or would you prefer to review the plan first?
3. Are there any specific endpoints or features you want me to prioritize testing?
