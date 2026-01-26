<rpg-method>
# QA Refactoring Plan - FPL Platform
</rpg-method>

<overview>
## Problem Statement
The FPL Prediction Platform lacks critical test coverage (0%), inconsistent error handling, and validates models too late (runtime vs startup). This creates high risk for production stability.

## Success Metrics
- Test Coverage: >80% for pure functions, >60% for services
- Startup Safety: API fails fast (exit code 1) if models are invalid
- Error Consistency: 100% usage of `AppException` across all services
</overview>

<functional-decomposition>
## Capability Tree

### Capability: Core Stability
Enforcing strict validation and error handling boundaries.

#### Feature: Startup Validation
- **Description**: Validate ML models and DB connection before API accepts traffic.
- **Inputs**: Model paths, DB credentials.
- **Outputs**: Boolean (Healthy/Unhealthy).
- **Behavior**: Check file checksums, load test models, exit process on failure.

#### Feature: Error Standardization
- **Description**: Replace generic `Exception` with domain-specific `AppException`.
- **Behavior**: Map low-level errors (SQLAlchemy, FileNotFoundError) to API-friendly error codes.

### Capability: Test Infrastructure
Establishing the testing pyramid.

#### Feature: Unit Test Suite
- **Description**: Isolated tests for `team_solver`, `feature_engineering` and `fpl_api`.
- **Inputs**: Mocked data.
- **Outputs**: Test reports.
</functional-decomposition>

<structural-decomposition>
## Module Definitions

### Module: infrastructure
- **Maps to capability**: Core Stability
- **Files**:
  - `backend/app/exceptions.py` (Standardized errors)
  - `backend/app/main.py` (Startup events)
  - `backend/scripts/validate_models.py` (New validation script)

### Module: testing
- **Maps to capability**: Test Infrastructure
- **Files**:
  - `backend/tests/conftest.py` (Fixtures)
  - `backend/tests/unit/` (Unit tests)
</structural-decomposition>

<dependency-graph>
## Dependency Chain

### Foundation Layer (Phase 0)
- **infrastructure**: No dependencies. (Must fix error classes and validation first).

### Refactoring Layer (Phase 1)
- **ml_engine_refactor**: Depends on [infrastructure]. (Inject `ModelLoader` to make it testable).
- **team_solver_refactor**: Depends on [infrastructure]. (Add timeout and input validation).

### Quality Layer (Phase 2)
- **testing**: Depends on [ml_engine_refactor, team_solver_refactor]. (Cannot write clean tests until refactoring is done).
</dependency-graph>

<implementation-roadmap>
## Development Phases

### Phase 0: Foundation (Critical Risks)
**Goal**: Stop the bleeding. Prevent API from starting in a broken state.
**Tasks**:
- [ ] Implement `Startup Validation` logic in `main.py`
- [ ] Create `scripts/validate_models.py` for Docker healthcheck
- [ ] Standardize `AppException` in `team_solver.py` and `etl_service.py`

### Phase 1: Testability Refactoring
**Goal**: Make code testable by decoupling dependencies.
**Tasks**:
- [ ] Refactor `PLEngine` to use dependency injection for `ModelLoader`
- [ ] Extract constraint logic from `TeamSolver` into pure functions

### Phase 2: Test Implementation
**Goal**: Reach 80% coverage on core logic.
**Entry Criteria**: Phase 1 complete.
**Tasks**:
- [ ] Setup `pytest` and `conftest.py` fixtures
- [ ] Write unit tests for `feature_engineering.py` (Pure functions)
- [ ] Write integration tests for `optimize-team` endpoint
</implementation-roadmap>