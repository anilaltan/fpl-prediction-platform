<overview>

Problem Statement
The FPL Prediction Platform features a strong modular architecture but suffers from critical test coverage gaps (estimated >70% missing) and potential data integrity risks due to race conditions. Currently, there is no verification layer to ensure that ML models produce functional results or that the optimization solver handles large-scale problem instances without hanging.

Target Users
FPL Managers: End-users expecting high-fidelity point predictions and stable team optimizations.

Platform Developers: Engineers requiring a regression-proof codebase to maintain Moneyball-style architectural principles.

Success Metrics
Test Coverage: Achieve 80%+ unit test coverage for core logic functions (Feature Engineering, Solver constraints).

Operational Readiness: 100% detection of degraded states via an enhanced Health Check (including DB and Model status).

Data Integrity: Zero duplicate entries or corrupted stats during concurrent ETL operations.

</overview>

<functional-decomposition>

Capability Tree
Capability: ML Prediction Reliability
Ensures that machine learning models are not only loaded but are mathematically functional and providing consistent point predictions.

Feature: Functional Model Validation
Description: Verifies models produce non-zero predictions post-load to detect silent corruption.

Inputs: Mock player/fixture data.

Outputs: Validation boolean + test prediction result.

Behavior: Executes a "smoke test" prediction immediately after startup; triggers critical exit if result is 0.0.

Capability: Team Optimization (Solver)
Utilizes Integer Linear Programming (ILP) to generate the highest-value squad under complex constraints.

Feature: Solver Timeout & Error Handling
Description: Prevents the API from hanging on large problem instances.

Inputs: Player list, budget, time limit parameter.

Outputs: Optimal solution or a handled Timeout exception.

Behavior: Configures the PuLP solver with a hard timeLimit (default 30s) and returns a structured error if exceeded.

</functional-decomposition>

<structural-decomposition>

Repository Structure
backend/
├── app/
│   ├── services/
│   │   ├── ml/
│   │   │   ├── engine.py       # ML Injection & Smoke Testing
│   │   ├── etl/
│   │   │   ├── service.py      # Transaction Isolation Logic
│   ├── api/
│   │   ├── health.py           # Enhanced Health Check Endpoints
├── tests/                      # NEW: Test Infrastructure
│   ├── unit/
│   ├── integration/
│   └── performance/
Module Definitions
Module: PLEngine Injection
Maps to capability: ML Prediction Reliability.

Responsibility: Decouple model loading and feature engineering from the execution engine to allow unit testing with mocks.

Exports: validate_models_loaded(), calculate_expected_points().

</structural-decomposition>

<dependency-graph>

Dependency Chain
Foundation Layer (Phase 0)
Test Infrastructure: Setup Pytest, async fixtures, and a dedicated test database.

Centralized Exception Handlers: Standardize AppException usage across all services to prevent silent failures.

Logic & Data Layer (Phase 1)
ETL Transaction Safety: Implement SELECT FOR UPDATE and explicit transaction blocks to prevent race conditions.

Solver Constraint Extractors: Decouple constraint logic from PuLP internals for isolated unit testing.

Integration Layer (Phase 2)
Enhanced Health Monitoring: Multi-point check covering DB connectivity, Model status, and Memory usage.

E2E Prediction Pipeline: Integration test covering FPL API -> ETL -> ML -> Frontend Response.

</dependency-graph>

<implementation-roadmap>

Development Phases
Phase 0: Stability & Foundation
Goal: Address high-risk operational gaps and prepare the environment for testing.

Tasks:

[ ] Infrastructure: Configure tests/ directory and conftest.py.

[ ] ETL Locking: Implement asyncio.Lock for cache updates and transaction isolation for DB UPSERTs.

[ ] Solver Hardening: Add a 30s timeout to TeamSolver and error fallback for empty solutions.

Phase 1: Quality Assurance & Coverage
Goal: Reach the target coverage and implement proactive health monitoring.

Tasks:

[ ] Unit Tests: Achieve 100% coverage for Feature Engineering pure functions and Solver constraints.

[ ] Health Endpoint: Deploy /api/health/models for real-time model status tracking.

</implementation-roadmap>

<test-strategy>

Test Pyramid
        /\
       /E2E\       ← 10% (End-to-end prediction flow)
      /------\
     /Integration\ ← 30% (API-to-Database-to-ML interactions)
    /------------\
   /  Unit Tests  \ ← 60% (Isolated mathematical & logic checks)
  /----------------\
Critical Test Scenarios
ETL Race Condition: Simultaneously updating the same player record must not result in duplicate entries or database locks.

Solver Boundary: Testing the optimizer with 200+ players over a 5-week horizon to verify performance targets.

ML Smoke Test: Validating that loading a model with corrupted weights triggers a ModelError during the startup phase.

</test-strategy>

<risks>

Technical Risks
Risk: Non-deterministic Solver Solutions

Impact: High (Inconsistent team recommendations).

Likelihood: Medium.

Mitigation: Implement Property-Based Testing using Hypothesis to verify that generated solutions always satisfy constraints (budget, squad size, position limits).

</risks>