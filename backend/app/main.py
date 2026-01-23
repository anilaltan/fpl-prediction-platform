from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.routing import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from app.database import engine, Base, get_db
from app.exceptions import (
    AppException,
    ValidationError,
    NotFoundError,
    DatabaseError,
    ExternalAPIError,
    ModelError,
    RateLimitError,
    handle_app_exception,
    handle_generic_exception,
    handle_http_exception
)
from app.services.feature_engineering import FeatureEngineeringService
from app.services.fpl import FPLAPIService
from app.schemas import (
    FormAlphaOptimizeRequest, FormAlphaResponse,
    FDRFitRequest, FDRResponse, DefConFeaturesResponse,
    StochasticFDRRequest, StochasticFDRResponse,
    FDRComparisonResponse, FDRVerificationResponse,
    XMinsPredictionRequest, XMinsPredictionResponse,
    AttackPredictionRequest, AttackPredictionResponse,
    DefensePredictionRequest, DefensePredictionResponse,
    MomentumPredictionRequest, MomentumPredictionResponse,
    ComprehensivePredictionRequest, ComprehensivePredictionResponse,
    TeamOptimizationRequest, TeamOptimizationResponse,
    OwnershipArbitrageRequest, OwnershipArbitrageResponse,
    CaptainSelectionRequest, CaptainSelectionResponse,
    ChipAnalysisRequest, ChipAnalysisResponse,
    ComprehensiveRiskAnalysisRequest, ComprehensiveRiskAnalysisResponse,
    BacktestRequest, BacktestResponse,
    PlayerDisplayData, DreamTeamResponse, DreamTeamPlayer,
    UnderstatPlayerData, FBrefDefensiveData, EnrichedPlayerData,
    PlayerResolutionRequest, PlayerResolutionResponse,
    BulkResolutionRequest, BulkResolutionResponse,
    ManualMappingRequest, OverrideMappingRequest,
    BulkResolutionReport,
    DataCleaningRequest, DataCleaningResponse,
    BulkCleaningRequest, BulkCleaningResponse,
    DefConMetricsResponse,
    MarketIntelligenceResponse, MarketIntelligencePlayer,
    TeamPlanRequest, TeamPlanResponse, TransferStrategy
)
from app.services.team_solver import TeamSolver
from app.services.risk_management import RiskManagementService
from app.services.backtesting import BacktestingEngine
from app.services.ml_engine import PLEngine
from app.services.third_party_data import ThirdPartyDataService, UnderstatService, FBrefService
from app.services.entity_resolution import EntityResolutionService
from app.services.data_cleaning import DataCleaningService
from app.services.etl_service import ETLService
from app.services.market_intelligence import MarketIntelligenceService
from app.models import Player, PlayerGameweekStats, Prediction
import logging
import asyncio
from functools import lru_cache
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FPL Point Prediction API",
    description="Machine Learning API for Fantasy Premier League point predictions",
    version="2.0.0"
)

# =============================================================================
# CENTRALIZED ERROR HANDLING (Task 4.1)
# =============================================================================

@app.exception_handler(AppException)
async def app_exception_handler(request, exc: AppException):
    """Handle application-specific exceptions."""
    return handle_app_exception(exc)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle FastAPI HTTPExceptions with standardized format."""
    return handle_http_exception(exc)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    """Handle all other exceptions and convert to standardized format."""
    return handle_generic_exception(exc)

# Initialize services
fpl_api = FPLAPIService()
third_party_service = ThirdPartyDataService()
entity_resolution = EntityResolutionService()
data_cleaning = DataCleaningService()
etl_service = ETLService()
feature_service = FeatureEngineeringService(
    third_party_service=third_party_service,
    db_session=None  # Will be passed per-request via Depends(get_db)
)
team_solver = TeamSolver()
risk_service = RiskManagementService()
backtesting_engine = BacktestingEngine()
ml_engine = PLEngine()
market_intelligence_service = MarketIntelligenceService()

# =============================================================================
# GLOBAL IN-MEMORY CACHE SYSTEM (Graceful Degradation Architecture)
# =============================================================================
# Purpose: Prevent server overload from repeated ML calculations
# Strategy: Cache-first approach with fallback to basic data

DATA_CACHE = {
    "players_by_gw": {},       # Dict[int, List[PlayerDisplayData]] - gameweek -> players
    "dream_team_by_gw": {},    # Dict[int, DreamTeamResponse] - gameweek -> dream team
    "last_updated_by_gw": {},  # Dict[int, float] - gameweek -> timestamp
    "is_computing": False,     # Lock flag to prevent concurrent calculations
    "error_count": 0,          # Track consecutive errors
    "current_gameweek": None,  # Cached current gameweek from FPL API
    "gameweek_last_updated": 0,  # Timestamp of last gameweek update
}

# Cache TTL: 10 minutes (600 seconds)
CACHE_TTL_SECONDS = 600
GAMEWEEK_CACHE_TTL_SECONDS = 3600  # 1 hour for gameweek (changes less frequently)

# Max errors before fallback mode
MAX_ERROR_COUNT = 3

def _is_cache_valid(gameweek: int) -> bool:
    """Check if cache is valid for a specific gameweek (not expired and has data)"""
    if gameweek not in DATA_CACHE["players_by_gw"]:
        return False
    if gameweek not in DATA_CACHE["last_updated_by_gw"]:
        return False
    elapsed = datetime.now().timestamp() - DATA_CACHE["last_updated_by_gw"][gameweek]
    return elapsed < CACHE_TTL_SECONDS

def _get_cached_players(gameweek: int):
    """Get cached players list for a specific gameweek if valid"""
    if _is_cache_valid(gameweek):
        return DATA_CACHE["players_by_gw"][gameweek]
    return None

def _get_cached_dream_team(gameweek: int):
    """Get cached dream team for a specific gameweek if valid"""
    if _is_cache_valid(gameweek) and gameweek in DATA_CACHE["dream_team_by_gw"]:
        return DATA_CACHE["dream_team_by_gw"][gameweek]
    return None

def _update_cache(gameweek: int, players_data: List, dream_team=None):
    """Update the global cache with new data for a specific gameweek"""
    DATA_CACHE["players_by_gw"][gameweek] = players_data
    if dream_team:
        DATA_CACHE["dream_team_by_gw"][gameweek] = dream_team
    DATA_CACHE["last_updated_by_gw"][gameweek] = datetime.now().timestamp()
    DATA_CACHE["error_count"] = 0
    logger.info(f"Cache updated for GW{gameweek} with {len(players_data)} players at {datetime.now()}")


async def _get_current_gameweek() -> int:
    """
    Get current gameweek from FPL API with caching.
    Falls back to 1 if API call fails.
    """
    # Check cache first
    current_time = datetime.now().timestamp()
    if DATA_CACHE["current_gameweek"] is not None:
        elapsed = current_time - DATA_CACHE["gameweek_last_updated"]
        if elapsed < GAMEWEEK_CACHE_TTL_SECONDS:
            return DATA_CACHE["current_gameweek"]
    
    # Fetch from FPL API
    try:
        current_gw = await fpl_api.get_current_gameweek()
        if current_gw:
            DATA_CACHE["current_gameweek"] = current_gw
            DATA_CACHE["gameweek_last_updated"] = current_time
            logger.info(f"Current gameweek updated: {current_gw}")
            return current_gw
    except Exception as e:
        logger.warning(f"Failed to get current gameweek: {str(e)}")
    
    # Fallback to cached value or default
    if DATA_CACHE["current_gameweek"]:
        return DATA_CACHE["current_gameweek"]
    
    logger.warning("Using default gameweek: 1")
    return 1


async def _get_next_gameweek() -> int:
    """
    Get next/upcoming gameweek from FPL API for Dashboard.
    Prioritizes is_next=True, then finds first unfinished gameweek.
    Falls back to current gameweek if next not found.
    """
    try:
        next_gw = await fpl_api.get_next_gameweek()
        if next_gw:
            logger.info(f"Next gameweek for Dashboard: {next_gw}")
            return next_gw
        
        # Fallback to current gameweek
        current_gw = await _get_current_gameweek()
        logger.warning(f"No next gameweek found, using current: {current_gw}")
        return current_gw
    except Exception as e:
        logger.warning(f"Failed to get next gameweek: {str(e)}")
        # Fallback to current gameweek
        return await _get_current_gameweek()

# Scheduler for daily refresh
scheduler = AsyncIOScheduler()

# CORS middleware
# Allow requests from localhost, Docker internal network, and VPS public IP
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://frontend:3000",  # Docker internal service name
    "http://46.224.178.180:3000",  # VPS public IP
    "*"  # Allow all origins for development (can be restricted in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "FPL Point Prediction API",
        "version": "2.0.0",
        "status": "operational",
        "architecture": "Graceful Degradation"
    }

@app.get("/health")
async def health_check():
    current_gw = DATA_CACHE.get("current_gameweek") or await _get_current_gameweek()
    cache_valid = _is_cache_valid(current_gw)
    cache_age = 0
    if current_gw in DATA_CACHE["last_updated_by_gw"]:
        cache_age = datetime.now().timestamp() - DATA_CACHE["last_updated_by_gw"][current_gw]
    cached_count = len(DATA_CACHE["players_by_gw"].get(current_gw, []))
    
    return {
        "status": "healthy",
        "current_gameweek": current_gw,
        "cache_valid": cache_valid,
        "cache_age_seconds": round(cache_age, 1) if cache_age > 0 else None,
        "cached_players_count": cached_count,
        "is_computing": DATA_CACHE["is_computing"],
        "error_count": DATA_CACHE["error_count"]
    }


@app.get("/api/cache/status")
async def get_cache_status():
    """
    Get detailed cache status for debugging.
    """
    current_gw = DATA_CACHE.get("current_gameweek") or await _get_current_gameweek()
    cache_valid = _is_cache_valid(current_gw)
    cache_age = 0
    if current_gw in DATA_CACHE["last_updated_by_gw"]:
        cache_age = datetime.now().timestamp() - DATA_CACHE["last_updated_by_gw"][current_gw]
    ttl_remaining = max(0, CACHE_TTL_SECONDS - cache_age) if cache_age > 0 else 0
    cached_count = len(DATA_CACHE["players_by_gw"].get(current_gw, []))
    
    return {
        "current_gameweek": current_gw,
        "cache_valid": cache_valid,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "cache_age_seconds": round(cache_age, 1) if cache_age > 0 else None,
        "ttl_remaining_seconds": round(ttl_remaining, 1),
        "cached_players_count": cached_count,
        "dream_team_cached": current_gw in DATA_CACHE["dream_team_by_gw"],
        "is_computing": DATA_CACHE["is_computing"],
        "error_count": DATA_CACHE["error_count"],
        "last_updated": datetime.fromtimestamp(DATA_CACHE["last_updated_by_gw"][current_gw]).isoformat() if current_gw in DATA_CACHE["last_updated_by_gw"] else None
    }


@app.post("/api/cache/refresh")
async def refresh_cache(db: Session = Depends(get_db)):
    """
    Force refresh the player cache.
    Use this after model training or data updates.
    """
    # Clear existing cache for current gameweek
    current_gw = await _get_current_gameweek()
    if current_gw in DATA_CACHE["players_by_gw"]:
        del DATA_CACHE["players_by_gw"][current_gw]
    if current_gw in DATA_CACHE["dream_team_by_gw"]:
        del DATA_CACHE["dream_team_by_gw"][current_gw]
    if current_gw in DATA_CACHE["last_updated_by_gw"]:
        del DATA_CACHE["last_updated_by_gw"][current_gw]
    
    # Trigger new calculation with current gameweek
    logger.info("[CACHE] Manual refresh triggered")
    
    try:
        current_gw = await _get_current_gameweek()
        players = await get_all_players(gameweek=current_gw, db=db)
        return {
            "status": "success",
            "message": f"Cache refreshed with {len(players)} players for GW{current_gw}",
            "gameweek": current_gw,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.post("/api/predictions/update")
async def update_predictions_endpoint(
    gameweek: Optional[int] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Manually trigger batch prediction update.
    Calculates ML predictions for all players and stores in Prediction table.
    
    Args:
        gameweek: Optional gameweek number (default: current gameweek)
        background_tasks: If provided, runs in background
    
    Returns:
        Status message
    """
    try:
        from app.scripts.update_predictions import update_predictions_for_gameweek, update_predictions_for_current_gameweek
        from app.database import SessionLocal
        
        if background_tasks:
            # Run in background
            async def bg_task():
                db = SessionLocal()
                try:
                    if gameweek:
                        await update_predictions_for_gameweek(db, gameweek)
                    else:
                        await update_predictions_for_current_gameweek()
                finally:
                    db.close()
            
            background_tasks.add_task(bg_task)
            return {
                "status": "started",
                "message": "Prediction update started in background",
                "gameweek": gameweek or "current"
            }
        else:
            # Run synchronously
            db = SessionLocal()
            try:
                if gameweek:
                    result = await update_predictions_for_gameweek(db, gameweek)
                else:
                    current_gw = await _get_current_gameweek()
                    result = await update_predictions_for_gameweek(db, current_gw)
                
                return {
                    "status": "success",
                    "message": f"Predictions updated: {result.get('updated_count', 0)} players",
                    "result": result
                }
            finally:
                db.close()
                
    except Exception as e:
        logger.error(f"Error updating predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Feature Engineering Endpoints

@app.post("/api/features/optimize-form-alpha", response_model=FormAlphaResponse)
async def optimize_form_alpha(
    request: FormAlphaOptimizeRequest,
    db: Session = Depends(get_db),
    store_result: bool = True
):
    """
    Optimize form alpha coefficient using Bayesian Optimization.
    Minimizes RMSE by finding optimal exponential decay weight.
    Tracks convergence and stores results in database.
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.historical_data)
        
        if 'points' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="Historical data must include 'points' column"
            )
        
        # Optimize alpha with convergence tracking
        result = feature_service.optimize_form_alpha(
            df,
            lookback_weeks=request.lookback_weeks,
            n_calls=request.n_calls if hasattr(request, 'n_calls') else 50
        )
        
        # Store result in database if requested
        if store_result:
            try:
                from app.models import FormAlpha
                from sqlalchemy import and_
                
                # Get current gameweek (or use 1 as default)
                current_gw = fpl_api.get_current_gameweek() if hasattr(fpl_api, 'get_current_gameweek') else 1
                
                # Check if entry exists for this gameweek
                existing = db.query(FormAlpha).filter(
                    FormAlpha.gameweek == current_gw
                ).first()
                
                if existing:
                    # Update existing entry
                    existing.optimal_alpha = result['optimal_alpha']
                    existing.rmse = result['best_rmse']
                    existing.lookback_weeks = request.lookback_weeks
                else:
                    # Create new entry
                    new_entry = FormAlpha(
                        gameweek=current_gw,
                        optimal_alpha=result['optimal_alpha'],
                        rmse=result['best_rmse'],
                        lookback_weeks=request.lookback_weeks
                    )
                    db.add(new_entry)
                
                db.commit()
                logger.info(f"Stored optimized alpha for gameweek {current_gw}")
            except Exception as e:
                logger.warning(f"Failed to store alpha in database: {str(e)}")
                db.rollback()
        
        return FormAlphaResponse(
            optimal_alpha=result['optimal_alpha'],
            rmse=result['best_rmse'],
            lookback_weeks=request.lookback_weeks,
            converged=result.get('converged', False),
            iterations=result.get('iterations', 0)
        )
    except Exception as e:
        logger.error(f"Error optimizing form alpha: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/features/fit-fdr")
async def fit_fdr_model(request: FDRFitRequest, db: Session = Depends(get_db)):
    """
    Fit Dixon-Coles model for calculating team attack/defense strengths.
    Uses Poisson Regression to estimate FDR (Fixture Difficulty Rating).
    """
    try:
        feature_service.fit_fdr_model(request.fixtures)
        
        if not feature_service.fdr_model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Failed to fit FDR model. Check fixture data format."
            )
        
        # Return FDR data for all teams
        fdr_data = []
        for team_name in feature_service.fdr_model.attack_strengths.keys():
            fdr_data.append(FDRResponse(
                team_name=team_name,
                attack_strength=feature_service.fdr_model.attack_strengths[team_name],
                defense_strength=feature_service.fdr_model.defense_strengths.get(team_name, 0.0),
                home_advantage=feature_service.fdr_model.home_advantage
            ))
        
        return {
            "status": "success",
            "home_advantage": feature_service.fdr_model.home_advantage,
            "teams": fdr_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/features/fdr/{team_name}", response_model=FDRResponse)
async def get_team_fdr(team_name: str):
    """Get FDR data for a specific team"""
    if not feature_service.fdr_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="FDR model not fitted. Call /api/features/fit-fdr first."
        )
    
    if team_name not in feature_service.fdr_model.attack_strengths:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")
    
    return FDRResponse(
        team_name=team_name,
        attack_strength=feature_service.fdr_model.attack_strengths[team_name],
        defense_strength=feature_service.fdr_model.defense_strengths.get(team_name, 0.0),
        home_advantage=feature_service.fdr_model.home_advantage
    )


@app.post("/api/features/stochastic-fdr", response_model=StochasticFDRResponse)
async def get_stochastic_fdr(request: StochasticFDRRequest):
    """
    Calculate stochastic fixture difficulty (FDR 2.0) using Poisson distribution.
    Provides probability distribution of outcomes rather than just expected value.
    """
    if not feature_service.fdr_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="FDR model not fitted. Call /api/features/fit-fdr first."
        )
    
    try:
        result = feature_service.fdr_model.get_stochastic_fdr(
            team=request.team,
            opponent=request.opponent,
            is_home=request.is_home,
            n_simulations=request.n_simulations
        )
        
        return StochasticFDRResponse(**result)
    except Exception as e:
        logger.error(f"Error calculating stochastic FDR: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/features/compare-fdr", response_model=FDRComparisonResponse)
async def compare_fdr_with_fpl(request: FDRFitRequest):
    """
    Compare FDR 2.0 ratings with official FPL FDR.
    Requires FDR model to be fitted first.
    """
    if not feature_service.fdr_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="FDR model not fitted. Call /api/features/fit-fdr first."
        )
    
    try:
        result = feature_service.fdr_model.compare_with_fpl_fdr(request.fixtures)
        return FDRComparisonResponse(**result)
    except Exception as e:
        logger.error(f"Error comparing FDR: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/features/verify-fdr", response_model=FDRVerificationResponse)
async def verify_fdr_with_outcomes(request: FDRFitRequest):
    """
    Verify FDR 2.0 predictions correlate with actual goal outcomes in historical data.
    """
    if not feature_service.fdr_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="FDR model not fitted. Call /api/features/fit-fdr first."
        )
    
    try:
        result = feature_service.fdr_model.verify_with_actual_outcomes(request.fixtures)
        return FDRVerificationResponse(**result)
    except Exception as e:
        logger.error(f"Error verifying FDR: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/features/defcon", response_model=DefConFeaturesResponse)
async def get_defcon_features(
    player_id: int,
    position: str = "MID",
    minutes: int = 90
):
    """
    Calculate DefCon (Defensive Contribution) features for a player.
    Returns floor points based on 2025/26 FPL rules.
    """
    try:
        # Fetch player data from FPL API
        player_data = await fpl_api.get_player_data(player_id)
        
        # Extract element data
        element = player_data.get('history', [])
        if element:
            # Use most recent data
            recent_data = element[0] if element else {}
        else:
            recent_data = {}
        
        # Add position and minutes
        recent_data['position'] = position
        recent_data['minutes'] = minutes
        
        # Calculate DefCon features
        defcon_features = feature_service.defcon_engine.extract_defcon_features(
            recent_data, 
            position
        )
        
        return DefConFeaturesResponse(
            floor_points=defcon_features['floor_points'],
            blocks_per_90=defcon_features['blocks_per_90'],
            interventions_per_90=defcon_features['interventions_per_90'],
            passes_per_90=defcon_features['passes_per_90'],
            defcon_score=defcon_features['defcon_score']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Predictive Engine Endpoints

@app.post("/api/predictive/xmins", response_model=XMinsPredictionResponse)
async def predict_xmins(request: XMinsPredictionRequest):
    """
    Predict starting 11 probability (P_start) and expected minutes.
    Uses XGBoost or Random Forest model.
    """
    try:
        ml_engine._ensure_models_loaded()
        p_start = ml_engine.xmins_model.predict_start_probability(
            request.player_data,
            request.fixture_data
        )
        expected_minutes = ml_engine.xmins_model.predict_expected_minutes(
            request.player_data,
            request.fixture_data
        )
        
        return XMinsPredictionResponse(
            p_start=p_start,
            expected_minutes=expected_minutes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predictive/attack", response_model=AttackPredictionResponse)
async def predict_attack(request: AttackPredictionRequest):
    """
    Predict expected goals (xG) and expected assists (xA) using LightGBM.
    """
    try:
        ml_engine._ensure_models_loaded()
        # Extract opponent_data if available from fixture_data
        opponent_data = None
        if request.fixture_data:
            opponent_team_id = request.fixture_data.get('opponent_team')
            if opponent_team_id:
                # Create basic opponent_data structure
                opponent_data = {
                    'xgc_per_90': 1.5,  # Default estimate
                    'defense_strength': request.fdr_data.get('opponent_defense_strength', 0.0) if request.fdr_data else 0.0
                }
        
        predictions = ml_engine.attack_model.predict(
            request.player_data,
            request.fixture_data,
            request.fdr_data,
            opponent_data
        )
        
        return AttackPredictionResponse(
            xg=predictions['xg'],
            xa=predictions['xa']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predictive/defense", response_model=DefensePredictionResponse)
async def predict_defense(request: DefensePredictionRequest):
    """
    Predict clean sheet probability (xCS) using Poisson Distribution.
    Formula: xCS = e^(-λ) where λ is expected goals conceded.
    """
    try:
        ml_engine._ensure_models_loaded()
        xcs = ml_engine.defense_model.predict_clean_sheet_probability(
            team_data=request.team_data,
            opponent_data=request.opponent_data,
            is_home=request.is_home
        )
        
        # Calculate expected goals conceded (λ) from Poisson formula
        # If xCS = e^(-λ), then λ = -ln(xCS)
        if xcs > 0:
            expected_goals_conceded = float(-np.log(xcs))
        else:
            # Fallback: estimate from team defense strength
            team_defense = float(request.team_data.get('defense_strength', 1.0))
            opponent_attack = float(request.opponent_data.get('attack_strength', 1.0))
            home_factor = 0.9 if request.is_home else 1.0
            expected_goals_conceded = 1.5 * (1.0 / max(0.1, team_defense)) * opponent_attack * home_factor
        
        return DefensePredictionResponse(
            xcs=xcs,
            expected_goals_conceded=float(np.clip(expected_goals_conceded, 0.0, 5.0))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predictive/momentum", response_model=MomentumPredictionResponse)
async def predict_momentum(request: MomentumPredictionRequest):
    """
    Predict momentum and trend using simple trend analysis.
    Note: LSTM momentum layer removed - using simple trend calculation.
    """
    try:
        historical_points = request.historical_points
        if len(historical_points) < 2:
            return MomentumPredictionResponse(
                momentum=0.0,
                trend=0.0,
                forecast=historical_points[0] if historical_points else 0.0
            )
        
        # Simple trend calculation
        recent = np.mean(historical_points[:min(3, len(historical_points))])
        if len(historical_points) >= 6:
            previous = np.mean(historical_points[3:6])
        elif len(historical_points) >= 3:
            previous = np.mean(historical_points[1:])
        else:
            previous = historical_points[-1] if len(historical_points) > 1 else recent
        
        trend = float(recent - previous)
        forecast = float(recent + trend)
        momentum = trend  # Same as trend for simplicity
        
        return MomentumPredictionResponse(
            momentum=momentum,
            trend=trend,
            forecast=forecast
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predictive/comprehensive", response_model=ComprehensivePredictionResponse)
async def predict_comprehensive(request: ComprehensivePredictionRequest):
    """
    Get comprehensive predictions from all sub-models:
    - xMins (P_start, expected_minutes)
    - Attack (xG, xA)
    - Defense (xCS)
    - Momentum (trend, forecast)
    """
    try:
        # Use PLEngine's calculate_expected_points for main predictions
        predictions = ml_engine.calculate_expected_points(
            player_data=request.player_data,
            fixture_data=request.fixture_data,
            fdr_data=request.fdr_data,
            team_data=request.team_data,
            opponent_data=request.opponent_data,
            historical_points=request.historical_points
        )
        
        # Calculate momentum/trend from historical points
        historical_points = request.historical_points or []
        if len(historical_points) >= 2:
            recent = np.mean(historical_points[:min(3, len(historical_points))])
            if len(historical_points) >= 6:
                previous = np.mean(historical_points[3:6])
            elif len(historical_points) >= 3:
                previous = np.mean(historical_points[1:])
            else:
                previous = historical_points[-1] if len(historical_points) > 1 else recent
            trend = float(recent - previous)
            momentum = trend
        else:
            momentum = 0.0
            trend = 0.0
        
        return ComprehensivePredictionResponse(
            p_start=predictions.get('p_start', 0.0),
            expected_minutes=predictions.get('xmins', 0.0),
            xg=predictions.get('xg', 0.0),
            xa=predictions.get('xa', 0.0),
            xcs=predictions.get('xcs', 0.0),
            momentum=momentum,
            trend=trend
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Team Optimization Solver Endpoints

@app.post("/api/solver/optimize-team", response_model=TeamOptimizationResponse)
async def optimize_team(request: TeamOptimizationRequest):
    """
    Optimize FPL team selection using Multi-Period ILP.
    Maximizes expected points while minimizing transfer costs.
    
    Constraints:
    - Budget: 100M
    - Squad: 2GK, 5DEF, 5MID, 3FWD
    - Max 3 players per team
    - Transfer penalty: -4 points per extra transfer
    """
    try:
        # Convert Pydantic models to dicts for solver
        players_data = []
        for player in request.players:
            player_dict = {
                'id': player.id,
                'name': player.name,
                'position': player.position,
                'price': player.price,
                'team_id': player.team_id,
                'team_name': player.team_name,
                'expected_points_gw1': player.expected_points_gw1
            }
            
            # Add expected points for other weeks if provided
            for week in range(2, request.horizon_weeks + 1):
                attr_name = f'expected_points_gw{week}'
                if hasattr(player, attr_name):
                    value = getattr(player, attr_name)
                    if value is not None:
                        player_dict[attr_name] = value
                    else:
                        # Use GW1 as fallback
                        player_dict[attr_name] = player.expected_points_gw1
                else:
                    player_dict[attr_name] = player.expected_points_gw1
            
            players_data.append(player_dict)
        
        # Create solver with custom parameters
        solver = TeamSolver(
            budget=request.budget,
            horizon_weeks=request.horizon_weeks,
            free_transfers=request.free_transfers
        )
        
        # Solve
        solution = solver.solve(
            players=players_data,
            current_squad=request.current_squad,
            locked_players=request.locked_players,
            excluded_players=request.excluded_players
        )
        
        # Convert to response format
        return TeamOptimizationResponse(**solution)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")


# New Team Optimization Endpoints (Task 10.1)

@app.post("/team/optimize", response_model=TeamOptimizationResponse)
async def optimize_team_endpoint(request: TeamOptimizationRequest):
    """
    Optimize FPL team selection for a single gameweek using ILP.
    This endpoint provides a simplified single-gameweek optimization.
    
    Args:
        request: Team optimization request with players and constraints
    
    Returns:
        Optimized team with squad, starting XI, and expected points
    """
    try:
        # Convert Pydantic models to dicts for solver
        players_data = []
        for player in request.players:
            player_dict = {
                'id': player.id,
                'name': player.name,
                'position': player.position,
                'price': player.price,
                'team_id': player.team_id,
                'team_name': player.team_name,
                'expected_points_gw1': player.expected_points_gw1
            }
            # For single gameweek, use GW1 points for all weeks
            for week in range(2, request.horizon_weeks + 1):
                attr_name = f'expected_points_gw{week}'
                if hasattr(player, attr_name) and getattr(player, attr_name) is not None:
                    player_dict[attr_name] = getattr(player, attr_name)
                else:
                    player_dict[attr_name] = player.expected_points_gw1
            players_data.append(player_dict)
        
        # Create solver with custom parameters
        solver = TeamSolver(
            budget=request.budget,
            horizon_weeks=1,  # Single gameweek optimization
            free_transfers=request.free_transfers
        )
        
        # Solve
        solution = solver.solve(
            players=players_data,
            current_squad=request.current_squad,
            locked_players=request.locked_players,
            excluded_players=request.excluded_players
        )
        
        # Convert to response format
        return TeamOptimizationResponse(**solution)
        
    except Exception as e:
        logger.error(f"Error in team optimization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")


@app.post("/team/plan", response_model=TeamPlanResponse)
async def plan_team_multi_period(request: TeamPlanRequest):
    """
    Generate multi-period transfer strategy (3-5 week horizon).
    Optimizes team selection across multiple gameweeks with transfer planning.
    
    Args:
        request: Team planning request with players and multi-week constraints
    
    Returns:
        Multi-period plan with squads, starting XIs, and transfer strategy for each week
    """
    try:
        # Convert Pydantic models to dicts for solver
        players_data = []
        for player in request.players:
            player_dict = {
                'id': player.id,
                'name': player.name,
                'position': player.position,
                'price': player.price,
                'team_id': player.team_id,
                'team_name': player.team_name,
                'expected_points_gw1': player.expected_points_gw1
            }
            # Add expected points for all weeks in horizon
            for week in range(2, request.horizon_weeks + 1):
                attr_name = f'expected_points_gw{week}'
                if hasattr(player, attr_name) and getattr(player, attr_name) is not None:
                    player_dict[attr_name] = getattr(player, attr_name)
                else:
                    # Fallback to GW1 if not provided
                    player_dict[attr_name] = player.expected_points_gw1
            players_data.append(player_dict)
        
        # Create solver with multi-period parameters
        solver = TeamSolver(
            budget=request.budget,
            horizon_weeks=request.horizon_weeks,
            free_transfers=request.free_transfers
        )
        
        # Solve multi-period optimization
        solution = solver.solve(
            players=players_data,
            current_squad=request.current_squad,
            locked_players=request.locked_players,
            excluded_players=request.excluded_players
        )
        
        # Convert transfers to TransferStrategy format
        transfer_strategy = []
        for week, transfer_info in solution.get('transfers', {}).items():
            transfer_strategy.append(TransferStrategy(
                gameweek=week,
                transfers_in=transfer_info.get('in', []),
                transfers_out=transfer_info.get('out', []),
                transfer_count=transfer_info.get('count', 0),
                transfer_cost=transfer_info.get('cost', 0.0),
                expected_points_gain=solution.get('points_breakdown', {}).get(week, {}).get('expected_points', 0.0)
            ))
        
        # Calculate net expected points (total points - transfer costs)
        total_transfer_cost = sum(ts.transfer_cost for ts in transfer_strategy)
        net_expected_points = solution.get('total_points', 0.0) - total_transfer_cost
        
        return TeamPlanResponse(
            status=solution.get('status', 'Unknown'),
            optimal=solution.get('optimal', False),
            horizon_weeks=request.horizon_weeks,
            squads=solution.get('squads', {}),
            starting_xis=solution.get('starting_xis', {}),
            transfer_strategy=transfer_strategy,
            total_expected_points=solution.get('total_points', 0.0),
            total_transfer_cost=total_transfer_cost,
            net_expected_points=net_expected_points,
            budget_used=solution.get('budget_used', {})
        )
        
    except Exception as e:
        logger.error(f"Error in team planning: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Planning error: {str(e)}")


@app.get("/market/intelligence", response_model=MarketIntelligenceResponse)
async def get_market_intelligence(
    gameweek: Optional[int] = None,
    season: str = "2025-26",
    db: Session = Depends(get_db)
):
    """
    Get market intelligence with ownership arbitrage analysis.
    Identifies differentials (high xP, low ownership) and overvalued players (low xP, high ownership).
    
    Args:
        gameweek: Gameweek number (default: current gameweek)
        season: Season string (default: "2025-26")
        db: Database session
    
    Returns:
        Market intelligence with player rankings, arbitrage scores, and categories
    """
    try:
        # Get gameweek if not provided
        if gameweek is None:
            gameweek = await _get_current_gameweek()
        
        # Calculate player ranks
        df = market_intelligence_service.calculate_player_ranks(
            db=db,
            gameweek=gameweek,
            season=season,
            use_fpl_api_ownership=True
        )
        
        if df.empty:
            logger.warning(f"No player data found for gameweek {gameweek}")
            return MarketIntelligenceResponse(
                gameweek=gameweek,
                season=season,
                players=[],
                total_players=0,
                differentials_count=0,
                overvalued_count=0,
                neutral_count=0
            )
        
        # Calculate arbitrage scores and categories
        df = market_intelligence_service.calculate_arbitrage_scores_and_categories(df)
        
        # Get player details from database (Player.id is the FPL ID)
        player_ids = df['player_id'].tolist()
        players_db = db.query(Player).filter(Player.id.in_(player_ids)).options(
            # Eager load team relationship
            joinedload(Player.team)
        ).all()
        player_map = {p.id: p for p in players_db}
        
        # Build response
        market_players = []
        for _, row in df.iterrows():
            player_db = player_map.get(row['player_id'])
            if not player_db:
                continue
            
            # Get team name from relationship or use 'Unknown'
            team_name = player_db.team.name if player_db.team else 'Unknown'
            
            market_players.append(MarketIntelligencePlayer(
                player_id=row['player_id'],
                name=row['name'],
                position=player_db.position,
                team=team_name,
                price=float(player_db.price) if player_db.price else 0.0,
                xp=float(row['xp']),
                ownership=float(row['ownership']),
                xp_rank=int(row['xp_rank']),
                ownership_rank=int(row['ownership_rank']),
                arbitrage_score=float(row['arbitrage_score']),
                category=row['category']
            ))
        
        # Count categories
        differentials_count = sum(1 for p in market_players if p.category == 'Differential')
        overvalued_count = sum(1 for p in market_players if p.category == 'Overvalued')
        neutral_count = sum(1 for p in market_players if p.category == 'Neutral')
        
        return MarketIntelligenceResponse(
            gameweek=gameweek,
            season=season,
            players=market_players,
            total_players=len(market_players),
            differentials_count=differentials_count,
            overvalued_count=overvalued_count,
            neutral_count=neutral_count
        )
        
    except Exception as e:
        logger.error(f"Error in market intelligence: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Market intelligence error: {str(e)}")


# Risk Management Endpoints

@app.post("/api/risk/ownership-arbitrage", response_model=OwnershipArbitrageResponse)
async def analyze_ownership_arbitrage(request: OwnershipArbitrageRequest):
    """
    Analyze ownership arbitrage opportunities.
    Finds over-owned players (high ownership, low xP) and suggests differential alternatives.
    """
    try:
        # Convert to dict format
        players_data = [
            {
                'id': p.id,
                'name': p.name,
                'position': p.position,
                'price': p.price,
                'team_id': getattr(p, 'team_id', 0),
                'team_name': getattr(p, 'team_name', ''),
                'ownership_percent': getattr(p, 'ownership_percent', 0.0),
                'expected_points_gw1': p.expected_points_gw1,
                'expected_points_gw2': getattr(p, 'expected_points_gw2', None),
                'expected_points_gw3': getattr(p, 'expected_points_gw3', None),
            }
            for p in request.players
        ]
        
        analysis = risk_service.arbitrage.analyze_arbitrage_opportunities(
            players_data, request.gameweek
        )
        
        return OwnershipArbitrageResponse(**analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/captain-selection", response_model=CaptainSelectionResponse)
async def select_captain(request: CaptainSelectionRequest):
    """
    Select optimal C/VC pair using advanced algorithm.
    Considers probability of captain not playing and weights vice-captain accordingly.
    """
    try:
        # Convert to dict format
        candidates_data = [
            {
                'id': p.id,
                'name': p.name,
                'position': p.position,
                'price': p.price,
                'expected_points_gw1': p.expected_points_gw1,
                'expected_points_gw2': getattr(p, 'expected_points_gw2', None),
                'expected_minutes': getattr(p, 'expected_minutes', 90.0),
                'status': getattr(p, 'status', 'a'),
                'rotation_risk': getattr(p, 'rotation_risk', 0.0),
                'recent_minutes': getattr(p, 'recent_minutes', [90])
            }
            for p in request.candidates
        ]
        
        result = risk_service.cvc_selector.select_optimal_captain_pair(
            candidates_data, request.gameweek
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Convert back to Pydantic models
        captain_pydantic = next(
            p for p in request.candidates if p.id == result['captain']['id']
        )
        vc_pydantic = next(
            p for p in request.candidates if p.id == result['vice_captain']['id']
        )
        
        return CaptainSelectionResponse(
            captain=captain_pydantic,
            vice_captain=vc_pydantic,
            analysis=result['analysis']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/chip-analysis", response_model=ChipAnalysisResponse)
async def analyze_chips(request: ChipAnalysisRequest):
    """
    Analyze optimal timing for FPL chips (Wildcard, Bench Boost, Free Hit).
    Determines if chips should be played based on expected point gains.
    """
    try:
        # Convert to dict format
        current_squad_data = [
            {
                'id': p.id,
                'expected_points_gw1': p.expected_points_gw1,
                'expected_points_gw2': getattr(p, 'expected_points_gw2', None),
                'expected_points_gw3': getattr(p, 'expected_points_gw3', None),
            }
            for p in request.current_squad
        ]
        
        optimized_squad_data = None
        if request.optimized_squad:
            optimized_squad_data = [
                {
                    'id': p.id,
                    'expected_points_gw1': p.expected_points_gw1,
                    'expected_points_gw2': getattr(p, 'expected_points_gw2', None),
                    'expected_points_gw3': getattr(p, 'expected_points_gw3', None),
                }
                for p in request.optimized_squad
            ]
        
        free_hit_squad_data = None
        if request.free_hit_squad:
            free_hit_squad_data = [
                {
                    'id': p.id,
                    'expected_points_gw1': p.expected_points_gw1,
                }
                for p in request.free_hit_squad
            ]
        
        analysis = risk_service.chip_timing.analyze_all_chips(
            current_squad_data,
            optimized_squad_data,
            free_hit_squad_data,
            request.gameweek
        )
        
        return ChipAnalysisResponse(**analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/comprehensive", response_model=ComprehensiveRiskAnalysisResponse)
async def comprehensive_risk_analysis(request: ComprehensiveRiskAnalysisRequest):
    """
    Get comprehensive risk management analysis including:
    - Ownership arbitrage opportunities
    - Chip timing recommendations
    """
    try:
        # Convert to dict format
        players_data = [
            {
                'id': p.id,
                'name': p.name,
                'position': p.position,
                'price': p.price,
                'ownership_percent': getattr(p, 'ownership_percent', 0.0),
                'expected_points_gw1': p.expected_points_gw1,
            }
            for p in request.players
        ]
        
        current_squad_data = [
            {
                'id': p.id,
                'expected_points_gw1': p.expected_points_gw1,
            }
            for p in request.current_squad
        ]
        
        optimized_squad_data = None
        if request.optimized_squad:
            optimized_squad_data = [
                {
                    'id': p.id,
                    'expected_points_gw1': p.expected_points_gw1,
                }
                for p in request.optimized_squad
            ]
        
        analysis = risk_service.get_comprehensive_analysis(
            players_data,
            current_squad_data,
            request.gameweek,
            optimized_squad_data
        )
        
        return ComprehensiveRiskAnalysisResponse(**analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backtesting Endpoints

@app.post("/api/backtesting/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Run backtesting using expanding window or rolling window methodology.
    Tests model performance without look-ahead bias.
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.historical_data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No historical data provided")
        
        # Define prediction function wrapper
        def prediction_function(train_data, test_data):
            # This is a simplified version - in practice, you'd use the actual ML engine
            # For now, return mean of training data as prediction
            mean_points = train_data['points'].mean() if 'points' in train_data.columns else 0.0
            return pd.Series([mean_points] * len(test_data), index=test_data.index)
        
        # Run backtest
        if request.methodology == "expanding_window":
            result = backtesting_engine.expanding_window_backtest(
                df,
                prediction_function,
                min_train_weeks=request.min_train_weeks
            )
        else:
            window_size = request.window_size or request.min_train_weeks
            result = backtesting_engine.rolling_window_backtest(
                df,
                prediction_function,
                window_size=window_size
            )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return BacktestResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model Performance Endpoints

@app.get("/api/models/performance")
async def get_model_performance(
    season: str = "2025-26",
    db: Session = Depends(get_db)
):
    """
    Get model performance metrics including backtest summaries and results.
    """
    try:
        from app.models import BacktestSummary, BacktestResult, ModelPerformance
        
        # Get backtest summaries
        summaries = db.query(BacktestSummary).filter(
            BacktestSummary.season == season
        ).all()
        
        # Get backtest results (detailed per-gameweek)
        results = db.query(BacktestResult).filter(
            BacktestResult.season == season
        ).order_by(BacktestResult.gameweek).all()
        
        # Get model performance records
        model_perf = db.query(ModelPerformance).order_by(
            ModelPerformance.gameweek
        ).all()
        
        import math
        
        def safe_float(value):
            """Convert value to float, handling None, inf, and nan."""
            if value is None:
                return None
            try:
                f = float(value)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
            except (ValueError, TypeError):
                return None
        
        return {
            "season": season,
            "summaries": [
                {
                    "model_version": s.model_version,
                    "methodology": s.methodology,
                    "season": s.season,
                    "total_weeks_tested": s.total_weeks_tested,
                    "overall_rmse": safe_float(s.overall_rmse),
                    "overall_mae": safe_float(s.overall_mae),
                    "overall_spearman_corr": safe_float(s.overall_spearman_corr),
                    "r_squared": safe_float(s.r_squared),
                    "total_predictions": s.total_predictions,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None
                }
                for s in summaries
            ],
            "results": [
                {
                    "model_version": r.model_version,
                    "methodology": r.methodology,
                    "gameweek": r.gameweek,
                    "rmse": safe_float(r.rmse),
                    "mae": safe_float(r.mae),
                    "spearman_corr": safe_float(r.spearman_corr),
                    "n_predictions": r.n_predictions,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                }
                for r in results
            ],
            "model_performance": [
                {
                    "model_version": m.model_version,
                    "gameweek": m.gameweek,
                    "mae": safe_float(m.mae),
                    "rmse": safe_float(m.rmse),
                    "accuracy": safe_float(m.accuracy),
                    "created_at": m.created_at.isoformat() if m.created_at else None
                }
                for m in model_perf
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching model performance: {str(e)}")


# Frontend Integration Endpoints

@app.get("/api/players/all", response_model=List[PlayerDisplayData])
async def get_all_players(
    gameweek: Optional[int] = None, 
    db: Session = Depends(get_db), 
    limit: Optional[int] = None,
    use_next_gameweek: bool = False
):
    """
    Get all players with ML-powered predictions for All Players page.
    
    Architecture: Batch Prediction (Fast Database Read)
    - Reads pre-calculated predictions from Prediction table
    - No ML computation during API request (ultra-fast)
    - Falls back to basic data if predictions not available
    
    Args:
        gameweek: Gameweek number (default: next gameweek if use_next_gameweek=True, else current gameweek)
        limit: Optional limit on number of players to return
        use_next_gameweek: If True, uses next/upcoming gameweek (for Dashboard). Default: False
    """
    try:
        # ==========================================================================
        # STEP 0: GET GAMEWEEK IF NOT PROVIDED
        # ==========================================================================
        if gameweek is None:
            try:
                if use_next_gameweek:
                    # Dashboard mode: Use next/upcoming gameweek
                    gameweek = await _get_next_gameweek()
                    logger.info(f"[BATCH] Using next gameweek for Dashboard: {gameweek}")
                else:
                    # Default mode: Use current gameweek
                    gameweek = await _get_current_gameweek()
                    logger.info(f"[BATCH] Using current gameweek: {gameweek}")
            except Exception as gw_error:
                logger.error(f"[BATCH] Failed to get gameweek: {str(gw_error)}")
                gameweek = 1  # Fallback to gameweek 1
        
        # Ensure gameweek is valid
        if gameweek is None or gameweek < 1:
            gameweek = 1
            logger.warning(f"[BATCH] Invalid gameweek, using default: 1")
        
        # ==========================================================================
        # STEP 1: CHECK GLOBAL CACHE FIRST (Fastest path)
        # ==========================================================================
        try:
            cached_players = _get_cached_players(gameweek)
            if cached_players is not None:
                logger.info(f"[CACHE HIT] Returning {len(cached_players)} cached players for GW{gameweek}")
                if limit:
                    return cached_players[:limit]
                return cached_players
        except Exception as cache_error:
            logger.warning(f"[CACHE] Cache lookup failed: {str(cache_error)}, continuing...")
    
    except Exception as outer_error:
        logger.error(f"[BATCH] Outer error in get_all_players: {str(outer_error)}")
        # Fall through to try block below
    
    try:
        # ======================================================================
        # STEP 2: READ PREDICTIONS FROM DATABASE (Batch Prediction)
        # ======================================================================
        predictions = db.query(Prediction).filter(
            Prediction.gameweek == gameweek,
            Prediction.season == "2025-26"
        ).all()
        
        logger.info(f"[BATCH] Loaded {len(predictions)} predictions from database for GW{gameweek}")
        
        # Create prediction map by fpl_id (Prediction.fpl_id matches Player.id)
        prediction_map = {p.fpl_id: p for p in predictions}
        
        # ======================================================================
        # STEP 3: FETCH ALL PLAYERS AND JOIN WITH PREDICTIONS
        # OPTIMIZATION: Limit query if too many players to prevent timeout
        # ======================================================================
        # Use joinedload to eagerly load team relationship to avoid lazy loading issues
        # If no limit specified, default to 500 players max to prevent timeout
        max_players = limit if limit else 500
        players = db.query(Player).options(joinedload(Player.team)).limit(max_players).all()
        
        if not players:
            logger.warning("[BATCH] No players in database")
            return []
        
        logger.info(f"[BATCH] Processing {len(players)} players (limit: {max_players})")
        
        # ======================================================================
        # STEP 4: FETCH FPL BOOTSTRAP DATA ONCE (Ownership info)
        # OPTIMIZATION: Use asyncio.wait_for with timeout to prevent blocking
        # ======================================================================
        ownership_map = {}
        try:
            # Set a 5-second timeout for bootstrap data fetch
            # If it takes longer, skip ownership data (non-critical)
            bootstrap = await asyncio.wait_for(
                fpl_api.get_bootstrap_data(),
                timeout=5.0
            )
            elements = bootstrap.get('elements', [])
            for element in elements:
                fpl_id = element.get('id')
                if fpl_id:
                    ownership_map[fpl_id] = float(element.get('selected_by_percent', 0.0))
            logger.info(f"[FPL API] Loaded ownership data for {len(ownership_map)} players")
        except asyncio.TimeoutError:
            logger.warning("[FPL API] Bootstrap data fetch timed out (>5s), skipping ownership data")
        except Exception as e:
            logger.warning(f"[FPL API] Failed to fetch ownership data: {str(e)}")
        
        # ======================================================================
        # STEP 5: PRE-FETCH FORM DATA IN BATCH (OPTIMIZATION - prevents N+1 queries)
        # ======================================================================
        player_ids = [p.id for p in players]
        form_map = {}
        
        try:
            # Get latest stats for all players in one batch query
            # This replaces N individual queries with 1 batch query
            from sqlalchemy import func
            
            # Get max gameweek per player using subquery
            max_gw_subq = db.query(
                PlayerGameweekStats.fpl_id,
                func.max(PlayerGameweekStats.gameweek).label('max_gw')
            ).filter(
                PlayerGameweekStats.fpl_id.in_(player_ids),
                PlayerGameweekStats.season == "2025-26"
            ).group_by(PlayerGameweekStats.fpl_id).subquery()
            
            # Join to get actual stats for latest gameweek per player
            latest_stats = db.query(PlayerGameweekStats).join(
                max_gw_subq,
                (PlayerGameweekStats.fpl_id == max_gw_subq.c.fpl_id) &
                (PlayerGameweekStats.gameweek == max_gw_subq.c.max_gw)
            ).filter(
                PlayerGameweekStats.season == "2025-26"
            ).all()
            
            for stat in latest_stats:
                form_map[stat.fpl_id] = float(stat.total_points or 0) / 10.0
            
            logger.info(f"[FORM] Loaded form data for {len(form_map)} players in batch (replaced {len(player_ids)} individual queries)")
        except Exception as form_batch_error:
            logger.warning(f"[FORM] Batch form query failed: {str(form_batch_error)}, using 0.0 as default")
            form_map = {}
        
        # ======================================================================
        # STEP 6: BUILD RESPONSE FROM PREDICTIONS + PLAYER DATA
        # ======================================================================
        players_data = []
        missing_predictions = 0
        
        for player in players:
            try:
                # Get prediction from database (Player.id is the FPL ID)
                prediction = prediction_map.get(player.id)
                
                if prediction:
                    # Use pre-calculated prediction
                    xp = prediction.xp or 0.0
                    xg = prediction.xg or 0.0
                    xa = prediction.xa or 0.0
                    xmins = prediction.xmins or 0.0
                    xcs = prediction.xcs or 0.0
                    defcon_score = prediction.defcon_score or 0.0
                else:
                    # No prediction found - use fallback values
                    missing_predictions += 1
                    if player.position == 'FWD':
                        xp = 2.5
                    elif player.position == 'MID':
                        xp = 2.0
                    elif player.position == 'DEF':
                        xp = 1.5
                    else:  # GK
                        xp = 1.5
                    xg = 0.0
                    xa = 0.0
                    xmins = 45.0
                    xcs = 0.0
                    defcon_score = 0.0
                
                # Get ownership
                ownership = ownership_map.get(player.id, 0.0)
                
                # Get form from pre-fetched batch map (much faster than per-player query)
                form = form_map.get(player.id, 0.0)
                
                # Get team name from relationship (safe handling)
                try:
                    team_name = player.team.name if player.team and hasattr(player.team, 'name') else 'Unknown'
                except Exception:
                    team_name = 'Unknown'
                
                # Ensure all required fields are valid
                player_id = int(player.id) if player.id is not None else 0
                player_name = str(player.name) if player.name else 'Unknown Player'
                player_position = str(player.position) if player.position else 'MID'
                
                # Create player display data with safe type conversions
                players_data.append(PlayerDisplayData(
                    id=player_id,
                    fpl_id=player_id,  # Player.id is the FPL ID
                    name=player_name,
                    position=player_position,
                    team=str(team_name),
                    price=float(player.price) if player.price is not None else 0.0,
                    expected_points=float(round(xp, 2)),
                    ownership_percent=float(round(ownership, 1)),
                    form=float(round(form, 1)),
                    xg=float(round(xg, 3)) if xg is not None else 0.0,
                    xa=float(round(xa, 3)) if xa is not None else 0.0,
                    xmins=float(round(xmins, 1)) if xmins is not None else 0.0,
                    xcs=float(round(xcs, 3)) if xcs is not None else 0.0,
                    defcon_score=float(round(defcon_score, 2)) if defcon_score is not None else 0.0
                ))
                
            except Exception as player_error:
                logger.error(f"[SKIP] Player {player.id} ({player.name}): {str(player_error)}")
                continue
        
        # ======================================================================
        # STEP 7: UPDATE GLOBAL CACHE
        # ======================================================================
        try:
            _update_cache(gameweek, players_data)
        except Exception as cache_error:
            logger.warning(f"[CACHE] Failed to update cache: {str(cache_error)}")
        
        # Log summary
        logger.info(
            f"[BATCH COMPLETE] Loaded {len(players_data)} players from predictions, "
            f"Missing predictions: {missing_predictions}"
        )
        
        if limit:
            return players_data[:limit]
        return players_data
        
    except Exception as e:
        # ======================================================================
        # GRACEFUL DEGRADATION: Return basic data on catastrophic failure
        # ======================================================================
        logger.error(f"[BATCH ERROR] get_all_players failed: {str(e)}")
        import traceback
        logger.error(f"[BATCH ERROR] Traceback: {traceback.format_exc()}")
        
        # Try to return basic data instead of 500 error
        try:
            basic_data = await _get_basic_player_data(db, limit)
            if basic_data:
                logger.info(f"[FALLBACK] Returning {len(basic_data)} basic player records")
                return basic_data
        except Exception as fallback_error:
            logger.error(f"[FALLBACK FAILED] {str(fallback_error)}")
            import traceback
            logger.error(f"[FALLBACK ERROR] Traceback: {traceback.format_exc()}")
        
        # Last resort: return empty list (better than 500 error)
        logger.warning("[FALLBACK] Returning empty list as last resort")
        return []


async def _get_basic_player_data(db: Session, limit: Optional[int] = None) -> List[PlayerDisplayData]:
    """
    Fallback function: Return basic player data without ML predictions.
    Used when ML models are unavailable or during errors.
    """
    try:
        # Use joinedload to eagerly load team relationship
        query = db.query(Player).options(joinedload(Player.team))
        if limit:
            query = query.limit(limit)
        players = query.all()
        
        players_data = []
        for player in players:
            try:
                # Safe team name extraction
                try:
                    team_name = player.team.name if player.team and hasattr(player.team, 'name') else 'Unknown'
                except Exception:
                    team_name = 'Unknown'
                
                # Ensure all required fields are valid
                player_id = int(player.id) if player.id is not None else 0
                player_name = str(player.name) if player.name else 'Unknown Player'
                player_position = str(player.position) if player.position else 'MID'
                
                players_data.append(PlayerDisplayData(
                    id=player_id,
                    fpl_id=player_id,  # Player.id is the FPL ID
                    name=player_name,
                    position=player_position,
                    team=str(team_name),
                    price=float(player.price) if player.price is not None else 0.0,
                    expected_points=2.0,  # Default baseline xP
                    ownership_percent=0.0,
                    form=0.0,
                    xg=0.0,
                    xa=0.0,
                    xmins=45.0,
                    xcs=0.0,
                    defcon_score=0.0
                ))
            except Exception as player_error:
                logger.error(f"[BASIC] Failed to process player {getattr(player, 'id', 'unknown')}: {str(player_error)}")
                continue
        
        logger.info(f"[FALLBACK] Returned basic data for {len(players_data)} players")
        return players_data
    except Exception as e:
        logger.error(f"[FALLBACK ERROR] {str(e)}")
        return []


@app.get("/api/dream-team", response_model=DreamTeamResponse)
async def get_dream_team(gameweek: Optional[int] = None, db: Session = Depends(get_db)):
    """
    Get optimal Dream Team for a gameweek using TeamSolver optimization.
    
    Architecture: Cache-First with Graceful Degradation
    - OPTIMIZATION: Uses cached player predictions from get_all_players
    - Falls back to simple selection if solver fails
    - Never returns 500 error
    
    Args:
        gameweek: Gameweek number (default: current gameweek from FPL API)
    """
    # ==========================================================================
    # STEP 0: GET CURRENT GAMEWEEK IF NOT PROVIDED
    # ==========================================================================
    if gameweek is None:
        gameweek = await _get_current_gameweek()
        logger.info(f"[GAMEWEEK] Using current gameweek: {gameweek}")
    
    # ==========================================================================
    # STEP 1: CHECK DREAM TEAM CACHE
    # ==========================================================================
    cached_dream_team = _get_cached_dream_team(gameweek)
    if cached_dream_team is not None:
        logger.info(f"[CACHE HIT] Returning cached dream team for GW{gameweek}")
        return cached_dream_team
    
    try:
        # ======================================================================
        # STEP 2: USE CACHED PLAYER DATA (Don't recalculate ML predictions!)
        # ======================================================================
        cached_players = _get_cached_players(gameweek)
        
        if cached_players is None:
            # Cache is empty - trigger player data calculation first
            logger.info(f"[CACHE MISS] Triggering player data calculation for GW{gameweek}")
            cached_players = await get_all_players(gameweek=gameweek, db=db)
        
        if not cached_players:
            logger.warning("[ERROR] No player data available for dream team")
            return DreamTeamResponse(
                gameweek=gameweek,
                squad=[],
                starting_xi=[],
                total_expected_points=0.0,
                total_cost=0.0,
                formation="0-0-0"
            )
        
        # ======================================================================
        # STEP 3: FETCH TEAM ID MAPPING (for solver constraints)
        # ======================================================================
        team_id_map = {}
        try:
            bootstrap = await fpl_api.get_bootstrap_data()
            teams = bootstrap.get('teams', [])
            for team in teams:
                team_name = team.get('name', '')
                team_id = team.get('id', 0)
                if team_name and team_id:
                    team_id_map[team_name] = team_id
        except Exception as e:
            logger.warning(f"[FPL API] Failed to fetch team mapping: {str(e)}")
            # Create fallback team_id_map from cached players
            for i, p in enumerate(cached_players):
                if p.team and p.team not in team_id_map:
                    team_id_map[p.team] = i + 1
        
        # ======================================================================
        # STEP 4: CONVERT CACHED PLAYERS TO SOLVER FORMAT
        # ======================================================================
        players_data = []
        for player in cached_players:
            try:
                team_id = team_id_map.get(player.team, 0)
                players_data.append({
                    'id': player.fpl_id,
                    'name': player.name,
                    'position': player.position,
                    'price': player.price,
                    'team_id': team_id,
                    'team_name': player.team,
                    'expected_points_gw1': player.expected_points,
                    'expected_points_gw2': player.expected_points,
                    'expected_points_gw3': player.expected_points,
                    'p_start': 0.8 if player.xmins and player.xmins > 60 else 0.5
                })
            except Exception as e:
                logger.warning(f"[SKIP] Player {player.name}: {str(e)}")
                continue
        
        logger.info(f"[SOLVER] Prepared {len(players_data)} players for optimization")
        
        # ======================================================================
        # STEP 5: RUN TEAM SOLVER (with fallback)
        # ======================================================================
        dream_team_response = None
        
        try:
            solution = team_solver.solve(
                players=players_data,
                current_squad=None,
                locked_players=None,
                excluded_players=None
            )
            
            if solution.get('optimal', False):
                # Build response from solver solution
                # Solver returns integer keys (1, 2, 3) not 'week1', 'week2', etc.
                squad_week1 = solution.get('squads', {}).get(1, [])
                starting_xi_week1 = solution.get('starting_xis', {}).get(1, [])
                
                squad = []
                for player_id in squad_week1:
                    player_dict = next((p for p in players_data if p['id'] == player_id), None)
                    if player_dict:
                        squad.append(DreamTeamPlayer(
                            player_id=player_dict['id'],
                            name=player_dict['name'],
                            position=player_dict['position'],
                            team=player_dict.get('team_name', 'Unknown'),  # team should be string
                            expected_points=player_dict['expected_points_gw1'],
                            price=player_dict['price']
                        ))
                
                starting_xi = []
                for player_id in starting_xi_week1:
                    player_dict = next((p for p in players_data if p['id'] == player_id), None)
                    if player_dict:
                        starting_xi.append(DreamTeamPlayer(
                            player_id=player_dict['id'],
                            name=player_dict['name'],
                            position=player_dict['position'],
                            team=player_dict.get('team_name', 'Unknown'),  # team should be string
                            expected_points=player_dict['expected_points_gw1'],
                            price=player_dict['price']
                        ))
                
                dream_team_response = DreamTeamResponse(
                    gameweek=gameweek,
                    squad=squad,
                    starting_xi=starting_xi,
                    total_expected_points=solution.get('total_points', 0.0),
                    total_cost=sum(p.price for p in squad),
                    formation=_determine_formation(starting_xi)
                )
                
                logger.info(f"[SOLVER SUCCESS] Dream team optimized: {len(squad)} players")
            else:
                logger.warning(f"[SOLVER] Optimization not optimal: {solution.get('status', 'unknown')}")
                
        except Exception as solver_error:
            logger.error(f"[SOLVER ERROR] {str(solver_error)}")
        
        # ======================================================================
        # STEP 6: FALLBACK TO SIMPLE SELECTION IF SOLVER FAILED
        # ======================================================================
        if dream_team_response is None:
            logger.info("[FALLBACK] Using simple dream team selection")
            dream_team_response = await _get_fallback_dream_team(players_data, gameweek)
        
        # ======================================================================
        # STEP 7: UPDATE CACHE WITH DREAM TEAM
        # ======================================================================
        if dream_team_response and dream_team_response.squad:
            DATA_CACHE["dream_team_by_gw"][gameweek] = dream_team_response
            logger.info(f"[CACHE] Dream team cached for GW{gameweek}")
        
        return dream_team_response
        
    except Exception as e:
        logger.error(f"[CRITICAL] get_dream_team failed: {str(e)}")
        
        # Return empty dream team instead of 500 error
        return DreamTeamResponse(
            gameweek=gameweek,
            squad=[],
            starting_xi=[],
            total_expected_points=0.0,
            total_cost=0.0,
            formation="0-0-0"
        )


async def _get_fallback_dream_team(players_data: List[Dict], gameweek: int) -> DreamTeamResponse:
    """
    Fallback dream team selection if solver fails.
    Uses greedy selection by position with max 3 per team constraint.
    """
    try:
        # Sort by expected points
        sorted_players = sorted(
            players_data,
            key=lambda x: float(x.get('expected_points_gw1', 0) or 0),
            reverse=True
        )
        
        # Select by position with team limit
        squad = []
        positions = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        pos_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        team_counts = {}  # Track players per team
        
        for player in sorted_players:
            pos = player.get('position', 'MID')
            if pos not in positions:
                pos = 'MID'  # Default position
            
            team_id = player.get('team_id', 0)
            team_name = player.get('team_name', 'Unknown')
            
            # Check position limit
            if pos_counts.get(pos, 0) >= positions.get(pos, 0):
                continue
            
            # Check team limit (max 3 per team)
            if team_counts.get(team_id, 0) >= 3 and team_id != 0:
                continue
            
            # Validate required fields
            player_id = player.get('id')
            player_name = player.get('name', 'Unknown')
            player_price = float(player.get('price', 0) or 0)
            player_xp = float(player.get('expected_points_gw1', 0) or 0)
            
            if player_id is None:
                continue
            
            try:
                squad.append(DreamTeamPlayer(
                    player_id=int(player_id),
                    name=str(player_name),
                    position=str(pos),
                    team=str(team_name) if team_name else "Unknown",  # team should be string (team name)
                    expected_points=round(player_xp, 2),
                    price=round(player_price, 1)
                ))
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
            except Exception as e:
                logger.warning(f"[FALLBACK SKIP] Player {player_name}: {str(e)}")
                continue
            
            # Check if squad is complete (15 players)
            if len(squad) >= 15:
                break
        
        if not squad:
            logger.error("[FALLBACK] Could not build any squad!")
            return DreamTeamResponse(
                gameweek=gameweek,
                squad=[],
                starting_xi=[],
                total_expected_points=0.0,
                total_cost=0.0,
                formation="0-0-0"
            )
        
        # Select starting XI (best 11 with valid formation)
        # Sort squad by xP
        sorted_squad = sorted(squad, key=lambda x: x.expected_points, reverse=True)
        
        # Build valid starting XI
        starting_xi = []
        xi_pos_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        xi_pos_limits = {'GK': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}  # Max in starting XI
        xi_pos_mins = {'GK': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}     # Min in starting XI
        
        # First pass: ensure minimums
        for player in sorted_squad:
            pos = player.position
            if xi_pos_counts.get(pos, 0) < xi_pos_mins.get(pos, 0):
                starting_xi.append(player)
                xi_pos_counts[pos] = xi_pos_counts.get(pos, 0) + 1
        
        # Second pass: fill remaining slots with best players
        for player in sorted_squad:
            if len(starting_xi) >= 11:
                break
            if player in starting_xi:
                continue
            pos = player.position
            if xi_pos_counts.get(pos, 0) < xi_pos_limits.get(pos, 0):
                starting_xi.append(player)
                xi_pos_counts[pos] = xi_pos_counts.get(pos, 0) + 1
        
        formation = _determine_formation(starting_xi)
        total_xp = sum(p.expected_points for p in starting_xi)
        total_cost = sum(p.price for p in squad)
        
        logger.info(f"[FALLBACK SUCCESS] Squad: {len(squad)}, XI: {len(starting_xi)}, Formation: {formation}")
        
        return DreamTeamResponse(
            gameweek=gameweek,
            squad=squad,
            starting_xi=starting_xi,
            total_expected_points=round(total_xp, 2),
            total_cost=round(total_cost, 1),
            formation=formation
        )
        
    except Exception as e:
        logger.error(f"[FALLBACK CRITICAL] {str(e)}")
        return DreamTeamResponse(
            gameweek=gameweek,
            squad=[],
            starting_xi=[],
            total_expected_points=0.0,
            total_cost=0.0,
            formation="0-0-0"
        )


def _determine_formation(starting_xi: List[DreamTeamPlayer]) -> str:
    """Determine formation from starting XI"""
    pos_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    for player in starting_xi:
        pos = player.position
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    # Standard formation format: DEF-MID-FWD
    return f"{pos_counts.get('DEF', 0)}-{pos_counts.get('MID', 0)}-{pos_counts.get('FWD', 0)}"


# FPL API Raw Data Endpoints

@app.get("/api/fpl/bootstrap")
async def get_bootstrap():
    """
    Get bootstrap-static data: all players, teams, and gameweek information.
    """
    try:
        bootstrap = await fpl_api.get_bootstrap_data()
        return {
            'players': fpl_api.extract_players_from_bootstrap(bootstrap),
            'teams': fpl_api.extract_teams_from_bootstrap(bootstrap),
            'events': bootstrap.get('events', []),
            'element_types': bootstrap.get('element_types', []),
            'total_players': len(bootstrap.get('elements', []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fpl/player/{player_id}/history")
async def get_player_history(player_id: int):
    """
    Get detailed player history from element-summary endpoint.
    Includes match-by-match statistics: minutes, BPS, ICT Index, xG, xA, etc.
    """
    try:
        player_summary = await fpl_api.get_player_data(player_id)
        history = fpl_api.extract_player_history(player_summary)
        
        return {
            'player_id': player_id,
            'history': history,
            'history_past': player_summary.get('history_past', []),
            'fixtures': player_summary.get('fixtures', []),
            'total_matches': len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fpl/fixtures")
async def get_fixtures_data(
    gameweek: Optional[int] = None,
    include_difficulty: bool = True,
    future_only: bool = True
):
    """
    Get fixtures data with optional difficulty ratings.
    Dashboard-focused: Returns future fixtures by default (next gameweek, unfinished matches only).
    
    Args:
        gameweek: Optional gameweek filter. If None, uses next gameweek.
        include_difficulty: Calculate and include difficulty ratings
        future_only: If True, only return unfinished fixtures (default: True for Dashboard)
    """
    try:
        # Get next gameweek if not provided (for Dashboard)
        next_gameweek = None
        if gameweek is None:
            next_gameweek = await fpl_api.get_next_gameweek()
            if next_gameweek:
                gameweek = next_gameweek
                logger.info(f"[FIXTURES] Using next gameweek: {gameweek}")
        
        # Fetch fixtures (future_only=True by default for Dashboard)
        fixtures = await fpl_api.get_fixtures(gameweek=gameweek, future_only=future_only)
        
        if include_difficulty:
            bootstrap = await fpl_api.get_bootstrap_data()
            teams = fpl_api.extract_teams_from_bootstrap(bootstrap)
            fixtures = fpl_api.extract_fixtures_with_difficulty(fixtures, teams)
        
        # Ensure we have a gameweek ID for the response
        current_gameweek_id = gameweek
        if current_gameweek_id is None:
            current_gameweek_id = await fpl_api.get_next_gameweek()
        
        return {
            'fixtures': fixtures,
            'count': len(fixtures),
            'gameweek': gameweek,
            'current_gameweek_id': current_gameweek_id  # Next/upcoming gameweek for Dashboard
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fpl/fixtures/future")
async def get_future_fixtures(
    gameweeks: Optional[str] = None,  # Comma-separated list
    team_id: Optional[int] = None
):
    """
    Get future fixtures for specified gameweeks and/or team.
    
    Args:
        gameweeks: Comma-separated list of gameweeks (e.g., "1,2,3")
        team_id: Optional team ID to filter
    """
    try:
        gw_list = None
        if gameweeks:
            gw_list = [int(gw.strip()) for gw in gameweeks.split(',')]
        
        fixtures = await fpl_api.get_future_fixtures(gw_list, team_id)
        
        bootstrap = await fpl_api.get_bootstrap_data()
        teams = fpl_api.extract_teams_from_bootstrap(bootstrap)
        fixtures = fpl_api.extract_fixtures_with_difficulty(fixtures, teams)
        
        return {
            'fixtures': fixtures,
            'count': len(fixtures),
            'gameweeks': gw_list,
            'team_id': team_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fpl/players/bulk-history")
async def get_players_bulk_history(
    player_ids: Optional[str] = None,  # Comma-separated list
    max_players: Optional[int] = 50
):
    """
    Fetch multiple players with their detailed history statistics.
    Implements rate limiting to avoid API throttling.
    
    Args:
        player_ids: Comma-separated list of player IDs
        max_players: Maximum number of players to fetch (default: 50)
    """
    try:
        id_list = None
        if player_ids:
            id_list = [int(pid.strip()) for pid in player_ids.split(',')]
        
        players = await fpl_api.get_all_players_with_history(
            player_ids=id_list,
            max_players=max_players
        )
        
        return {
            'players': players,
            'count': len(players),
            'total_with_history': sum(1 for p in players if p.get('history'))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Third-Party Data Endpoints

@app.get("/api/third-party/understat/players")
async def get_understat_players(season: str = "2025", player_name: Optional[str] = None):
    """
    Get player statistics from Understat (xG, xA, NPxG).
    """
    try:
        understat_service = UnderstatService()
        players = await understat_service.get_player_stats(season, player_name)
        
        return {
            'season': season,
            'players': players,
            'count': len(players)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/third-party/fbref/defensive")
async def get_fbref_defensive_stats(season: str = "2025-2026", player_name: Optional[str] = None):
    """
    Get defensive statistics from FBref (blocks, interventions, passes).
    Critical for DefCon metrics in 2025/26 FPL rules.
    """
    try:
        fbref_service = FBrefService()
        stats = await fbref_service.get_player_defensive_stats(season, player_name)
        
        return {
            'season': season,
            'players': stats,
            'count': len(stats)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/third-party/enrich/{player_id}")
async def enrich_player_with_third_party(player_id: int, season: str = "2025"):
    """
    Enrich a single FPL player with Understat and FBref data.
    """
    try:
        # Get FPL player data
        bootstrap = await fpl_api.get_bootstrap_data()
        players = fpl_api.extract_players_from_bootstrap(bootstrap)
        
        fpl_player = next((p for p in players if p['id'] == player_id), None)
        if not fpl_player:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
        
        # Enrich with third-party data
        enriched = await third_party_service.enrich_player_data(fpl_player, season)
        
        return enriched
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/third-party/enrich/bulk")
async def enrich_players_bulk(
    player_ids: Optional[str] = None,  # Comma-separated list
    season: str = "2025",
    max_players: Optional[int] = 50
):
    """
    Enrich multiple players with third-party data.
    Implements rate limiting to avoid throttling.
    """
    try:
        # Get FPL players
        bootstrap = await fpl_api.get_bootstrap_data()
        all_players = fpl_api.extract_players_from_bootstrap(bootstrap)
        
        if player_ids:
            id_list = [int(pid.strip()) for pid in player_ids.split(',')]
            fpl_players = [p for p in all_players if p['id'] in id_list]
        else:
            fpl_players = all_players[:max_players] if max_players else all_players
        
        # Enrich with third-party data
        enriched = await third_party_service.enrich_players_bulk(
            fpl_players,
            season,
            max_players
        )
        
        return {
            'players': enriched,
            'count': len(enriched),
            'season': season
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/third-party/map-players")
async def map_players_across_sources(season: str = "2025", db: Session = Depends(get_db)):
    """
    Create mapping between FPL players and third-party data sources.
    Uses Entity Resolution Engine to match player names with high accuracy.
    """
    try:
        # Get FPL players
        bootstrap = await fpl_api.get_bootstrap_data()
        fpl_players = fpl_api.extract_players_from_bootstrap(bootstrap)
        
        # Initialize services with Entity Resolution
        understat_service = UnderstatService(
            entity_resolution_service=entity_resolution,
            db_session=db
        )
        fbref_service = FBrefService(
            entity_resolution_service=entity_resolution,
            db_session=db
        )
        
        # Get Understat data
        understat_data = await understat_service.get_player_stats(season)
        
        # Get FBref data
        fbref_data = await fbref_service.get_player_defensive_stats(season)
        
        # Create mappings using Entity Resolution (async)
        understat_mapping = await understat_service.map_to_fpl_players(
            understat_data, 
            fpl_players,
            use_entity_resolution=True
        )
        fbref_mapping = await fbref_service.map_to_fpl_players(
            fbref_data, 
            fpl_players,
            use_entity_resolution=True
        )
        
        # Calculate match statistics
        understat_high_confidence = sum(
            1 for m in understat_mapping.values() 
            if m.get('confidence', 0.0) >= 0.85
        )
        fbref_high_confidence = sum(
            1 for m in fbref_mapping.values() 
            if m.get('confidence', 0.0) >= 0.85
        )
        
        return {
            'understat_mapping': {
                'total_fpl_players': len(fpl_players),
                'matched': len(understat_mapping),
                'high_confidence_matches': understat_high_confidence,
                'match_methods': {
                    'entity_resolution': sum(1 for m in understat_mapping.values() if m.get('match_method') == 'entity_resolution'),
                    'fuzzy_fallback': sum(1 for m in understat_mapping.values() if m.get('match_method') == 'fuzzy_fallback'),
                    'simple_name_match': sum(1 for m in understat_mapping.values() if m.get('match_method') == 'simple_name_match'),
                },
                'mapping': understat_mapping
            },
            'fbref_mapping': {
                'total_fpl_players': len(fpl_players),
                'matched': len(fbref_mapping),
                'high_confidence_matches': fbref_high_confidence,
                'match_methods': {
                    'entity_resolution': sum(1 for m in fbref_mapping.values() if m.get('match_method') == 'entity_resolution'),
                    'fuzzy_fallback': sum(1 for m in fbref_mapping.values() if m.get('match_method') == 'fuzzy_fallback'),
                    'simple_name_match': sum(1 for m in fbref_mapping.values() if m.get('match_method') == 'simple_name_match'),
                },
                'mapping': fbref_mapping
            }
        }
        
    except Exception as e:
        logger.error(f"Error mapping players: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Entity Resolution Endpoints

@app.on_event("startup")
async def startup_event():
    """Load Master ID Map, ML models, and start scheduler on startup"""
    # Reduce log noise from TensorFlow and other libraries
    import os
    import warnings
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Only show errors
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*protected namespace.*')
    
    try:
        await entity_resolution.load_master_map()
        logger.info("Master ID Map loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Master ID Map on startup: {str(e)}")
    
    # CRITICAL: Load ML models into memory
    try:
        ml_engine._ensure_models_loaded()
        logger.info("ML Engine models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ML Engine models on startup: {str(e)}")
    
    # BATCH PREDICTION: Update predictions in background (non-blocking)
    try:
        from app.scripts.update_predictions import update_predictions_for_current_gameweek
        
        async def update_predictions_background():
            """Background task to update predictions"""
            try:
                # Wait a bit for app to fully start
                await asyncio.sleep(5)
                logger.info("[BATCH PREDICTION] Starting background prediction update")
                await update_predictions_for_current_gameweek()
                logger.info("[BATCH PREDICTION] Background prediction update completed")
            except Exception as e:
                logger.error(f"[BATCH PREDICTION] Background update failed: {str(e)}")
        
        # Run in background (don't block startup)
        asyncio.create_task(update_predictions_background())
        logger.info("[BATCH PREDICTION] Background prediction update task started")
    except Exception as e:
        logger.warning(f"Failed to start prediction update task: {str(e)}")
    
    # Start daily ETL scheduler (runs at 2 AM daily)
    try:
        scheduler.add_job(
            daily_etl_refresh,
            CronTrigger(hour=2, minute=0),
            id='daily_etl_refresh',
            name='Daily ETL Refresh',
            replace_existing=True
        )
        # Add prediction update job (runs at 2:30 AM daily, after ETL)
        scheduler.add_job(
            update_predictions_daily,
            CronTrigger(hour=2, minute=30),
            id='daily_prediction_update',
            name='Daily Prediction Update',
            replace_existing=True
        )
        scheduler.start()
        logger.info("Daily ETL and prediction update schedulers started")
    except Exception as e:
        logger.warning(f"Failed to start schedulers: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown scheduler"""
    try:
        scheduler.shutdown()
        await etl_service.close()
        logger.info("Scheduler and ETL service shut down")
    except Exception as e:
        logger.warning(f"Error during shutdown: {str(e)}")


async def daily_etl_refresh():
    """
    Daily ETL refresh task.
    Fetches latest data from FPL API and loads into PostgreSQL.
    """
    try:
        logger.info("Starting daily ETL refresh...")
        result = await etl_service.sync_from_fpl_api(
            fpl_api,
            gameweek=None,  # Sync all gameweeks
            season="2025-26"
        )
        logger.info(f"Daily ETL refresh completed: {result}")
    except Exception as e:
        logger.error(f"Error in daily ETL refresh: {str(e)}")


async def update_predictions_daily():
    """
    Daily prediction update task.
    Calculates and stores ML predictions for all players.
    Runs after ETL refresh to use latest data.
    """
    try:
        from app.scripts.update_predictions import update_predictions_for_current_gameweek
        
        logger.info("[BATCH PREDICTION] Starting daily prediction update...")
        await update_predictions_for_current_gameweek()
        logger.info("[BATCH PREDICTION] Daily prediction update completed")
    except Exception as e:
        logger.error(f"[BATCH PREDICTION] Error in daily prediction update: {str(e)}")


@app.get("/api/entity-resolution/load-map")
async def load_master_map(force_reload: bool = False):
    """
    Load or reload Master ID Map from GitHub.
    """
    try:
        success = await entity_resolution.load_master_map(force_reload=force_reload)
        if success:
            return {
                'status': 'success',
                'message': f'Master ID Map loaded: {len(entity_resolution.master_map) if entity_resolution.master_map is not None else 0} mappings'
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load Master ID Map")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/entity-resolution/resolve", response_model=PlayerResolutionResponse)
async def resolve_player_entity(request: PlayerResolutionRequest):
    """
    Resolve a single player entity across all data sources.
    Uses Master ID Map and fuzzy matching.
    """
    try:
        resolution = entity_resolution.resolve_player_entity(
            fpl_id=request.fpl_id,
            fpl_name=request.fpl_name,
            fpl_team=request.fpl_team,
            understat_name=request.understat_name,
            fbref_name=request.fbref_name
        )
        
        return PlayerResolutionResponse(**resolution)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/entity-resolution/resolve-bulk", response_model=BulkResolutionResponse)
async def resolve_players_bulk(request: BulkResolutionRequest):
    """
    Resolve multiple players at once.
    """
    try:
        # Convert to list of dicts
        players_data = [
            {
                'id': p.fpl_id,
                'fpl_id': p.fpl_id,
                'fpl_name': p.fpl_name,
                'team_name': p.fpl_team,
                'understat_name': p.understat_name,
                'fbref_name': p.fbref_name
            }
            for p in request.players
        ]
        
        resolved = entity_resolution.resolve_players_bulk(
            players_data,
            include_fuzzy=request.include_fuzzy
        )
        
        # Convert to response format
        resolved_responses = {
            fpl_id: PlayerResolutionResponse(**data)
            for fpl_id, data in resolved.items()
        }
        
        matched_count = sum(1 for r in resolved_responses.values() if r.matched)
        unmatched = [r for r in resolved_responses.values() if not r.matched]
        
        return BulkResolutionResponse(
            resolved=resolved_responses,
            total_players=len(resolved),
            matched_count=matched_count,
            unmatched_count=len(unmatched),
            unmatched_players=unmatched
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/entity-resolution/unmatched")
async def get_unmatched_players():
    """
    Get list of unmatched players for manual review.
    """
    try:
        unmatched = entity_resolution.get_unmatched_players()
        return {
            'unmatched_players': unmatched,
            'count': len(unmatched)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/entity-resolution/manual-mapping")
async def add_manual_mapping(request: ManualMappingRequest):
    """
    Add manual mapping for unmatched players.
    Updates the Master ID Map with custom mappings.
    """
    try:
        entity_resolution.add_manual_mapping(
            fpl_id=request.fpl_id,
            fpl_name=request.fpl_name,
            understat_id=request.understat_id,
            fbref_id=request.fbref_id,
            understat_name=request.understat_name,
            fbref_name=request.fbref_name
        )
        
        return {
            'status': 'success',
            'message': f'Manual mapping added for {request.fpl_name} (FPL ID: {request.fpl_id})'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/entity-resolution/override")
async def override_mapping(
    request: OverrideMappingRequest,
    db: Session = Depends(get_db)
):
    """
    Manually override/correct a low-confidence entity mapping (score < 0.85).
    
    This endpoint allows manual correction of mappings by:
    - Updating the mapping with corrected names
    - Marking as manually_verified=True
    - Setting confidence_score to 1.0
    - Validating to prevent duplicate mappings
    
    Use this for low-confidence matches that need manual review and correction.
    """
    try:
        mapping = entity_resolution.override_mapping(
            db=db,
            fpl_id=request.fpl_id,
            understat_name=request.understat_name,
            fbref_name=request.fbref_name,
            fpl_name=request.fpl_name,
            canonical_name=request.canonical_name
        )
        
        return {
            'status': 'success',
            'message': f'Mapping overridden for FPL ID {request.fpl_id}',
            'mapping': {
                'fpl_id': mapping.fpl_id,
                'canonical_name': mapping.canonical_name,
                'understat_name': mapping.understat_name,
                'fbref_name': mapping.fbref_name,
                'confidence_score': float(mapping.confidence_score) if mapping.confidence_score else None,
                'manually_verified': mapping.manually_verified,
                'updated_at': mapping.updated_at.isoformat() if mapping.updated_at else None
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"Database constraint violation: {str(e)}")
    except Exception as e:
        logger.error(f"Error overriding mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/entity-resolution/fuzzy-match")
async def fuzzy_match_player(
    player_name: str,
    threshold: float = 0.8
):
    """
    Find fuzzy matches for a player name.
    Useful for identifying potential matches for unmatched players.
    """
    try:
        if entity_resolution.master_map is None:
            await entity_resolution.load_master_map()
        
        # Get all names from master map for fuzzy matching
        all_names = []
        if entity_resolution.master_map is not None:
            if 'FPL_Name' in entity_resolution.master_map.columns:
                all_names.extend(entity_resolution.master_map['FPL_Name'].dropna().tolist())
            if 'Understat_Name' in entity_resolution.master_map.columns:
                all_names.extend(entity_resolution.master_map['Understat_Name'].dropna().tolist())
        
        matches = entity_resolution.fuzzy_match(player_name, all_names, threshold)
        
        return {
            'player_name': player_name,
            'threshold': threshold,
            'matches': [{'name': name, 'similarity': score} for name, score in matches[:10]]  # Top 10
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/entity-resolution/resolve-all", response_model=BulkResolutionReport)
async def resolve_all_players(
    store_mappings: bool = True,
    db: Session = Depends(get_db)
):
    """
    Resolve all FPL players and store mappings in database.
    Generates a comprehensive report of match accuracy, low-confidence matches, and unmatched players.
    
    This endpoint:
    - Fetches all FPL players from bootstrap data
    - Resolves entity mappings for each player
    - Stores mappings in database (if store_mappings=True)
    - Returns detailed report with statistics
    
    Use this for bulk processing and generating reports for manual review.
    """
    try:
        # Load master map if not already loaded
        if entity_resolution.master_map is None:
            await entity_resolution.load_master_map()
        
        # Fetch all FPL players from bootstrap
        bootstrap = await fpl_api.get_bootstrap_data()
        fpl_players = fpl_api.extract_players_from_bootstrap(bootstrap)
        
        logger.info(f"Resolving {len(fpl_players)} FPL players...")
        
        # Resolve all players and generate report
        report = entity_resolution.resolve_all_players(
            db=db,
            fpl_players=fpl_players,
            store_mappings=store_mappings
        )
        
        return BulkResolutionReport(**report)
        
    except Exception as e:
        logger.error(f"Error in bulk resolution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Cleaning Endpoints

@app.post("/api/data-cleaning/clean", response_model=DataCleaningResponse)
async def clean_player_data(request: DataCleaningRequest):
    """
    Clean and normalize a single player's data.
    Includes DGW normalization, DefCon calculation, and type conversion.
    """
    try:
        cleaned = data_cleaning.clean_player_data(
            request.player_data,
            normalize_dgw=request.normalize_dgw,
            calculate_defcon=request.calculate_defcon,
            convert_types=request.convert_types,
            position=request.position
        )
        
        # Track type conversions
        type_conversions = {}
        if request.convert_types:
            for key, value in cleaned.items():
                if key in request.player_data:
                    original_type = type(request.player_data[key]).__name__
                    new_type = type(value).__name__
                    if original_type != new_type:
                        type_conversions[key] = f"{original_type} -> {new_type}"
        
        return DataCleaningResponse(
            cleaned_data=cleaned,
            normalized_points=cleaned.get('normalized_points'),
            defcon_floor_points=cleaned.get('defcon_floor_points'),
            type_conversions=type_conversions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-cleaning/clean-bulk", response_model=BulkCleaningResponse)
async def clean_players_bulk(request: BulkCleaningRequest):
    """
    Clean and normalize multiple players' data at once.
    """
    try:
        cleaned_players = data_cleaning.clean_bulk_player_data(
            request.players_data,
            normalize_dgw=request.normalize_dgw,
            calculate_defcon=request.calculate_defcon,
            convert_types=request.convert_types
        )
        
        normalized_count = sum(1 for p in cleaned_players if 'normalized_points' in p)
        defcon_count = sum(1 for p in cleaned_players if 'defcon_floor_points' in p)
        
        return BulkCleaningResponse(
            cleaned_players=cleaned_players,
            total_players=len(cleaned_players),
            normalized_count=normalized_count,
            defcon_calculated_count=defcon_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-cleaning/normalize-dgw")
async def normalize_dgw_points(
    points: float,
    matches_played: int,
    gameweek_type: str = "normal"
):
    """
    Normalize points for Double Gameweeks (DGW) or Blank Gameweeks (BGW).
    Prevents overfitting by dividing DGW points by match count.
    """
    try:
        normalized = data_cleaning.normalize_dgw_points(
            points,
            matches_played,
            gameweek_type
        )
        
        return {
            'original_points': points,
            'matches_played': matches_played,
            'normalized_points': normalized,
            'gameweek_type': gameweek_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-cleaning/defcon", response_model=DefConMetricsResponse)
async def calculate_defcon_metrics(
    player_data: Dict,
    position: str,
    minutes: int = 90
):
    """
    Calculate DefCon floor points for a player based on 2025/26 FPL rules.
    Includes blocks, interventions, and pass bonus calculations.
    """
    try:
        metrics = data_cleaning.get_defcon_metrics(player_data, position)
        
        return DefConMetricsResponse(**metrics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-cleaning/convert-types")
async def convert_metrics_to_float(
    data: Dict,
    metric_columns: Optional[List[str]] = None
):
    """
    Convert metric columns to float type to prevent calculation errors.
    Useful for ICT Index, xG, xA, and other numeric metrics.
    """
    try:
        converted = data_cleaning.convert_metrics_to_float(data, metric_columns)
        
        # Track conversions
        conversions = {}
        for key, value in converted.items():
            if key in data:
                original_type = type(data[key]).__name__
                new_type = type(value).__name__
                if original_type != new_type:
                    conversions[key] = f"{original_type} -> {new_type}"
        
        return {
            'converted_data': converted,
            'conversions': conversions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ETL Endpoints

@app.post("/api/etl/sync")
async def trigger_etl_sync(
    gameweek: Optional[int] = None,
    season: str = "2025-26",
    background_tasks: BackgroundTasks = None
):
    """
    Trigger ETL sync process.
    Fetches data from FPL API and loads into PostgreSQL.
    Can run in background or synchronously.
    """
    try:
        if background_tasks:
            # Run in background
            background_tasks.add_task(
                etl_service.sync_from_fpl_api,
                fpl_api,
                gameweek,
                season
            )
            return {
                'status': 'started',
                'message': 'ETL sync started in background',
                'gameweek': gameweek,
                'season': season
            }
        else:
            # Run synchronously
            result = await etl_service.sync_from_fpl_api(
                fpl_api,
                gameweek,
                season
            )
            return {
                'status': 'completed',
                'result': result
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/etl/upsert-player")
async def upsert_player(player_data: Dict):
    """
    UPSERT a single player record.
    """
    try:
        result = await etl_service.upsert_player(player_data)
        return {
            'status': 'success',
            'player': result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/etl/upsert-gameweek-stats")
async def upsert_gameweek_stats(stats_data: Dict):
    """
    UPSERT a single gameweek statistics record.
    """
    try:
        result = await etl_service.upsert_player_gameweek_stats(stats_data)
        return {
            'status': 'success',
            'stats': result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/etl/bulk-upsert-players")
async def bulk_upsert_players(
    players_data: List[Dict],
    batch_size: int = 100
):
    """
    Bulk UPSERT multiple players.
    """
    try:
        result = await etl_service.bulk_upsert_players(players_data, batch_size)
        return {
            'status': 'success',
            'result': result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/etl/bulk-upsert-stats")
async def bulk_upsert_stats(
    stats_data: List[Dict],
    batch_size: int = 100
):
    """
    Bulk UPSERT multiple gameweek statistics records.
    """
    try:
        result = await etl_service.bulk_upsert_gameweek_stats(stats_data, batch_size)
        return {
            'status': 'success',
            'result': result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/etl/status")
async def get_etl_status():
    """
    Get ETL service status and scheduler information.
    """
    try:
        jobs = []
        for job in scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            })
        
        return {
            'scheduler_running': scheduler.running,
            'scheduled_jobs': jobs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
