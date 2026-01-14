from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.routing import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session
import pandas as pd
from typing import List, Optional, Dict
from app.database import engine, Base, get_db
from app.services.feature_engineering import FeatureEngineeringService
from app.services.fpl_api import FPLAPIService
from app.schemas import (
    FormAlphaOptimizeRequest, FormAlphaResponse,
    FDRFitRequest, FDRResponse, DefConFeaturesResponse,
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
    ManualMappingRequest,
    DataCleaningRequest, DataCleaningResponse,
    BulkCleaningRequest, BulkCleaningResponse,
    DefConMetricsResponse
)
from app.services.predictive_engine import PredictiveEngine
from app.services.team_solver import TeamSolver
from app.services.risk_management import RiskManagementService
from app.services.backtesting import BacktestingEngine
from app.services.ml_engine import PLEngine
from app.services.third_party_data import ThirdPartyDataService, UnderstatService, FBrefService
from app.services.entity_resolution import EntityResolutionService
from app.services.data_cleaning import DataCleaningService
from app.services.etl_service import ETLService
import logging

logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FPL Point Prediction API",
    description="Machine Learning API for Fantasy Premier League point predictions",
    version="2.0.0"
)

# Initialize services
feature_service = FeatureEngineeringService()
fpl_api = FPLAPIService()
predictive_engine = PredictiveEngine()
team_solver = TeamSolver()
risk_service = RiskManagementService()
backtesting_engine = BacktestingEngine()
ml_engine = PLEngine()
third_party_service = ThirdPartyDataService()
entity_resolution = EntityResolutionService()
data_cleaning = DataCleaningService()
etl_service = ETLService()

# Scheduler for daily refresh
scheduler = AsyncIOScheduler()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "FPL Point Prediction API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Feature Engineering Endpoints

@app.post("/api/features/optimize-form-alpha", response_model=FormAlphaResponse)
async def optimize_form_alpha(request: FormAlphaOptimizeRequest):
    """
    Optimize form alpha coefficient using Bayesian Optimization.
    Minimizes RMSE by finding optimal exponential decay weight.
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.historical_data)
        
        if 'points' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="Historical data must include 'points' column"
            )
        
        # Optimize alpha
        optimal_alpha = feature_service.optimize_form_alpha(df)
        
        # Calculate final RMSE
        rmse = feature_service.form_alpha._calculate_rmse(
            optimal_alpha, 
            df, 
            request.lookback_weeks
        )
        
        return FormAlphaResponse(
            optimal_alpha=optimal_alpha,
            rmse=rmse,
            lookback_weeks=request.lookback_weeks
        )
    except Exception as e:
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
        p_start = predictive_engine.xmins_model.predict_start_probability(
            request.player_data,
            request.fixture_data
        )
        expected_minutes = predictive_engine.xmins_model.predict_expected_minutes(
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
        predictions = predictive_engine.attack_model.predict(
            request.player_data,
            request.fixture_data,
            request.fdr_data
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
        xcs = predictive_engine.defense_model.predict_clean_sheet_probability(
            request.team_data,
            request.opponent_data,
            request.is_home
        )
        
        expected_goals_conceded = predictive_engine.defense_model.calculate_expected_goals_conceded(
            request.team_data,
            request.opponent_data,
            request.is_home
        )
        
        return DefensePredictionResponse(
            xcs=xcs,
            expected_goals_conceded=expected_goals_conceded
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predictive/momentum", response_model=MomentumPredictionResponse)
async def predict_momentum(request: MomentumPredictionRequest):
    """
    Predict momentum and trend using LSTM time-series analysis.
    """
    try:
        momentum = predictive_engine.momentum_layer.predict_momentum(
            request.historical_points,
            request.forecast_steps
        )
        
        return MomentumPredictionResponse(
            momentum=momentum['momentum'],
            trend=momentum['trend'],
            forecast=momentum['forecast']
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
        predictions = predictive_engine.predict_comprehensive(
            player_data=request.player_data,
            historical_points=request.historical_points,
            fixture_data=request.fixture_data,
            fdr_data=request.fdr_data,
            team_data=request.team_data,
            opponent_data=request.opponent_data
        )
        
        return ComprehensivePredictionResponse(
            p_start=predictions['p_start'],
            expected_minutes=predictions['expected_minutes'],
            xg=predictions['xg'],
            xa=predictions['xa'],
            xcs=predictions['xcs'],
            momentum=predictions['momentum'],
            trend=predictions['trend']
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


# Frontend Integration Endpoints

@app.get("/api/players/all", response_model=List[PlayerDisplayData])
async def get_all_players(gameweek: int = 1):
    """
    Get all players with predictions for All Players page.
    """
    try:
        # Fetch bootstrap data
        bootstrap = await fpl_api.get_bootstrap_data()
        
        players_data = []
        elements = bootstrap.get('elements', [])
        
        for element in elements[:50]:  # Limit for demo
            player_id = element.get('id')
            
            # Get player summary
            try:
                player_summary = await fpl_api.get_player_data(player_id)
                history = player_summary.get('history', [])
                
                # Get predictions (simplified - in production, use ML engine)
                expected_points = element.get('points_per_game', 0.0) * 90 / 90  # Simplified
                
                players_data.append(PlayerDisplayData(
                    id=player_id,
                    fpl_id=player_id,
                    name=element.get('web_name', ''),
                    position=['GK', 'DEF', 'MID', 'FWD'][element.get('element_type', 1) - 1],
                    team=element.get('team', 0),
                    price=element.get('now_cost', 0) / 10.0,
                    expected_points=expected_points,
                    ownership_percent=float(element.get('selected_by_percent', 0.0)),
                    form=float(element.get('form', 0.0))
                ))
            except:
                continue
        
        return players_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dream-team", response_model=DreamTeamResponse)
async def get_dream_team(gameweek: int = 1):
    """
    Get optimal Dream Team for a gameweek.
    """
    try:
        # This would use the solver to find optimal team
        # For now, return a simplified version
        bootstrap = await fpl_api.get_bootstrap_data()
        elements = bootstrap.get('elements', [])
        
        # Sort by points per game and select best team
        sorted_players = sorted(
            elements,
            key=lambda x: x.get('points_per_game', 0),
            reverse=True
        )
        
        # Select by position (simplified)
        squad = []
        positions = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        for pos_name, count in positions.items():
            pos_num = [k for k, v in pos_map.items() if v == pos_name][0]
            pos_players = [p for p in sorted_players if p.get('element_type') == pos_num]
            
            for i, player in enumerate(pos_players[:count]):
                squad.append(DreamTeamPlayer(
                    player_id=player.get('id'),
                    name=player.get('web_name', ''),
                    position=pos_name,
                    team=player.get('team', 0),
                    expected_points=player.get('points_per_game', 0.0),
                    price=player.get('now_cost', 0) / 10.0
                ))
        
        # Starting XI (best 11)
        starting_xi = sorted(squad, key=lambda x: x.expected_points, reverse=True)[:11]
        
        return DreamTeamResponse(
            gameweek=gameweek,
            squad=squad,
            starting_xi=starting_xi,
            total_expected_points=sum(p.expected_points for p in starting_xi),
            total_cost=sum(p.price for p in squad),
            formation="4-4-2"  # Simplified
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    include_difficulty: bool = True
):
    """
    Get fixtures data with optional difficulty ratings.
    
    Args:
        gameweek: Optional gameweek filter
        include_difficulty: Calculate and include difficulty ratings
    """
    try:
        fixtures = await fpl_api.get_fixtures(gameweek)
        
        if include_difficulty:
            bootstrap = await fpl_api.get_bootstrap_data()
            teams = fpl_api.extract_teams_from_bootstrap(bootstrap)
            fixtures = fpl_api.extract_fixtures_with_difficulty(fixtures, teams)
        
        return {
            'fixtures': fixtures,
            'count': len(fixtures),
            'gameweek': gameweek
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
async def map_players_across_sources(season: str = "2025"):
    """
    Create mapping between FPL players and third-party data sources.
    Uses entity resolution to match player names.
    """
    try:
        # Get FPL players
        bootstrap = await fpl_api.get_bootstrap_data()
        fpl_players = fpl_api.extract_players_from_bootstrap(bootstrap)
        
        # Get Understat data
        understat_service = UnderstatService()
        understat_data = await understat_service.get_player_stats(season)
        
        # Get FBref data
        fbref_service = FBrefService()
        fbref_data = await fbref_service.get_player_defensive_stats(season)
        
        # Create mappings
        understat_mapping = understat_service.map_to_fpl_players(understat_data, fpl_players)
        fbref_mapping = fbref_service.map_to_fpl_players(fbref_data, fpl_players)
        
        return {
            'understat_mapping': {
                'total_fpl_players': len(fpl_players),
                'matched': len(understat_mapping),
                'mapping': understat_mapping
            },
            'fbref_mapping': {
                'total_fpl_players': len(fpl_players),
                'matched': len(fbref_mapping),
                'mapping': fbref_mapping
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Entity Resolution Endpoints

@app.on_event("startup")
async def startup_event():
    """Load Master ID Map and start scheduler on startup"""
    try:
        await entity_resolution.load_master_map()
        logger.info("Master ID Map loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Master ID Map on startup: {str(e)}")
    
    # Start daily ETL scheduler (runs at 2 AM daily)
    try:
        scheduler.add_job(
            daily_etl_refresh,
            CronTrigger(hour=2, minute=0),
            id='daily_etl_refresh',
            name='Daily ETL Refresh',
            replace_existing=True
        )
        scheduler.start()
        logger.info("Daily ETL scheduler started (runs at 2 AM daily)")
    except Exception as e:
        logger.warning(f"Failed to start ETL scheduler: {str(e)}")


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
