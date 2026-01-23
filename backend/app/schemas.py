from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime


# Player Schemas
class PlayerBase(BaseModel):
    fpl_id: int
    name: str
    team: str
    position: str
    price: float


class PlayerCreate(PlayerBase):
    pass


class Player(PlayerBase):
    id: int
    total_points: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


# Prediction Schemas
class PredictionBase(BaseModel):
    player_id: int
    gameweek: int
    predicted_points: float
    confidence_score: float
    model_version: str


class PredictionCreate(PredictionBase):
    pass


class Prediction(PredictionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Model Performance Schemas
class ModelPerformanceBase(BaseModel):
    model_version: str
    gameweek: int
    mae: float
    rmse: float
    accuracy: float


class ModelPerformanceCreate(ModelPerformanceBase):
    pass


class ModelPerformance(ModelPerformanceBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# API Response Schemas
class PredictionResponse(BaseModel):
    player_id: int
    player_name: str
    gameweek: int
    predicted_points: float
    confidence_score: float


# Feature Engineering Schemas
class FormAlphaOptimizeRequest(BaseModel):
    historical_data: List[Dict]  # List of player data with points
    lookback_weeks: int = 5
    n_calls: int = 50


class FormAlphaResponse(BaseModel):
    optimal_alpha: float
    rmse: float
    lookback_weeks: int
    converged: Optional[bool] = False
    iterations: Optional[int] = 0


class FDRFitRequest(BaseModel):
    fixtures: List[Dict]  # List of fixture data


class FDRResponse(BaseModel):
    team_name: str
    attack_strength: float
    defense_strength: float
    home_advantage: float


class StochasticFDRRequest(BaseModel):
    team: str
    opponent: str
    is_home: bool
    n_simulations: Optional[int] = 10000


class StochasticFDRResponse(BaseModel):
    fdr_mean: float
    fdr_std: float
    win_prob: float
    draw_prob: float
    loss_prob: float
    clean_sheet_prob: float
    expected_goals_for: float
    expected_goals_against: float
    goal_distribution: Dict


class FDRComparisonResponse(BaseModel):
    correlation: float
    mean_absolute_error: float
    r_squared: float
    comparison_data: List[Dict]


class FDRVerificationResponse(BaseModel):
    correlation_goals_for: float
    correlation_goals_against: float
    prediction_accuracy: float
    n_fixtures: int


class DefConFeaturesResponse(BaseModel):
    floor_points: float
    blocks_per_90: float
    interventions_per_90: float
    passes_per_90: float
    defcon_score: float


# Predictive Engine Schemas
class XMinsPredictionRequest(BaseModel):
    player_data: Dict
    fixture_data: Optional[Dict] = None


class XMinsPredictionResponse(BaseModel):
    p_start: float
    expected_minutes: float


class AttackPredictionRequest(BaseModel):
    player_data: Dict
    fixture_data: Optional[Dict] = None
    fdr_data: Optional[Dict] = None


class AttackPredictionResponse(BaseModel):
    xg: float
    xa: float


class DefensePredictionRequest(BaseModel):
    team_data: Dict
    opponent_data: Dict
    is_home: bool = True


class DefensePredictionResponse(BaseModel):
    xcs: float
    expected_goals_conceded: float


class MomentumPredictionRequest(BaseModel):
    historical_points: List[float]
    forecast_steps: int = 1


class MomentumPredictionResponse(BaseModel):
    momentum: float
    trend: float
    forecast: float


class ComprehensivePredictionRequest(BaseModel):
    player_data: Dict
    historical_points: List[float]
    fixture_data: Optional[Dict] = None
    fdr_data: Optional[Dict] = None
    team_data: Optional[Dict] = None
    opponent_data: Optional[Dict] = None


class ComprehensivePredictionResponse(BaseModel):
    p_start: float
    expected_minutes: float
    xg: float
    xa: float
    xcs: float
    momentum: float
    trend: float


# Team Solver Schemas
class PlayerOptimizationData(BaseModel):
    id: int
    name: str
    position: str
    price: float
    team_id: int
    team_name: str
    expected_points_gw1: float
    expected_points_gw2: Optional[float] = None
    expected_points_gw3: Optional[float] = None
    expected_points_gw4: Optional[float] = None
    expected_points_gw5: Optional[float] = None


class TeamOptimizationRequest(BaseModel):
    players: List[PlayerOptimizationData]
    current_squad: Optional[List[int]] = None
    locked_players: Optional[List[int]] = None
    excluded_players: Optional[List[int]] = None
    budget: float = 100.0
    horizon_weeks: int = 3
    free_transfers: int = 1


class TransferInfo(BaseModel):
    transfers_in: List[int]  # Changed from 'in' (Python keyword) to 'transfers_in'
    transfers_out: List[int]  # Changed from 'out' for consistency
    count: int
    cost: float


class WeekPoints(BaseModel):
    expected_points: float
    transfer_cost: float
    net_points: float


class TeamOptimizationResponse(BaseModel):
    status: str
    optimal: bool
    squads: Dict[int, List[int]]
    starting_xis: Dict[int, List[int]]
    transfers: Dict[int, TransferInfo]
    points_breakdown: Dict[int, WeekPoints]
    total_points: float
    total_transfers: int
    budget_used: Dict[int, float]


# Risk Management Schemas
class OwnershipArbitrageRequest(BaseModel):
    players: List[PlayerOptimizationData]
    gameweek: int = 1
    ownership_threshold: float = 20.0
    xp_threshold: float = 5.0


class OverOwnedPlayer(BaseModel):
    player_id: int
    name: str
    position: str
    team: str
    ownership_percent: float
    expected_points: float
    price: float
    value_score: float


class DifferentialAlternative(BaseModel):
    player_id: int
    name: str
    position: str
    team: str
    ownership_percent: float
    expected_points: float
    price: float
    value_score: float
    xp_improvement: float
    ownership_differential: float
    price_diff: float


class ArbitrageOpportunity(BaseModel):
    over_owned_player: OverOwnedPlayer
    alternatives: List[DifferentialAlternative]
    potential_gain: float


class OwnershipArbitrageResponse(BaseModel):
    gameweek: int
    over_owned_count: int
    opportunities: List[ArbitrageOpportunity]
    total_potential_gain: float
    recommendations: List[str]


class CaptainSelectionRequest(BaseModel):
    candidates: List[PlayerOptimizationData]
    gameweek: int = 1


class CaptainAnalysis(BaseModel):
    captain_xp: float
    vice_captain_xp: float
    captain_no_play_prob: float
    vice_captain_no_play_prob: float
    weighted_expected_value: float
    risk_adjusted_value: float
    should_swap: bool
    expected_gain_if_swap: float


class CaptainSelectionResponse(BaseModel):
    captain: PlayerOptimizationData
    vice_captain: PlayerOptimizationData
    analysis: CaptainAnalysis


class ChipAnalysisRequest(BaseModel):
    current_squad: List[PlayerOptimizationData]
    optimized_squad: Optional[List[PlayerOptimizationData]] = None
    free_hit_squad: Optional[List[PlayerOptimizationData]] = None
    gameweek: int = 1


class ChipValue(BaseModel):
    chip: str
    expected_gain: float
    threshold: float
    should_play: bool
    recommendation: str


class ChipAnalysisResponse(BaseModel):
    gameweek: int
    chips: Dict[str, ChipValue]
    recommendation: str


class ComprehensiveRiskAnalysisRequest(BaseModel):
    players: List[PlayerOptimizationData]
    current_squad: List[PlayerOptimizationData]
    gameweek: int = 1
    optimized_squad: Optional[List[PlayerOptimizationData]] = None


class ComprehensiveRiskAnalysisResponse(BaseModel):
    gameweek: int
    ownership_arbitrage: OwnershipArbitrageResponse
    chip_analysis: ChipAnalysisResponse


# Backtesting Schemas
class WeeklyMetrics(BaseModel):
    gameweek: int
    rmse: float
    mae: float
    spearman_corr: float
    n_predictions: int
    mean_actual: float
    mean_predicted: float


class OverallMetrics(BaseModel):
    rmse: float
    mae: float
    spearman_corr: float
    r_squared: float
    mean_actual: float
    mean_predicted: float
    n_predictions: int


class BacktestRequest(BaseModel):
    historical_data: List[Dict]  # List of player records with gameweek, points, etc.
    min_train_weeks: int = 5
    methodology: str = "expanding_window"  # "expanding_window" or "rolling_window"
    window_size: Optional[int] = None


class BacktestResponse(BaseModel):
    methodology: str
    min_train_weeks: Optional[int] = None
    window_size: Optional[int] = None
    total_weeks_tested: int
    weekly_results: List[WeeklyMetrics]
    overall_metrics: OverallMetrics
    total_predictions: int


# Frontend Integration Schemas
class PlayerDisplayData(BaseModel):
    id: int
    fpl_id: int
    name: str
    position: str
    team: str
    price: float
    expected_points: float
    ownership_percent: float
    form: float
    xg: Optional[float] = None
    xa: Optional[float] = None
    xmins: Optional[float] = None
    xcs: Optional[float] = None
    defcon_score: Optional[float] = None


class DreamTeamPlayer(BaseModel):
    player_id: int
    name: str
    position: str
    team: str
    expected_points: float
    price: float


class DreamTeamResponse(BaseModel):
    gameweek: int
    squad: List[DreamTeamPlayer]
    starting_xi: List[DreamTeamPlayer]
    total_expected_points: float
    total_cost: float
    formation: str


# Third-Party Data Schemas
class UnderstatPlayerData(BaseModel):
    name: str
    team: str
    position: str
    games: int
    time: int
    goals: int
    assists: int
    xg: float
    xa: float
    npxg: float
    xg_per_90: float
    xa_per_90: float
    npxg_per_90: float


class FBrefDefensiveData(BaseModel):
    name: str
    team: str
    position: str
    games: int
    minutes: int
    blocks: int
    blocks_per_90: float
    tackles: int
    interceptions: int
    interventions: int
    interventions_per_90: float
    passes: int
    passes_per_90: float


class EnrichedPlayerData(BaseModel):
    fpl_id: int
    name: str
    position: str
    team: str
    # FPL data
    price: float
    form: float
    expected_points: float
    # Understat data
    understat_xg: Optional[float] = None
    understat_xa: Optional[float] = None
    understat_npxg: Optional[float] = None
    understat_xg_per_90: Optional[float] = None
    understat_xa_per_90: Optional[float] = None
    understat_npxg_per_90: Optional[float] = None
    # FBref data
    fbref_blocks: Optional[int] = None
    fbref_blocks_per_90: Optional[float] = None
    fbref_interventions: Optional[int] = None
    fbref_interventions_per_90: Optional[float] = None
    fbref_passes: Optional[int] = None
    fbref_passes_per_90: Optional[float] = None


# Entity Resolution Schemas
class PlayerResolutionRequest(BaseModel):
    fpl_id: int
    fpl_name: str
    fpl_team: Optional[str] = None
    understat_name: Optional[str] = None
    fbref_name: Optional[str] = None


class PlayerResolutionResponse(BaseModel):
    fpl_id: int
    fpl_name: str
    fpl_team: Optional[str] = None
    matched: bool
    match_method: Optional[str] = None  # 'master_map', 'fuzzy_match', None
    global_id: Optional[int] = None
    understat_id: Optional[int] = None
    fbref_id: Optional[str] = None
    confidence: float
    fuzzy_match_name: Optional[str] = None


class BulkResolutionRequest(BaseModel):
    players: List[PlayerResolutionRequest]
    include_fuzzy: bool = True


class BulkResolutionResponse(BaseModel):
    resolved: Dict[int, PlayerResolutionResponse]
    total_players: int
    matched_count: int
    unmatched_count: int
    unmatched_players: List[PlayerResolutionResponse]


class ManualMappingRequest(BaseModel):
    fpl_id: int
    fpl_name: str
    understat_id: Optional[int] = None
    fbref_id: Optional[str] = None
    understat_name: Optional[str] = None
    fbref_name: Optional[str] = None


class OverrideMappingRequest(BaseModel):
    """Request schema for manually overriding low-confidence entity mappings"""

    fpl_id: int
    understat_name: Optional[str] = None
    fbref_name: Optional[str] = None
    fpl_name: Optional[str] = None
    canonical_name: Optional[str] = None


class BulkResolutionReport(BaseModel):
    """Response schema for bulk resolution report"""

    total_players: int
    matched_count: int
    unmatched_count: int
    high_confidence_count: int
    low_confidence_count: int
    manually_verified_count: int
    low_confidence_mappings: List[Dict]
    unmatched_players: List[Dict]
    match_accuracy: float
    mappings_stored: int


# Data Cleaning Schemas
class DataCleaningRequest(BaseModel):
    player_data: Dict
    normalize_dgw: bool = True
    calculate_defcon: bool = True
    convert_types: bool = True
    position: Optional[str] = None


class DataCleaningResponse(BaseModel):
    cleaned_data: Dict
    normalized_points: Optional[float] = None
    defcon_floor_points: Optional[float] = None
    type_conversions: Dict[str, str]  # Original type -> converted type


class BulkCleaningRequest(BaseModel):
    players_data: List[Dict]
    normalize_dgw: bool = True
    calculate_defcon: bool = True
    convert_types: bool = True


class BulkCleaningResponse(BaseModel):
    cleaned_players: List[Dict]
    total_players: int
    normalized_count: int
    defcon_calculated_count: int


class DefConMetricsResponse(BaseModel):
    floor_points: float
    floor_points_90: float
    blocks_per_90: float
    interventions_per_90: float
    passes_per_90: float
    defcon_score: float


# Market Intelligence Schemas
class MarketIntelligencePlayer(BaseModel):
    player_id: int
    name: str
    position: str
    team: str
    price: float
    xp: float
    ownership: float
    xp_rank: int
    ownership_rank: int
    arbitrage_score: float
    category: str  # 'Differential', 'Overvalued', or 'Neutral'


class MarketIntelligenceResponse(BaseModel):
    gameweek: int
    season: str
    players: List[MarketIntelligencePlayer]
    total_players: int
    differentials_count: int
    overvalued_count: int
    neutral_count: int


# Team Planning Schemas
class TeamPlanRequest(BaseModel):
    players: List[PlayerOptimizationData]
    current_squad: Optional[List[int]] = None
    locked_players: Optional[List[int]] = None
    excluded_players: Optional[List[int]] = None
    budget: float = 100.0
    horizon_weeks: int = 3
    free_transfers: int = 1


class TransferStrategy(BaseModel):
    gameweek: int
    transfers_in: List[int]
    transfers_out: List[int]
    transfer_count: int
    transfer_cost: float
    expected_points_gain: float


class TeamPlanResponse(BaseModel):
    status: str
    optimal: bool
    horizon_weeks: int
    squads: Dict[int, List[int]]
    starting_xis: Dict[int, List[int]]
    transfer_strategy: List[TransferStrategy]
    total_expected_points: float
    total_transfer_cost: float
    net_expected_points: float
    budget_used: Dict[int, float]
