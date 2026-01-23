from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, UniqueConstraint, Numeric, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

# ============================================================================
# CORE REFERENCE TABLES (PRD Schema)
# ============================================================================

class Team(Base):
    """Core reference table for Premier League teams (PRD Schema)"""
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    short_name = Column(String(10))
    strength_attack = Column(Integer)
    strength_defense = Column(Integer)
    strength_overall = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    players = relationship("Player", back_populates="team")
    home_fixtures = relationship("Fixture", foreign_keys="Fixture.home_team_id", back_populates="home_team")
    away_fixtures = relationship("Fixture", foreign_keys="Fixture.away_team_id", back_populates="away_team")
    team_stats = relationship("TeamStats", back_populates="team")


class Player(Base):
    """
    Core reference table for FPL players (PRD Schema)
    Note: This model matches the PRD specification where id is the FPL player ID
    """
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, index=True)  # FPL player ID
    name = Column(String(255), nullable=False)
    position = Column(String(3), nullable=False)  # GK, DEF, MID, FWD
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    price = Column(Numeric(5, 2))
    ownership = Column(Numeric(5, 2))
    canonical_name = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    team = relationship("Team", back_populates="players")
    predictions = relationship("Prediction", back_populates="player")
    entity_mapping = relationship("EntityMapping", back_populates="player", uselist=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("position IN ('GK', 'DEF', 'MID', 'FWD')", name='check_position'),
    )


class Fixture(Base):
    """Fixture information for historical and future matches (PRD Schema)"""
    __tablename__ = "fixtures"
    
    id = Column(Integer, primary_key=True, index=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    gameweek = Column(Integer, nullable=False, index=True)
    season = Column(String(9), nullable=False, index=True)
    kickoff_time = Column(DateTime(timezone=True))
    finished = Column(Boolean, default=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    fdr_home = Column(Numeric(4, 2))
    fdr_away = Column(Numeric(4, 2))
    xgc_home = Column(Numeric(6, 3))
    xgc_away = Column(Numeric(6, 3))
    xgs_home = Column(Numeric(6, 3))
    xgs_away = Column(Numeric(6, 3))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_fixtures")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_fixtures")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('home_team_id', 'away_team_id', 'gameweek', 'season', 
                        name='uq_fixtures_teams_gameweek_season'),
    )


class EntityMapping(Base):
    """Entity resolution mappings across FPL, Understat, and FBref (PRD Schema)"""
    __tablename__ = "entity_mappings"
    
    id = Column(Integer, primary_key=True, index=True)
    fpl_id = Column(Integer, ForeignKey("players.id"), nullable=False, unique=True)
    understat_name = Column(String(255))
    fbref_name = Column(String(255))
    canonical_name = Column(String(255), nullable=False)
    confidence_score = Column(Numeric(3, 2))  # 0.0 to 1.0
    manually_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="entity_mapping")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("confidence_score >= 0.0 AND confidence_score <= 1.0", 
                       name='check_confidence_score'),
    )


class PlayerStats(Base):
    """
    Time-series statistics for FPL players per gameweek (PRD Schema)
    Converted to TimescaleDB hypertable for time-series optimization
    """
    __tablename__ = "player_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    gameweek = Column(Integer, nullable=False, index=True)
    season = Column(String(9), nullable=False, index=True)
    
    # Match statistics
    minutes = Column(Integer, default=0)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    clean_sheets = Column(Integer, default=0)
    points = Column(Integer, default=0)
    
    # Expected statistics
    xg = Column(Numeric(6, 3))
    xa = Column(Numeric(6, 3))
    npxg = Column(Numeric(6, 3))
    xmins = Column(Numeric(5, 2))
    xp = Column(Numeric(6, 2))
    defcon_points = Column(Integer, default=0)
    
    # Time-series timestamp (for TimescaleDB hypertable)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    player = relationship("Player")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('player_id', 'gameweek', 'season', name='uq_player_stats_player_gameweek_season'),
    )


class TeamStats(Base):
    """
    Time-series statistics for Premier League teams per gameweek (PRD Schema)
    Converted to TimescaleDB hypertable for time-series optimization
    """
    __tablename__ = "team_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    gameweek = Column(Integer, nullable=False, index=True)
    season = Column(String(9), nullable=False, index=True)
    
    # Team statistics
    xgc = Column(Numeric(6, 3))  # Expected Goals Conceded
    xgs = Column(Numeric(6, 3))  # Expected Goals Scored
    possession = Column(Numeric(5, 2))  # Possession percentage (0-100)
    clean_sheets = Column(Integer, default=0)
    goals_conceded = Column(Integer, default=0)
    
    # Time-series timestamp (for TimescaleDB hypertable)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    team = relationship("Team")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('team_id', 'gameweek', 'season', name='uq_team_stats_team_gameweek_season'),
    )


# ============================================================================
# EXISTING MODELS (Legacy/Additional)
# ============================================================================

class Prediction(Base):
    """
    Batch Prediction table - stores pre-calculated ML predictions for fast API responses.
    Predictions are calculated in background and stored here, API just reads from this table.
    """
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    fpl_id = Column(Integer, nullable=False, index=True)  # Use fpl_id for consistency with PlayerGameweekStats
    gameweek = Column(Integer, nullable=False, index=True)
    season = Column(String, nullable=False, default="2025-26", index=True)
    
    # ML Prediction outputs
    xp = Column(Float, nullable=False)  # expected_points (renamed from predicted_points for clarity)
    xg = Column(Float, default=0.0)
    xa = Column(Float, default=0.0)
    xmins = Column(Float, default=0.0)
    xcs = Column(Float, default=0.0)
    defcon_score = Column(Float, default=0.0)
    confidence_score = Column(Float, nullable=False, default=0.7)  # Model confidence (0.0-1.0)
    
    # Metadata
    model_version = Column(String, nullable=True)
    calculated_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Optional: Keep relationship for backward compatibility
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    player = relationship("Player", back_populates="predictions")
    
    # Unique constraint: one prediction per player per gameweek per season
    __table_args__ = (
        UniqueConstraint('fpl_id', 'gameweek', 'season', name='uq_prediction_fpl_gameweek_season'),
    )

class ModelPerformance(Base):
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String, nullable=False)
    gameweek = Column(Integer, nullable=False)
    mae = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TeamFDR(Base):
    """Store team attack/defense strengths from Dixon-Coles model"""
    __tablename__ = "team_fdr"
    
    id = Column(Integer, primary_key=True, index=True)
    team_name = Column(String, nullable=False, unique=True, index=True)
    attack_strength = Column(Float, nullable=False)
    defense_strength = Column(Float, nullable=False)
    home_advantage = Column(Float, nullable=False)
    gameweek = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class FormAlpha(Base):
    """Store optimal form alpha coefficient"""
    __tablename__ = "form_alpha"
    
    id = Column(Integer, primary_key=True, index=True)
    gameweek = Column(Integer, nullable=False, unique=True, index=True)
    optimal_alpha = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    lookback_weeks = Column(Integer, nullable=False, default=5)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BacktestResult(Base):
    """Store backtesting results"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String, nullable=False)
    methodology = Column(String, nullable=False)  # 'expanding_window' or 'rolling_window'
    season = Column(String, nullable=False)
    gameweek = Column(Integer, nullable=False)
    rmse = Column(Float, nullable=False)
    mae = Column(Float, nullable=False)
    spearman_corr = Column(Float, nullable=False)
    n_predictions = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BacktestSummary(Base):
    """Store overall backtest summary metrics"""
    __tablename__ = "backtest_summary"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String, nullable=False, unique=True, index=True)
    methodology = Column(String, nullable=False)
    season = Column(String, nullable=False)
    total_weeks_tested = Column(Integer, nullable=False)
    overall_rmse = Column(Float, nullable=False)
    overall_mae = Column(Float, nullable=False)
    overall_spearman_corr = Column(Float, nullable=False)
    r_squared = Column(Float, nullable=False)
    total_predictions = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class PlayerGameweekStats(Base):
    """Store player statistics for each gameweek"""
    __tablename__ = "player_gameweek_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    fpl_id = Column(Integer, nullable=False, index=True)
    gameweek = Column(Integer, nullable=False, index=True)
    season = Column(String, nullable=False, index=True)
    
    # Match stats
    minutes = Column(Integer, default=0)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    clean_sheets = Column(Integer, default=0)
    goals_conceded = Column(Integer, default=0)
    own_goals = Column(Integer, default=0)
    penalties_saved = Column(Integer, default=0)
    penalties_missed = Column(Integer, default=0)
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)
    saves = Column(Integer, default=0)
    bonus = Column(Integer, default=0)
    bps = Column(Integer, default=0)
    
    # Points
    total_points = Column(Integer, default=0)
    normalized_points = Column(Float, default=0.0)  # DGW normalized
    
    # Expected stats
    xg = Column(Float, default=0.0)
    xa = Column(Float, default=0.0)
    xgi = Column(Float, default=0.0)
    xgc = Column(Float, default=0.0)
    npxg = Column(Float, default=0.0)
    
    # ICT Index
    influence = Column(Float, default=0.0)
    creativity = Column(Float, default=0.0)
    threat = Column(Float, default=0.0)
    ict_index = Column(Float, default=0.0)
    
    # DefCon metrics (2025/26)
    blocks = Column(Integer, default=0)
    interventions = Column(Integer, default=0)
    passes = Column(Integer, default=0)
    defcon_floor_points = Column(Float, default=0.0)
    
    # Match info
    was_home = Column(Boolean, default=True)
    opponent_team = Column(Integer, nullable=True)
    team_score = Column(Integer, nullable=True)
    opponent_score = Column(Integer, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Unique constraint: one record per player per gameweek per season
    __table_args__ = (
        UniqueConstraint('fpl_id', 'gameweek', 'season', name='uq_player_gameweek_season'),
    )


class MarketIntelligence(Base):
    """
    Market Intelligence table - stores ownership arbitrage analysis results.
    Identifies overvalued and undervalued assets based on ownership trends and xP.
    """
    __tablename__ = "market_intelligence"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    gameweek = Column(Integer, nullable=False, index=True)
    season = Column(String(9), nullable=False, index=True, default="2025-26")
    
    # Ranking metrics
    xp_rank = Column(Integer, nullable=False)  # Rank by expected points (descending: highest xP = rank 1)
    ownership_rank = Column(Integer, nullable=False)  # Rank by ownership percentage (descending: highest ownership = rank 1)
    
    # Arbitrage score: (xp_rank - ownership_rank)
    # Negative = Differential (high xP, low ownership)
    # Positive = Overvalued (low xP, high ownership)
    arbitrage_score = Column(Float, nullable=False)
    
    # Category: 'Differential', 'Overvalued', or 'Neutral'
    category = Column(String(50), nullable=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    player = relationship("Player")
    
    # Unique constraint: one record per player per gameweek per season
    __table_args__ = (
        UniqueConstraint('player_id', 'gameweek', 'season', name='uq_market_intelligence_player_gameweek_season'),
    )
