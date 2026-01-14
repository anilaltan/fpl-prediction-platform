from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class Player(Base):
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, index=True)
    fpl_id = Column(Integer, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    team = Column(String, nullable=False)
    position = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    total_points = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    predictions = relationship("Prediction", back_populates="player")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    gameweek = Column(Integer, nullable=False)
    predicted_points = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    player = relationship("Player", back_populates="predictions")

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
