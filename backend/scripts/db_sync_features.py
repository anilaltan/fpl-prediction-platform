"""
Database Feature Synchronization Script
Calculates and syncs feature engineering data:
1. Team xG/xGC stats from finished fixtures
2. Dixon-Coles FDR model (team_fdr table)
3. Dynamic Form Alpha optimization (form_alpha table)
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models import Fixture, Team, TeamStats, TeamFDR, FormAlpha, PlayerGameweekStats
from app.services.feature_engineering import DixonColesFDR, DynamicFormAlpha

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_team_xg_stats(db: Session, season: str = "2025-26") -> Dict:
    """
    Calculate team xG/xGC stats from finished fixtures and save to team_stats table.
    
    Args:
        db: Database session
        season: Season identifier
        
    Returns:
        Dictionary with summary statistics
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Calculating team xG/xGC stats from finished fixtures")
    logger.info("=" * 60)
    
    # Query finished fixtures
    finished_fixtures = db.query(Fixture).filter(
        Fixture.finished == True,
        Fixture.season == season
    ).all()
    
    logger.info(f"Found {len(finished_fixtures)} finished fixtures")
    
    if not finished_fixtures:
        logger.warning("No finished fixtures found. Skipping team stats calculation.")
        return {"processed": 0, "created": 0, "updated": 0}
    
    # Get all teams for lookup
    teams = {team.id: team for team in db.query(Team).all()}
    
    # Aggregate stats by team and gameweek
    team_stats_dict = {}
    
    for fixture in finished_fixtures:
        home_team_id = fixture.home_team_id
        away_team_id = fixture.away_team_id
        gameweek = fixture.gameweek
        
        # Home team stats
        if home_team_id not in team_stats_dict:
            team_stats_dict[home_team_id] = {}
        
        if gameweek not in team_stats_dict[home_team_id]:
            team_stats_dict[home_team_id][gameweek] = {
                "xgs": 0.0,
                "xgc": 0.0,
                "goals_conceded": 0,
                "clean_sheets": 0,
            }
        
        # Home team stats
        # xgs_home = expected goals scored by home team
        # xgc_home = expected goals conceded by home team (same as xgs_away)
        # Use += to handle double gameweeks (DGWs) where a team plays twice
        if fixture.xgs_home is not None:
            team_stats_dict[home_team_id][gameweek]["xgs"] += float(fixture.xgs_home)
        elif fixture.home_score is not None:
            # Fallback to actual goals if xG not available
            team_stats_dict[home_team_id][gameweek]["xgs"] += float(fixture.home_score)
        
        if fixture.xgc_home is not None:
            team_stats_dict[home_team_id][gameweek]["xgc"] += float(fixture.xgc_home)
        elif fixture.away_score is not None:
            # Fallback: xgc_home = goals scored by away team
            team_stats_dict[home_team_id][gameweek]["xgc"] += float(fixture.away_score)
        
        if fixture.away_score is not None:
            team_stats_dict[home_team_id][gameweek]["goals_conceded"] += fixture.away_score
            # Clean sheet if no goals conceded (for this match)
            if fixture.away_score == 0:
                team_stats_dict[home_team_id][gameweek]["clean_sheets"] = 1
        
        # Away team stats
        if away_team_id not in team_stats_dict:
            team_stats_dict[away_team_id] = {}
        
        if gameweek not in team_stats_dict[away_team_id]:
            team_stats_dict[away_team_id][gameweek] = {
                "xgs": 0.0,
                "xgc": 0.0,
                "goals_conceded": 0,
                "clean_sheets": 0,
            }
        
        # xgs_away = expected goals scored by away team
        # xgc_away = expected goals conceded by away team (same as xgs_home)
        # Use += to handle double gameweeks (DGWs) where a team plays twice
        if fixture.xgs_away is not None:
            team_stats_dict[away_team_id][gameweek]["xgs"] += float(fixture.xgs_away)
        elif fixture.away_score is not None:
            # Fallback to actual goals if xG not available
            team_stats_dict[away_team_id][gameweek]["xgs"] += float(fixture.away_score)
        
        if fixture.xgc_away is not None:
            team_stats_dict[away_team_id][gameweek]["xgc"] += float(fixture.xgc_away)
        elif fixture.home_score is not None:
            # Fallback: xgc_away = goals scored by home team
            team_stats_dict[away_team_id][gameweek]["xgc"] += float(fixture.home_score)
        
        if fixture.home_score is not None:
            team_stats_dict[away_team_id][gameweek]["goals_conceded"] += fixture.home_score
            # Clean sheet if no goals conceded (for this match)
            if fixture.home_score == 0:
                team_stats_dict[away_team_id][gameweek]["clean_sheets"] = 1
    
    # Save to database
    created = 0
    updated = 0
    
    for team_id, gameweeks in team_stats_dict.items():
        if team_id not in teams:
            logger.warning(f"Team ID {team_id} not found in teams table. Skipping.")
            continue
        
        for gameweek, stats in gameweeks.items():
            # Check if entry exists
            existing = db.query(TeamStats).filter(
                TeamStats.team_id == team_id,
                TeamStats.gameweek == gameweek,
                TeamStats.season == season
            ).first()
            
            if existing:
                # Update existing entry
                existing.xgs = stats["xgs"]
                existing.xgc = stats["xgc"]
                existing.goals_conceded = stats["goals_conceded"]
                existing.clean_sheets = stats["clean_sheets"]
                existing.updated_at = datetime.utcnow()
                updated += 1
            else:
                # Create new entry
                new_entry = TeamStats(
                    team_id=team_id,
                    gameweek=gameweek,
                    season=season,
                    xgs=stats["xgs"],
                    xgc=stats["xgc"],
                    goals_conceded=stats["goals_conceded"],
                    clean_sheets=stats["clean_sheets"],
                    timestamp=datetime.utcnow()
                )
                db.add(new_entry)
                created += 1
    
    db.commit()
    logger.info(f"Team stats: {created} created, {updated} updated")
    
    return {
        "processed": len(finished_fixtures),
        "created": created,
        "updated": updated
    }


def sync_team_fdr(db: Session, season: str = "2025-26") -> Dict:
    """
    Fit Dixon-Coles FDR model and save to team_fdr table.
    
    Args:
        db: Database session
        season: Season identifier
        
    Returns:
        Dictionary with summary statistics
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Fitting Dixon-Coles FDR model")
    logger.info("=" * 60)
    
    # Query finished fixtures with team names
    finished_fixtures = db.query(
        Fixture,
        Team.name.label("home_team_name")
    ).join(
        Team, Fixture.home_team_id == Team.id
    ).filter(
        Fixture.finished == True,
        Fixture.season == season,
        Fixture.home_score.isnot(None),
        Fixture.away_score.isnot(None)
    ).all()
    
    logger.info(f"Found {len(finished_fixtures)} finished fixtures with scores")
    
    if not finished_fixtures:
        logger.warning("No finished fixtures with scores found. Skipping FDR calculation.")
        return {"processed": 0, "saved": 0}
    
    # Get away team names
    fixtures_data = []
    for fixture, home_team_name in finished_fixtures:
        away_team = db.query(Team).filter(Team.id == fixture.away_team_id).first()
        if not away_team:
            continue
        
        fixtures_data.append({
            "team_h": home_team_name,
            "team_a": away_team.name,
            "goals_h": fixture.home_score,
            "goals_a": fixture.away_score,
        })
    
    if not fixtures_data:
        logger.warning("No valid fixture data after processing. Skipping FDR calculation.")
        return {"processed": 0, "saved": 0}
    
    # Fit Dixon-Coles model
    fdr_model = DixonColesFDR()
    fdr_model.fit(fixtures_data)
    
    if not fdr_model.is_fitted:
        logger.error("Failed to fit Dixon-Coles model")
        return {"processed": len(fixtures_data), "saved": 0}
    
    # Get current gameweek (use max gameweek from fixtures)
    current_gameweek = db.query(Fixture.gameweek).filter(
        Fixture.season == season,
        Fixture.finished == True
    ).order_by(Fixture.gameweek.desc()).first()
    
    gameweek = current_gameweek[0] if current_gameweek else 1
    
    # Save to database
    saved = 0
    
    for team_name, attack_strength in fdr_model.attack_strengths.items():
        defense_strength = fdr_model.defense_strengths.get(team_name, 0.0)
        
        # Check if entry exists
        existing = db.query(TeamFDR).filter(
            TeamFDR.team_name == team_name
        ).first()
        
        if existing:
            # Update existing entry
            existing.attack_strength = attack_strength
            existing.defense_strength = defense_strength
            existing.home_advantage = fdr_model.home_advantage
            existing.gameweek = gameweek
            existing.updated_at = datetime.utcnow()
        else:
            # Create new entry
            new_entry = TeamFDR(
                team_name=team_name,
                attack_strength=attack_strength,
                defense_strength=defense_strength,
                home_advantage=fdr_model.home_advantage,
                gameweek=gameweek
            )
            db.add(new_entry)
        
        saved += 1
    
    db.commit()
    logger.info(f"FDR data saved for {saved} teams")
    logger.info(f"Home advantage: {fdr_model.home_advantage:.4f}")
    
    return {
        "processed": len(fixtures_data),
        "saved": saved,
        "home_advantage": fdr_model.home_advantage
    }


def sync_form_alpha(db: Session, season: str = "2025-26", lookback_weeks: int = 5) -> Dict:
    """
    Optimize form alpha and save to form_alpha table.
    
    Args:
        db: Database session
        season: Season identifier
        lookback_weeks: Number of weeks to consider for form
        
    Returns:
        Dictionary with optimization results
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Optimizing form alpha")
    logger.info("=" * 60)
    
    # Query historical player points
    player_stats = db.query(PlayerGameweekStats).filter(
        PlayerGameweekStats.season == season
    ).order_by(
        PlayerGameweekStats.fpl_id,
        PlayerGameweekStats.gameweek
    ).all()
    
    logger.info(f"Found {len(player_stats)} player gameweek stats")
    
    if not player_stats:
        logger.warning("No player stats found. Skipping form alpha optimization.")
        return {"processed": 0, "saved": False}
    
    # Convert to DataFrame
    stats_data = []
    for stat in player_stats:
        stats_data.append({
            "fpl_id": stat.fpl_id,
            "gameweek": stat.gameweek,
            "points": stat.total_points if stat.total_points is not None else 0,
            "minutes": stat.minutes if stat.minutes is not None else 0,
        })
    
    df = pd.DataFrame(stats_data)
    
    if len(df) < lookback_weeks + 1:
        logger.warning(
            f"Insufficient data for alpha optimization (need at least {lookback_weeks + 1} gameweeks, "
            f"found {len(df['gameweek'].unique()) if 'gameweek' in df.columns else 0})"
        )
        return {"processed": len(df), "saved": False}
    
    # Optimize alpha
    form_alpha = DynamicFormAlpha()
    result = form_alpha.optimize_alpha(
        df,
        lookback_weeks=lookback_weeks,
        n_calls=50
    )
    
    if not result or "optimal_alpha" not in result:
        logger.error("Failed to optimize form alpha")
        return {"processed": len(df), "saved": False}
    
    # Get current gameweek (use max gameweek from player stats)
    current_gameweek = db.query(PlayerGameweekStats.gameweek).filter(
        PlayerGameweekStats.season == season
    ).order_by(PlayerGameweekStats.gameweek.desc()).first()
    
    gameweek = current_gameweek[0] if current_gameweek else 1
    
    # Save to database
    existing = db.query(FormAlpha).filter(
        FormAlpha.gameweek == gameweek
    ).first()
    
    if existing:
        # Update existing entry
        existing.optimal_alpha = result["optimal_alpha"]
        existing.rmse = result["best_rmse"]
        existing.lookback_weeks = lookback_weeks
    else:
        # Create new entry
        new_entry = FormAlpha(
            gameweek=gameweek,
            optimal_alpha=result["optimal_alpha"],
            rmse=result["best_rmse"],
            lookback_weeks=lookback_weeks
        )
        db.add(new_entry)
    
    db.commit()
    
    logger.info(f"Form alpha optimized: alpha={result['optimal_alpha']:.4f}, "
                f"RMSE={result['best_rmse']:.4f}, gameweek={gameweek}")
    
    return {
        "processed": len(df),
        "saved": True,
        "optimal_alpha": result["optimal_alpha"],
        "rmse": result["best_rmse"],
        "gameweek": gameweek
    }


def sync_all_features(season: str = "2025-26", lookback_weeks: int = 5):
    """
    Run all feature synchronization steps.
    
    Args:
        season: Season identifier
        lookback_weeks: Number of weeks to consider for form alpha
    """
    db = SessionLocal()
    
    try:
        logger.info("=" * 60)
        logger.info("DATABASE FEATURE SYNCHRONIZATION")
        logger.info("=" * 60)
        logger.info(f"Season: {season}")
        logger.info(f"Form alpha lookback weeks: {lookback_weeks}")
        logger.info("")
        
        # Step 1: Calculate team xG/xGC stats
        team_stats_result = calculate_team_xg_stats(db, season=season)
        
        # Step 2: Fit FDR model
        fdr_result = sync_team_fdr(db, season=season)
        
        # Step 3: Optimize form alpha
        form_alpha_result = sync_form_alpha(db, season=season, lookback_weeks=lookback_weeks)
        
        # Summary
        logger.info("=" * 60)
        logger.info("SYNCHRONIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Team Stats: {team_stats_result.get('created', 0)} created, "
                   f"{team_stats_result.get('updated', 0)} updated")
        logger.info(f"FDR: {fdr_result.get('saved', 0)} teams saved")
        logger.info(f"Form Alpha: alpha={form_alpha_result.get('optimal_alpha', 0):.4f}, "
                   f"RMSE={form_alpha_result.get('rmse', 0):.4f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during feature synchronization: {str(e)}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()
        logger.info("Database session closed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync feature engineering data to database")
    parser.add_argument(
        "--season",
        type=str,
        default="2025-26",
        help="Season identifier (default: 2025-26)"
    )
    parser.add_argument(
        "--lookback-weeks",
        type=int,
        default=5,
        help="Number of weeks to consider for form alpha (default: 5)"
    )
    
    args = parser.parse_args()
    
    sync_all_features(season=args.season, lookback_weeks=args.lookback_weeks)
