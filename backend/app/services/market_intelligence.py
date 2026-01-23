"""
Market Intelligence Service
Implements player ranking logic for ownership arbitrage analysis.
Calculates relative ranks for players based on xP and ownership percentage.
"""
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
import logging

from app.models import Player, Prediction, MarketIntelligence
from app.services.fpl import FPLAPIService

logger = logging.getLogger(__name__)


class MarketIntelligenceService:
    """
    Service for calculating player rankings and market intelligence metrics.
    Implements ranking logic for xP and ownership to identify arbitrage opportunities.
    """

    def __init__(self):
        """Initialize market intelligence service."""
        self._fpl_api = None

    @property
    def fpl_api(self) -> FPLAPIService:
        """Lazy-load FPL API service to avoid async initialization issues."""
        if self._fpl_api is None:
            self._fpl_api = FPLAPIService()
        return self._fpl_api

    def calculate_player_ranks(
        self,
        db: Session,
        gameweek: int,
        season: str = "2025-26",
        use_fpl_api_ownership: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate relative ranks for players based on xP and ownership percentage.

        Uses dense ranking (no gaps) where:
        - xP rank: descending (highest xP = rank 1)
        - Ownership rank: descending (highest ownership = rank 1)

        Args:
            db: Database session
            gameweek: Gameweek number
            season: Season string (default: "2025-26")
            use_fpl_api_ownership: If True, fetch ownership from FPL API;
                                   if False, use Player.ownership from database

        Returns:
            DataFrame with columns: player_id, name, xp, ownership, xp_rank, ownership_rank
        """
        # Fetch predictions for the gameweek
        predictions = (
            db.query(Prediction)
            .filter(and_(Prediction.gameweek == gameweek, Prediction.season == season))
            .all()
        )

        if not predictions:
            logger.warning(
                f"No predictions found for gameweek {gameweek}, season {season}"
            )
            return pd.DataFrame()

        # Fetch player data
        player_ids = [p.fpl_id for p in predictions]
        players = db.query(Player).filter(Player.id.in_(player_ids)).all()
        player_map = {p.id: p for p in players}

        # Get ownership data
        ownership_map = {}
        if use_fpl_api_ownership:
            # Fetch ownership for all players at once (more efficient)
            fpl_ids = [pred.fpl_id for pred in predictions]
            ownership_map = self._get_ownership_batch_from_api(fpl_ids)

        # Build DataFrame with predictions and player info
        data = []
        for pred in predictions:
            player = player_map.get(pred.fpl_id)
            if not player:
                continue

            # Get ownership
            if use_fpl_api_ownership and pred.fpl_id in ownership_map:
                ownership = ownership_map[pred.fpl_id]
            else:
                # Fallback to database
                ownership = float(player.ownership) if player.ownership else 0.0

            data.append(
                {
                    "player_id": pred.fpl_id,
                    "name": player.name,
                    "xp": float(pred.xp),
                    "ownership": ownership,
                }
            )

        if not data:
            logger.warning("No valid player data found for ranking")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Calculate dense ranks (descending: highest value = rank 1)
        # Use method='dense' to ensure no gaps in ranking
        df["xp_rank"] = df["xp"].rank(method="dense", ascending=False).astype(int)
        df["ownership_rank"] = (
            df["ownership"].rank(method="dense", ascending=False).astype(int)
        )

        # Sort by xP rank for easier inspection
        df = df.sort_values("xp_rank")

        logger.info(f"Calculated ranks for {len(df)} players for gameweek {gameweek}")

        return df

    async def _get_ownership_from_api_async(self, fpl_id: int) -> Optional[float]:
        """
        Get player ownership from FPL API (async version).

        Args:
            fpl_id: FPL player ID

        Returns:
            Ownership percentage (0-100) or None if not available
        """
        try:
            bootstrap = await self.fpl_api.get_bootstrap_data()
            elements = bootstrap.get("elements", [])
            for element in elements:
                if element.get("id") == fpl_id:
                    return float(element.get("selected_by_percent", 0.0))
            return None
        except Exception as e:
            logger.warning(
                f"Failed to fetch ownership from FPL API for player {fpl_id}: {str(e)}"
            )
            return None

    def _get_ownership_batch_from_api(self, fpl_ids: List[int]) -> Dict[int, float]:
        """
        Get ownership for multiple players from FPL API (sync wrapper).
        Note: This uses asyncio.run which should be used carefully in async contexts.

        Args:
            fpl_ids: List of FPL player IDs

        Returns:
            Dictionary mapping fpl_id to ownership percentage
        """
        import asyncio

        try:
            # Check if we're already in an event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, return empty dict and let caller handle async
                logger.warning(
                    "Cannot use sync API call in async context, returning empty ownership map"
                )
                return {}
        except RuntimeError:
            # No event loop, safe to create one
            pass

        try:
            bootstrap = asyncio.run(self.fpl_api.get_bootstrap_data())
            elements = bootstrap.get("elements", [])
            ownership_map = {}
            for element in elements:
                element_id = element.get("id")
                if element_id in fpl_ids:
                    ownership_map[element_id] = float(
                        element.get("selected_by_percent", 0.0)
                    )
            return ownership_map
        except Exception as e:
            logger.warning(f"Failed to fetch ownership from FPL API: {str(e)}")
            return {}

    def get_ranked_players(
        self,
        db: Session,
        gameweek: int,
        season: str = "2025-26",
        use_fpl_api_ownership: bool = True,
    ) -> List[Dict]:
        """
        Get ranked players as a list of dictionaries.

        Args:
            db: Database session
            gameweek: Gameweek number
            season: Season string (default: "2025-26")
            use_fpl_api_ownership: If True, fetch ownership from FPL API

        Returns:
            List of player dictionaries with ranking information
        """
        df = self.calculate_player_ranks(
            db=db,
            gameweek=gameweek,
            season=season,
            use_fpl_api_ownership=use_fpl_api_ownership,
        )

        if df.empty:
            return []

        return df.to_dict("records")

    def validate_ranks(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that ranks are correctly assigned (no gaps, proper ordering).

        Args:
            df: DataFrame with xp_rank and ownership_rank columns

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if df.empty:
            return True, []

        # Check for gaps in xP ranks
        xp_ranks = sorted(df["xp_rank"].unique())
        expected_xp_ranks = list(range(1, len(xp_ranks) + 1))
        if xp_ranks != expected_xp_ranks:
            errors.append(
                f"xP ranks have gaps: found {xp_ranks}, expected {expected_xp_ranks}"
            )

        # Check for gaps in ownership ranks
        ownership_ranks = sorted(df["ownership_rank"].unique())
        expected_ownership_ranks = list(range(1, len(ownership_ranks) + 1))
        if ownership_ranks != expected_ownership_ranks:
            errors.append(
                f"Ownership ranks have gaps: found {ownership_ranks}, expected {expected_ownership_ranks}"
            )

        # Check that rank 1 has highest value
        max_xp = df["xp"].max()
        rank_1_xp = df[df["xp_rank"] == 1]["xp"].values
        if len(rank_1_xp) > 0 and rank_1_xp[0] < max_xp:
            errors.append("Rank 1 xP is not the maximum value")

        max_ownership = df["ownership"].max()
        rank_1_ownership = df[df["ownership_rank"] == 1]["ownership"].values
        if len(rank_1_ownership) > 0 and rank_1_ownership[0] < max_ownership:
            errors.append("Rank 1 ownership is not the maximum value")

        return len(errors) == 0, errors

    def calculate_arbitrage_scores_and_categories(
        self,
        df: pd.DataFrame,
        overvalued_ownership_threshold: float = 30.0,
        differential_ownership_threshold: float = 10.0,
    ) -> pd.DataFrame:
        """
        Calculate arbitrage scores and assign market categories to players.

        Arbitrage Score Formula: (xp_rank - ownership_rank)
        - Negative score: High xP rank (low rank number), low ownership rank → Differential
        - Positive score: Low xP rank (high rank number), high ownership rank → Overvalued

        Categorization Logic:
        - Overvalued: ownership > 30% AND xP_rank is high (low xP)
        - Differential: ownership < 10% AND xP_rank is low (high xP)
        - Neutral: All other cases

        Args:
            df: DataFrame with columns: player_id, name, xp, ownership, xp_rank, ownership_rank
            overvalued_ownership_threshold: Ownership percentage threshold for 'Overvalued' (default: 30.0)
            differential_ownership_threshold: Ownership percentage threshold for 'Differential' (default: 10.0)

        Returns:
            DataFrame with additional columns: arbitrage_score, category
        """
        if df.empty:
            logger.warning("Cannot calculate arbitrage scores for empty DataFrame")
            return df

        # Validate required columns
        required_columns = ["xp_rank", "ownership_rank", "ownership"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

        # Calculate arbitrage score: (xp_rank - ownership_rank)
        df = df.copy()
        df["arbitrage_score"] = df["xp_rank"] - df["ownership_rank"]

        # Categorize players based on ownership and arbitrage score
        def categorize_player(row: pd.Series) -> str:
            """
            Categorize a player based on ownership percentage and arbitrage score.

            Note: xP_rank uses descending order (highest xP = rank 1, lowest xP = highest rank number)
            - Low rank number (e.g., 5) = high xP
            - High rank number (e.g., 200) = low xP

            Arbitrage Score = (xp_rank - ownership_rank):
            - Negative score: High xP (low xp_rank), low ownership (high ownership_rank) → Differential
            - Positive score: Low xP (high xp_rank), high ownership (low ownership_rank) → Overvalued

            Logic:
            - Overvalued: ownership > 30% AND positive arbitrage_score (low xP, high ownership)
            - Differential: ownership < 10% AND negative arbitrage_score (high xP, low ownership)
            - Neutral: All other cases
            """
            ownership = float(row["ownership"])
            arbitrage_score = float(row["arbitrage_score"])

            # Overvalued: High ownership (>30%) AND positive arbitrage score (low xP, high ownership)
            if ownership > overvalued_ownership_threshold and arbitrage_score > 0:
                return "Overvalued"

            # Differential: Low ownership (<10%) AND negative arbitrage score (high xP, low ownership)
            if ownership < differential_ownership_threshold and arbitrage_score < 0:
                return "Differential"

            # Neutral: All other cases
            return "Neutral"

        # Apply categorization
        df["category"] = df.apply(categorize_player, axis=1)

        # Log summary statistics
        category_counts = df["category"].value_counts()
        logger.info(f"Arbitrage categorization complete: {dict(category_counts)}")
        logger.info(
            f"Arbitrage score range: [{df['arbitrage_score'].min():.1f}, {df['arbitrage_score'].max():.1f}]"
        )

        return df

    def persist_market_intelligence(
        self, db: Session, df: pd.DataFrame, gameweek: int, season: str = "2025-26"
    ) -> Dict[str, int]:
        """
        Persist calculated market intelligence data to the database using bulk upsert.

        Performs a bulk upsert operation that:
        - Inserts new records for players not yet in the database for this gameweek
        - Updates existing records if they already exist (based on unique constraint)
        - Handles conflicts on (player_id, gameweek, season)

        Args:
            db: Database session
            df: DataFrame with columns: player_id, xp_rank, ownership_rank, arbitrage_score, category
            gameweek: Gameweek number
            season: Season string (default: "2025-26")

        Returns:
            Dictionary with 'inserted' and 'updated' counts
        """
        if df.empty:
            logger.warning("Cannot persist empty DataFrame")
            return {"inserted": 0, "updated": 0}

        # Validate required columns
        required_columns = [
            "player_id",
            "xp_rank",
            "ownership_rank",
            "arbitrage_score",
            "category",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

        try:
            # Convert DataFrame to list of dictionaries for bulk insert
            records = []
            for _, row in df.iterrows():
                records.append(
                    {
                        "player_id": int(row["player_id"]),
                        "gameweek": int(gameweek),
                        "season": season,
                        "xp_rank": int(row["xp_rank"]),
                        "ownership_rank": int(row["ownership_rank"]),
                        "arbitrage_score": float(row["arbitrage_score"]),
                        "category": str(row["category"]),
                    }
                )

            if not records:
                logger.warning("No records to persist")
                return {"inserted": 0, "updated": 0, "total": 0}

            # Count existing records before the upsert
            existing_count_before = (
                db.query(MarketIntelligence)
                .filter(
                    and_(
                        MarketIntelligence.gameweek == gameweek,
                        MarketIntelligence.season == season,
                    )
                )
                .count()
            )

            # Use PostgreSQL bulk upsert with ON CONFLICT DO UPDATE
            # Unique constraint: (player_id, gameweek, season)
            stmt = insert(MarketIntelligence).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_market_intelligence_player_gameweek_season",
                set_={
                    "xp_rank": stmt.excluded.xp_rank,
                    "ownership_rank": stmt.excluded.ownership_rank,
                    "arbitrage_score": stmt.excluded.arbitrage_score,
                    "category": stmt.excluded.category,
                    "updated_at": datetime.utcnow(),
                },
            )

            _result = db.execute(stmt)
            db.commit()

            # Count how many were inserted vs updated
            # After upsert, all records should exist
            total_after = len(records)
            inserted = max(0, total_after - existing_count_before)
            updated = min(existing_count_before, total_after)

            logger.info(
                f"Persisted market intelligence for gameweek {gameweek}, season {season}: "
                f"{inserted} inserted, {updated} updated (total: {total_after})"
            )

            return {"inserted": inserted, "updated": updated, "total": total_after}

        except Exception as e:
            db.rollback()
            logger.error(f"Error persisting market intelligence data: {str(e)}")
            raise

    def calculate_and_persist_market_intelligence(
        self,
        db: Session,
        gameweek: int,
        season: str = "2025-26",
        use_fpl_api_ownership: bool = True,
        overvalued_ownership_threshold: float = 30.0,
        differential_ownership_threshold: float = 10.0,
    ) -> Dict[str, any]:
        """
        Complete workflow: Calculate ranks, scores, categories, and persist to database.

        This is a convenience method that combines:
        1. calculate_player_ranks()
        2. calculate_arbitrage_scores_and_categories()
        3. persist_market_intelligence()

        Args:
            db: Database session
            gameweek: Gameweek number
            season: Season string (default: "2025-26")
            use_fpl_api_ownership: If True, fetch ownership from FPL API
            overvalued_ownership_threshold: Ownership threshold for 'Overvalued' (default: 30.0)
            differential_ownership_threshold: Ownership threshold for 'Differential' (default: 10.0)

        Returns:
            Dictionary with persistence results and summary statistics
        """
        logger.info(
            f"Starting market intelligence calculation and persistence for GW{gameweek}, {season}"
        )

        # Step 1: Calculate player ranks
        df = self.calculate_player_ranks(
            db=db,
            gameweek=gameweek,
            season=season,
            use_fpl_api_ownership=use_fpl_api_ownership,
        )

        if df.empty:
            logger.warning(
                f"No player data found for gameweek {gameweek}, season {season}"
            )
            return {
                "success": False,
                "message": "No player data found",
                "inserted": 0,
                "updated": 0,
                "total": 0,
            }

        # Step 2: Calculate arbitrage scores and categories
        df = self.calculate_arbitrage_scores_and_categories(
            df=df,
            overvalued_ownership_threshold=overvalued_ownership_threshold,
            differential_ownership_threshold=differential_ownership_threshold,
        )

        # Step 3: Persist to database
        persistence_result = self.persist_market_intelligence(
            db=db, df=df, gameweek=gameweek, season=season
        )

        # Add summary statistics
        category_counts = df["category"].value_counts().to_dict()

        result = {
            "success": True,
            "gameweek": gameweek,
            "season": season,
            "total_players": len(df),
            "category_distribution": category_counts,
            "arbitrage_score_range": {
                "min": float(df["arbitrage_score"].min()),
                "max": float(df["arbitrage_score"].max()),
            },
            **persistence_result,
        }

        logger.info(
            f"Market intelligence calculation and persistence complete: {result}"
        )

        return result

    def validate_market_intelligence_output(
        self,
        db: Session,
        gameweek: int,
        season: str = "2025-26",
        ownership_threshold: float = 30.0,
        min_previous_gameweeks: int = 2,
    ) -> Dict[str, any]:
        """
        Validate market intelligence output against the defined test strategy.

        Test Strategy:
        - Query market_intelligence table for players with >30% ownership and declining xP trends
        - Verify these players are correctly flagged as 'Overvalued'
        - Check consistency between Prediction table (xp) and market_intelligence table

        Args:
            db: Database session
            gameweek: Gameweek number to validate
            season: Season string (default: "2025-26")
            ownership_threshold: Ownership percentage threshold (default: 30.0)
            min_previous_gameweeks: Minimum number of previous gameweeks needed to detect declining trend (default: 2)

        Returns:
            Dictionary with validation results including:
            - is_valid: Boolean indicating if validation passed
            - errors: List of validation errors
            - warnings: List of validation warnings
            - overvalued_players: List of players with >30% ownership and declining xP
            - validation_summary: Summary statistics
        """
        errors = []
        warnings = []
        overvalued_players = []

        logger.info(
            f"Starting market intelligence validation for GW{gameweek}, {season}"
        )

        try:
            # Step 1: Get all market intelligence records for this gameweek
            mi_records = (
                db.query(MarketIntelligence)
                .filter(
                    and_(
                        MarketIntelligence.gameweek == gameweek,
                        MarketIntelligence.season == season,
                    )
                )
                .all()
            )

            if not mi_records:
                errors.append(
                    f"No market intelligence records found for gameweek {gameweek}, season {season}"
                )
                return {
                    "is_valid": False,
                    "errors": errors,
                    "warnings": warnings,
                    "overvalued_players": [],
                    "validation_summary": {},
                }

            logger.info(f"Found {len(mi_records)} market intelligence records")

            # Step 2: Get players with high ownership (>30%)
            high_ownership_players = (
                db.query(Player).filter(Player.ownership > ownership_threshold).all()
            )

            logger.info(
                f"Found {len(high_ownership_players)} players with ownership > {ownership_threshold}%"
            )

            # Step 3: For each high-ownership player, check:
            #   a) If they have market intelligence record
            #   b) If their xP is declining
            #   c) If they are correctly flagged as 'Overvalued'

            for player in high_ownership_players:
                # Find market intelligence record for this player
                mi_record = next(
                    (mi for mi in mi_records if mi.player_id == player.id), None
                )

                if not mi_record:
                    warnings.append(
                        f"Player {player.name} (ID: {player.id}) has {player.ownership}% ownership "
                        f"but no market intelligence record for GW{gameweek}"
                    )
                    continue

                # Get current gameweek xP from Prediction table
                current_prediction = (
                    db.query(Prediction)
                    .filter(
                        and_(
                            Prediction.fpl_id == player.id,
                            Prediction.gameweek == gameweek,
                            Prediction.season == season,
                        )
                    )
                    .first()
                )

                if not current_prediction:
                    warnings.append(
                        f"Player {player.name} (ID: {player.id}) has market intelligence record "
                        f"but no prediction for GW{gameweek}"
                    )
                    continue

                current_xp = float(current_prediction.xp)

                # Get previous gameweeks' xP to detect declining trend
                previous_predictions = (
                    db.query(Prediction)
                    .filter(
                        and_(
                            Prediction.fpl_id == player.id,
                            Prediction.gameweek < gameweek,
                            Prediction.season == season,
                        )
                    )
                    .order_by(Prediction.gameweek.desc())
                    .limit(min_previous_gameweeks)
                    .all()
                )

                # Check for declining xP trend
                has_declining_trend = False
                if len(previous_predictions) >= min_previous_gameweeks:
                    # Calculate average xP from previous gameweeks
                    previous_avg_xp = sum(
                        float(p.xp) for p in previous_predictions
                    ) / len(previous_predictions)
                    # Consider declining if current xP is at least 0.5 points lower than previous average
                    if current_xp < previous_avg_xp - 0.5:
                        has_declining_trend = True

                # Check if player should be flagged as Overvalued
                should_be_overvalued = float(
                    player.ownership
                ) > ownership_threshold and (
                    has_declining_trend or mi_record.arbitrage_score > 0
                )

                player_info = {
                    "player_id": player.id,
                    "name": player.name,
                    "ownership": float(player.ownership),
                    "current_xp": current_xp,
                    "xp_rank": mi_record.xp_rank,
                    "ownership_rank": mi_record.ownership_rank,
                    "arbitrage_score": float(mi_record.arbitrage_score),
                    "category": mi_record.category,
                    "has_declining_trend": has_declining_trend,
                    "should_be_overvalued": should_be_overvalued,
                }

                # If player has high ownership and declining xP, they should be Overvalued
                if (
                    float(player.ownership) > ownership_threshold
                    and has_declining_trend
                ):
                    overvalued_players.append(player_info)

                    # Validate category
                    if mi_record.category != "Overvalued":
                        errors.append(
                            f"Player {player.name} (ID: {player.id}) has {player.ownership}% ownership "
                            f"and declining xP trend (current: {current_xp:.2f}, previous avg: {previous_avg_xp:.2f}) "
                            f"but is categorized as '{mi_record.category}' instead of 'Overvalued'"
                        )
                    else:
                        logger.debug(
                            f"✓ Player {player.name} correctly flagged as Overvalued "
                            f"(ownership: {player.ownership}%, declining xP)"
                        )

                # Check consistency: arbitrage_score should match (xp_rank - ownership_rank)
                expected_score = mi_record.xp_rank - mi_record.ownership_rank
                if abs(float(mi_record.arbitrage_score) - expected_score) > 0.01:
                    errors.append(
                        f"Player {player.name} (ID: {player.id}): arbitrage_score inconsistency. "
                        f"Expected {expected_score}, got {mi_record.arbitrage_score}"
                    )

            # Step 4: Check for players flagged as Overvalued but shouldn't be
            overvalued_mi_records = [
                mi for mi in mi_records if mi.category == "Overvalued"
            ]

            for mi_record in overvalued_mi_records:
                player = (
                    db.query(Player).filter(Player.id == mi_record.player_id).first()
                )
                if not player:
                    warnings.append(
                        f"Market intelligence record references non-existent player_id {mi_record.player_id}"
                    )
                    continue

                if (
                    float(player.ownership) <= ownership_threshold
                    and mi_record.arbitrage_score <= 0
                ):
                    warnings.append(
                        f"Player {player.name} (ID: {player.id}) is flagged as 'Overvalued' "
                        f"but has ownership {player.ownership}% (<= {ownership_threshold}%) "
                        f"and arbitrage_score {mi_record.arbitrage_score} (<= 0)"
                    )

            # Step 5: Summary statistics
            category_counts = {}
            for mi_record in mi_records:
                category = mi_record.category
                category_counts[category] = category_counts.get(category, 0) + 1

            validation_summary = {
                "total_records": len(mi_records),
                "high_ownership_players": len(high_ownership_players),
                "overvalued_players_count": len(overvalued_players),
                "category_distribution": category_counts,
                "errors_count": len(errors),
                "warnings_count": len(warnings),
            }

            is_valid = len(errors) == 0

            result = {
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "overvalued_players": overvalued_players,
                "validation_summary": validation_summary,
            }

            if is_valid:
                logger.info(f"✓ Market intelligence validation passed for GW{gameweek}")
                logger.info(f"  Summary: {validation_summary}")
            else:
                logger.warning(
                    f"✗ Market intelligence validation failed for GW{gameweek}"
                )
                logger.warning(f"  Found {len(errors)} errors")

            return result

        except Exception as e:
            error_msg = f"Error during validation: {str(e)}"
            logger.error(error_msg)
            import traceback

            logger.error(traceback.format_exc())
            return {
                "is_valid": False,
                "errors": [error_msg],
                "warnings": warnings,
                "overvalued_players": [],
                "validation_summary": {},
            }
