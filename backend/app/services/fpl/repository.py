"""
FPL Repository Pattern
Database operations for FPL data using Repository Pattern.
"""
from typing import Dict, List, Optional
import logging

from app.services.etl_service import ETLService

logger = logging.getLogger(__name__)


class FPLRepository:
    """
    Repository for FPL database operations.

    Implements Repository Pattern to separate database logic
    from business logic. All database operations go through this class.
    """

    def __init__(self, etl_service: Optional[ETLService] = None) -> None:
        """
        Initialize repository.

        Args:
            etl_service: Optional ETLService instance
        """
        self.etl_service = etl_service or ETLService()

    async def save_player_gameweek_stats(
        self, player_id: int, gameweek: int, season: str, stats_data: Dict
    ) -> Dict:
        """
        Save player gameweek statistics to database.

        Args:
            player_id: FPL player ID
            gameweek: Gameweek number
            season: Season string (e.g., "2025-26")
            stats_data: Dictionary containing all gameweek statistics

        Returns:
            Dictionary with save status and saved data
        """
        try:
            # Prepare stats for database
            stats_for_db = {
                "fpl_id": player_id,
                "gameweek": gameweek,
                "season": season,
                "minutes": stats_data.get("minutes", 0),
                "goals": stats_data.get("goals_scored", 0),
                "assists": stats_data.get("assists", 0),
                "clean_sheets": stats_data.get("clean_sheets", 0),
                "goals_conceded": stats_data.get("goals_conceded", 0),
                "saves": stats_data.get("saves", 0),
                "bonus": stats_data.get("bonus", 0),
                "bps": stats_data.get("bps", 0),
                "total_points": stats_data.get("points", 0),
                "normalized_points": stats_data.get("normalized_points", 0.0),
                "xg": stats_data.get("expected_goals", 0.0),
                "xa": stats_data.get("expected_assists", 0.0),
                "xgi": stats_data.get("expected_goal_involvements", 0.0),
                "xgc": stats_data.get("expected_goals_conceded", 0.0),
                "influence": stats_data.get("influence", 0.0),
                "creativity": stats_data.get("creativity", 0.0),
                "threat": stats_data.get("threat", 0.0),
                "ict_index": stats_data.get("ict_index", 0.0),
                "blocks": stats_data.get("blocks", 0),
                "interventions": stats_data.get("interventions", 0),
                "passes": stats_data.get("passes", 0),
                "defcon_floor_points": stats_data.get("defcon_floor_points", 0.0),
                "was_home": stats_data.get("was_home", True),
                "opponent_team": stats_data.get("opponent_team"),
            }

            # Save to database using ETL service
            await self.etl_service.upsert_player_gameweek_stats(stats_for_db)

            return {
                "status": "success",
                "player_id": player_id,
                "gameweek": gameweek,
                "stats": stats_for_db,
            }

        except Exception as e:
            logger.error(
                f"Error saving gameweek stats for player {player_id}, GW {gameweek}: {str(e)}"
            )
            raise

    async def bulk_save_gameweek_stats(
        self, gameweek: int, season: str, players_stats: List[Dict]
    ) -> Dict:
        """
        Bulk save multiple players' gameweek statistics.

        Args:
            gameweek: Gameweek number
            season: Season string
            players_stats: List of dictionaries, each containing player_id and stats_data

        Returns:
            Dictionary with bulk save results
        """
        results = {
            "total_players": len(players_stats),
            "saved": 0,
            "errors": 0,
            "errors_list": [],
        }

        for player_stats in players_stats:
            try:
                player_id = player_stats["player_id"]
                stats_data = player_stats["stats_data"]

                await self.save_player_gameweek_stats(
                    player_id, gameweek, season, stats_data
                )
                results["saved"] += 1

            except Exception as e:
                results["errors"] += 1
                results["errors_list"].append(
                    {
                        "player_id": player_stats.get("player_id", "unknown"),
                        "error": str(e),
                    }
                )
                logger.warning(
                    f"Error saving stats for player {player_stats.get('player_id')}: {str(e)}"
                )

        return results

    async def close(self) -> None:
        """Close repository and underlying services."""
        await self.etl_service.close()
