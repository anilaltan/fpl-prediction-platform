"""
ETL Service for Loading Data to PostgreSQL
Handles async UPSERT operations for players and player_gameweek_stats tables
"""

from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert
import os
from dotenv import load_dotenv

from app.models import Player, PlayerGameweekStats, Team
from app.database import SessionLocal, engine

load_dotenv()

logger = logging.getLogger(__name__)


class ETLService:
    """
    ETL Service for loading FPL data into PostgreSQL.
    Implements async UPSERT operations for efficient data loading.
    """

    def __init__(self) -> None:
        # Use async engine for better performance
        database_url = os.getenv(
            "DATABASE_URL", "postgresql://fpl_user:fpl_password@localhost:5432/fpl_db"
        )
        # Convert to async URL
        if database_url.startswith("postgresql://"):
            async_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        else:
            async_url = database_url

        self.async_engine = create_async_engine(
            async_url, echo=False, pool_pre_ping=True
        )
        self.async_session_maker = async_sessionmaker(
            self.async_engine, class_=AsyncSession, expire_on_commit=False
        )

        # Sync engine for compatibility
        self.sync_engine = engine
        self.sync_session = SessionLocal

    async def upsert_player(
        self, player_data: Dict, session: Optional[AsyncSession] = None
    ) -> Player:
        """
        UPSERT a player record.
        Updates if exists (by fpl_id), inserts if new.

        Args:
            player_data: Player data dictionary
            session: Optional async session (creates new if not provided)

        Returns:
            Player model instance
        """
        should_close = False
        if session is None:
            session = self.async_session_maker()
            should_close = True

        try:
            fpl_id = player_data.get("fpl_id") or player_data.get("id")
            if not fpl_id:
                raise ValueError("Player data must include 'fpl_id' or 'id'")

            # Prepare data
            # FPL bootstrap extraction (`FPLAPIService.extract_players_from_bootstrap`) provides:
            # - position_id (int 1-4) and position (e.g. "GKP"/"DEF"/"MID"/"FWD")
            # It does NOT provide `element_type`, so using that key defaults everyone to MID.
            element_type = (
                player_data.get("element_type")
                or player_data.get("position_id")
                or player_data.get("position")
            )

            # Price handling:
            # - Raw FPL: `now_cost` is integer in 0.1 units (e.g. 45 => 4.5)
            # - Our extracted player dict already has `price` in millions (e.g. 4.5)
            raw_now_cost = player_data.get("now_cost")
            if raw_now_cost is not None:
                price_millions = float(raw_now_cost) / 10.0
            else:
                price_millions = float(player_data.get("price", 0.0))
                # Defensive: if a caller passes 0.1-units in `price`, normalize it.
                if price_millions > 100:
                    price_millions = price_millions / 10.0

            # Get team_id from player_data (FPL API provides 'team_id' or 'team')
            team_id = player_data.get("team_id") or player_data.get("team")
            if team_id:
                team_id = int(team_id)
            else:
                team_id = None

            # Get ownership percentage
            ownership = player_data.get("selected_by_percent")
            if ownership is not None:
                ownership = float(ownership)
            else:
                ownership = None

            player_dict = {
                "id": int(fpl_id),  # Use 'id' as primary key (FPL player ID)
                "name": player_data.get("web_name") or player_data.get("name", ""),
                "team_id": team_id,
                "position": self._normalize_position(element_type),
                "price": price_millions,
                "ownership": ownership,
            }

            # Use PostgreSQL UPSERT (ON CONFLICT)
            stmt = insert(Player).values(**player_dict)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],  # Use 'id' as primary key
                set_={
                    "name": stmt.excluded.name,
                    "team_id": stmt.excluded.team_id,
                    "position": stmt.excluded.position,
                    "price": stmt.excluded.price,
                    "ownership": stmt.excluded.ownership,
                    "updated_at": datetime.utcnow(),
                },
            )

            await session.execute(stmt)
            await session.commit()

            # Fetch the updated/inserted record
            from sqlalchemy import select

            result = await session.execute(select(Player).where(Player.id == fpl_id))
            player_row = result.scalar_one()

            logger.info(f"UPSERTed player: {player_dict['name']} (FPL ID: {fpl_id})")

            return player_row

        except Exception as e:
            await session.rollback()
            logger.error(
                f"Error UPSERTing player {player_data.get('fpl_id')}: {str(e)}"
            )
            raise
        finally:
            if should_close:
                await session.close()

    async def upsert_player_gameweek_stats(
        self, stats_data: Dict, session: Optional[AsyncSession] = None
    ) -> PlayerGameweekStats:
        """
        UPSERT player gameweek statistics.
        Updates if exists (by fpl_id + gameweek + season), inserts if new.

        Args:
            stats_data: Gameweek statistics dictionary
            session: Optional async session

        Returns:
            PlayerGameweekStats model instance
        """
        should_close = False
        if session is None:
            session = self.async_session_maker()
            should_close = True

        try:
            fpl_id = stats_data.get("fpl_id") or stats_data.get("element")
            gameweek = stats_data.get("gameweek") or stats_data.get("round")
            season = stats_data.get("season", "2025-26")

            if not fpl_id or not gameweek:
                raise ValueError("Stats data must include 'fpl_id' and 'gameweek'")

            # Prepare data
            stats_dict = {
                "fpl_id": int(fpl_id),
                "gameweek": int(gameweek),
                "season": str(season),
                "minutes": int(stats_data.get("minutes", 0)),
                "goals": int(
                    stats_data.get("goals_scored", stats_data.get("goals", 0))
                ),
                "assists": int(stats_data.get("assists", 0)),
                "clean_sheets": int(stats_data.get("clean_sheets", 0)),
                "goals_conceded": int(stats_data.get("goals_conceded", 0)),
                "own_goals": int(stats_data.get("own_goals", 0)),
                "penalties_saved": int(stats_data.get("penalties_saved", 0)),
                "penalties_missed": int(stats_data.get("penalties_missed", 0)),
                "yellow_cards": int(stats_data.get("yellow_cards", 0)),
                "red_cards": int(stats_data.get("red_cards", 0)),
                "saves": int(stats_data.get("saves", 0)),
                "bonus": int(stats_data.get("bonus", 0)),
                "bps": int(stats_data.get("bps", 0)),
                "total_points": int(
                    stats_data.get("total_points", stats_data.get("points", 0))
                ),
                "normalized_points": float(
                    stats_data.get(
                        "normalized_points", stats_data.get("total_points", 0)
                    )
                ),
                "xg": float(stats_data.get("expected_goals", stats_data.get("xg", 0))),
                "xa": float(
                    stats_data.get("expected_assists", stats_data.get("xa", 0))
                ),
                "xgi": float(
                    stats_data.get(
                        "expected_goal_involvements", stats_data.get("xgi", 0)
                    )
                ),
                "xgc": float(
                    stats_data.get("expected_goals_conceded", stats_data.get("xgc", 0))
                ),
                "npxg": float(
                    stats_data.get("npxg", stats_data.get("expected_goals", 0))
                ),
                "influence": float(stats_data.get("influence", 0)),
                "creativity": float(stats_data.get("creativity", 0)),
                "threat": float(stats_data.get("threat", 0)),
                "ict_index": float(stats_data.get("ict_index", 0)),
                "blocks": int(stats_data.get("blocks", 0)),
                "interventions": int(stats_data.get("interventions", 0)),
                "passes": int(
                    stats_data.get("passes", stats_data.get("successful_passes", 0))
                ),
                "defcon_floor_points": float(stats_data.get("defcon_floor_points", 0)),
                "was_home": bool(stats_data.get("was_home", True)),
                "opponent_team": int(stats_data.get("opponent_team", 0))
                if stats_data.get("opponent_team")
                else None,
                "team_score": int(stats_data.get("team_score", 0))
                if stats_data.get("team_score")
                else None,
                "opponent_score": int(stats_data.get("opponent_score", 0))
                if stats_data.get("opponent_score")
                else None,
            }

            # Use PostgreSQL UPSERT with composite unique key
            stmt = insert(PlayerGameweekStats).values(**stats_dict)
            stmt = stmt.on_conflict_do_update(
                index_elements=["fpl_id", "gameweek", "season"],
                set_={
                    key: stmt.excluded[key]
                    for key in stats_dict.keys()
                    if key not in ["fpl_id", "gameweek", "season"]
                },
            )

            await session.execute(stmt)
            await session.commit()

            logger.debug(
                f"UPSERTed stats: FPL ID {fpl_id}, GW {gameweek}, Season {season}"
            )

            return stats_dict

        except Exception as e:
            await session.rollback()
            logger.error(
                f"Error UPSERTing stats for FPL ID {stats_data.get('fpl_id')}, GW {stats_data.get('gameweek')}: {str(e)}"
            )
            raise
        finally:
            if should_close:
                await session.close()

    async def bulk_upsert_players(
        self, players_data: List[Dict], batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Bulk UPSERT multiple players.

        Args:
            players_data: List of player data dictionaries
            batch_size: Number of players to process per batch

        Returns:
            Dictionary with counts of inserted/updated players
        """
        session = self.async_session_maker()
        inserted = 0
        updated = 0
        errors = 0

        try:
            for i in range(0, len(players_data), batch_size):
                batch = players_data[i : i + batch_size]

                for player_data in batch:
                    try:
                        await self.upsert_player(player_data, session)
                        inserted += 1
                    except Exception as e:
                        logger.error(
                            f"Error in bulk upsert for player {player_data.get('fpl_id')}: {str(e)}"
                        )
                        errors += 1

                # Commit batch
                await session.commit()
                logger.info(
                    f"Processed batch {i // batch_size + 1}: {len(batch)} players"
                )

            return {
                "total": len(players_data),
                "inserted": inserted,
                "updated": updated,
                "errors": errors,
            }

        except Exception as e:
            await session.rollback()
            logger.error(f"Error in bulk upsert: {str(e)}")
            raise
        finally:
            await session.close()

    async def upsert_team(
        self, team_data: Dict, session: Optional[AsyncSession] = None
    ) -> Team:
        """
        UPSERT a team record.
        Updates if exists (by id), inserts if new.

        Args:
            team_data: Team data dictionary from FPL API
            session: Optional async session (creates new if not provided)

        Returns:
            Team model instance
        """
        should_close = False
        if session is None:
            session = self.async_session_maker()
            should_close = True

        try:
            team_id = team_data.get("id")
            if not team_id:
                raise ValueError("Team data must include 'id'")

            # Map FPL API fields to database schema
            # FPL provides: strength_attack_home, strength_attack_away
            # We store: strength_attack (average of home/away)
            strength_attack_home = team_data.get("strength_attack_home", 0)
            strength_attack_away = team_data.get("strength_attack_away", 0)
            strength_attack = (
                int((strength_attack_home + strength_attack_away) / 2)
                if (strength_attack_home + strength_attack_away) > 0
                else None
            )

            # FPL provides: strength_defence_home, strength_defence_away
            # We store: strength_defense (average of home/away)
            strength_defence_home = team_data.get("strength_defence_home", 0)
            strength_defence_away = team_data.get("strength_defence_away", 0)
            strength_defense = (
                int((strength_defence_home + strength_defence_away) / 2)
                if (strength_defence_home + strength_defence_away) > 0
                else None
            )

            # FPL provides: strength (overall)
            # We store: strength_overall
            strength_overall = team_data.get("strength")

            team_dict = {
                "id": int(team_id),
                "name": team_data.get("name", ""),
                "short_name": team_data.get("short_name", ""),
                "strength_attack": strength_attack,
                "strength_defense": strength_defense,
                "strength_overall": strength_overall,
            }

            # Use PostgreSQL UPSERT (ON CONFLICT)
            from sqlalchemy.dialects.postgresql import insert
            from app.models import Team
            from datetime import datetime

            stmt = insert(Team).values(**team_dict)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "name": stmt.excluded.name,
                    "short_name": stmt.excluded.short_name,
                    "strength_attack": stmt.excluded.strength_attack,
                    "strength_defense": stmt.excluded.strength_defense,
                    "strength_overall": stmt.excluded.strength_overall,
                    "updated_at": datetime.utcnow(),
                },
            )

            await session.execute(stmt)
            await session.commit()

            logger.info(f"UPSERTed team: {team_dict['name']} (ID: {team_id})")

            # Fetch the updated/inserted record
            from sqlalchemy import select

            result = await session.execute(select(Team).where(Team.id == team_id))
            team_row = result.scalar_one()

            return team_row

        except Exception as e:
            await session.rollback()
            logger.error(f"Error UPSERTing team {team_data.get('id')}: {str(e)}")
            raise
        finally:
            if should_close:
                await session.close()

    async def bulk_upsert_teams(
        self, teams_data: List[Dict], batch_size: int = 20
    ) -> Dict[str, int]:
        """
        Bulk UPSERT multiple teams.

        Args:
            teams_data: List of team data dictionaries
            batch_size: Number of teams to process per batch

        Returns:
            Dictionary with counts of inserted/updated teams
        """
        session = self.async_session_maker()
        inserted = 0
        updated = 0
        errors = 0

        try:
            for i in range(0, len(teams_data), batch_size):
                batch = teams_data[i : i + batch_size]

                for team_data in batch:
                    try:
                        await self.upsert_team(team_data, session)
                        inserted += 1
                    except Exception as e:
                        logger.error(
                            f"Error in bulk upsert for team {team_data.get('id')}: {str(e)}"
                        )
                        errors += 1

                # Commit batch
                await session.commit()
                logger.info(
                    f"Processed batch {i // batch_size + 1}: {len(batch)} teams"
                )

            return {
                "total": len(teams_data),
                "inserted": inserted,
                "updated": updated,
                "errors": errors,
            }

        except Exception as e:
            await session.rollback()
            logger.error(f"Error in bulk upsert: {str(e)}")
            raise
        finally:
            await session.close()

    async def bulk_upsert_gameweek_stats(
        self, stats_data: List[Dict], batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Bulk UPSERT multiple gameweek statistics.

        Args:
            stats_data: List of gameweek stats dictionaries
            batch_size: Number of records to process per batch

        Returns:
            Dictionary with counts of inserted/updated records
        """
        session = self.async_session_maker()
        inserted = 0
        updated = 0
        errors = 0

        try:
            for i in range(0, len(stats_data), batch_size):
                batch = stats_data[i : i + batch_size]

                for stats in batch:
                    try:
                        await self.upsert_player_gameweek_stats(stats, session)
                        inserted += 1
                    except Exception as e:
                        logger.error(f"Error in bulk upsert for stats: {str(e)}")
                        errors += 1

                # Commit batch
                await session.commit()
                logger.info(
                    f"Processed batch {i // batch_size + 1}: {len(batch)} stats records"
                )

            return {
                "total": len(stats_data),
                "inserted": inserted,
                "updated": updated,
                "errors": errors,
            }

        except Exception as e:
            await session.rollback()
            logger.error(f"Error in bulk upsert stats: {str(e)}")
            raise
        finally:
            await session.close()

    async def sync_from_fpl_api(
        self, fpl_api_service, gameweek: Optional[int] = None, season: str = "2025-26"
    ) -> Dict[str, int]:
        """
        Full ETL process: Fetch from FPL API and load to database.

        Args:
            fpl_api_service: FPLAPIService instance
            gameweek: Optional specific gameweek to sync
            season: Season string

        Returns:
            Dictionary with sync statistics
        """
        logger.info(
            f"Starting ETL sync for season {season}, gameweek {gameweek or 'all'}"
        )

        try:
            # 1. Fetch bootstrap data
            bootstrap = await fpl_api_service.get_bootstrap_data()
            players = fpl_api_service.extract_players_from_bootstrap(bootstrap)

            # 2. UPSERT players
            logger.info(f"UPSERTing {len(players)} players...")
            players_result = await self.bulk_upsert_players(players)

            # 3. Fetch and UPSERT gameweek stats
            stats_records = []

            if gameweek:
                # Single gameweek
                for player in players:
                    player_id = player.get("id")
                    if player_id:
                        try:
                            player_summary = await fpl_api_service.get_player_data(
                                player_id
                            )
                            history = fpl_api_service.extract_player_history(
                                player_summary
                            )

                            for match in history:
                                if match.get("round") == gameweek:
                                    match["fpl_id"] = player_id
                                    match["season"] = season
                                    stats_records.append(match)
                        except Exception as e:
                            logger.warning(
                                f"Error fetching stats for player {player_id}: {str(e)}"
                            )
                            continue
            else:
                # All gameweeks
                for player in players:
                    player_id = player.get("id")
                    if player_id:
                        try:
                            player_summary = await fpl_api_service.get_player_data(
                                player_id
                            )
                            history = fpl_api_service.extract_player_history(
                                player_summary
                            )

                            for match in history:
                                match["fpl_id"] = player_id
                                match["season"] = season
                                stats_records.append(match)
                        except Exception as e:
                            logger.warning(
                                f"Error fetching stats for player {player_id}: {str(e)}"
                            )
                            continue

            # 4. UPSERT stats
            logger.info(f"UPSERTing {len(stats_records)} gameweek stats records...")
            stats_result = await self.bulk_upsert_gameweek_stats(stats_records)

            return {
                "players": players_result,
                "gameweek_stats": stats_result,
                "total_players": len(players),
                "total_stats_records": len(stats_records),
            }

        except Exception as e:
            logger.error(f"Error in ETL sync: {str(e)}")
            raise

    def _normalize_position(self, element_type: Union[int, str]) -> str:
        """Convert FPL element_type to position string"""
        position_map = {
            1: "GK",
            2: "DEF",
            3: "MID",
            4: "FWD",
            # FPL bootstrap `singular_name_short` values
            "GKP": "GK",
            "GK": "GK",
            "DEF": "DEF",
            "MID": "MID",
            "FWD": "FWD",
        }
        return position_map.get(element_type, "MID")

    async def close(self) -> None:
        """Close async engine"""
        await self.async_engine.dispose()
