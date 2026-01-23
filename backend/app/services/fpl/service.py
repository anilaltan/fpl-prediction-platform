"""
FPL API Service
Main orchestrator for FPL API integration using modular components.
"""
import os
import asyncio
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging

from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz, process

from .client import FPLHTTPClient
from .cache import InMemoryCache
from .processors import FPLDataProcessor
from .repository import FPLRepository
from app.services.entity_resolution import EntityResolutionService
from app.services.data_cleaning import DataCleaningService
from app.services.etl_service import ETLService

load_dotenv()
logger = logging.getLogger(__name__)


class FPLAPIService:
    """
    Enhanced FPL API Service with comprehensive data integration.

    Orchestrates data fetching, ID mapping, normalization, and database saving
    using modular components:
    - FPLHTTPClient: HTTP requests with rate limiting
    - InMemoryCache: TTL-based caching
    - FPLDataProcessor: Data transformation
    - FPLRepository: Database operations
    """

    # Cache TTL constants (in seconds)
    BOOTSTRAP_CACHE_TTL = 24 * 60 * 60  # 24 hours
    ELEMENT_SUMMARY_CACHE_TTL = 60 * 60  # 1 hour

    # External API URLs
    FBREF_BASE_URL = "https://fbref.com"

    def __init__(self, rate_limit_delay: float = 0.1) -> None:
        """
        Initialize FPL API service with modular components.

        Args:
            rate_limit_delay: Delay between requests in seconds (default: 0.1s)
        """
        self.email = os.getenv("FPL_EMAIL")
        self.password = os.getenv("FPL_PASSWORD")
        self.rate_limit_delay = rate_limit_delay

        # Initialize modular components
        self.client = FPLHTTPClient(rate_limit_delay=rate_limit_delay)
        self.cache = InMemoryCache()

        # Initialize supporting services
        self.data_cleaning = DataCleaningService()
        self.entity_resolution = EntityResolutionService()
        self.etl_service = ETLService()

        # Initialize processors and repository
        self.processor = FPLDataProcessor(data_cleaning=self.data_cleaning)
        self.repository = FPLRepository(etl_service=self.etl_service)

        # Load Master ID Map on initialization
        asyncio.create_task(self.entity_resolution.load_master_map())

    # ==================== FPL Official API Methods ====================

    async def get_bootstrap_data(self, use_cache: bool = True) -> Dict:
        """
        Fetch bootstrap-static data containing all players, teams, and fixtures.
        Cached for 24 hours per DefCon requirements.

        Args:
            use_cache: Whether to use cached data if available (default: True)

        Returns:
            Dictionary with keys:
            - elements: List of all players
            - teams: List of all teams
            - events: List of gameweeks
            - element_types: Position types
        """
        cache_key = "bootstrap-static"

        # Check cache first
        if use_cache:
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                logger.info("Using cached bootstrap data (24h TTL)")
                return cached_data

        # Fetch from API
        data = await self.client.get_bootstrap_data()

        # Cache the data for 24 hours
        await self.cache.set(cache_key, data, self.BOOTSTRAP_CACHE_TTL)

        logger.info(
            f"Fetched bootstrap data: {len(data.get('elements', []))} players, {len(data.get('teams', []))} teams"
        )
        return data

    async def get_current_gameweek(self) -> Optional[int]:
        """
        Get the current active gameweek from FPL API.

        Returns:
            Current gameweek number, or None if not available
        """
        try:
            bootstrap = await self.get_bootstrap_data()
            return self.processor.extract_gameweek_from_events(bootstrap, is_next=False)
        except Exception as e:
            logger.error(f"Failed to get current gameweek: {str(e)}")
            return None

    async def get_next_gameweek(self) -> Optional[int]:
        """
        Get the next upcoming gameweek from FPL API.

        Returns:
            Next gameweek number, or None if not available
        """
        try:
            bootstrap = await self.get_bootstrap_data()
            return self.processor.extract_gameweek_from_events(bootstrap, is_next=True)
        except Exception as e:
            logger.error(f"Failed to get next gameweek: {str(e)}")
            return None

    def extract_players_from_bootstrap(self, bootstrap_data: Dict) -> List[Dict]:
        """
        Extract and structure player data from bootstrap-static.

        Args:
            bootstrap_data: Raw bootstrap data

        Returns:
            List of structured player dictionaries
        """
        return self.processor.extract_players_from_bootstrap(bootstrap_data)

    def extract_teams_from_bootstrap(self, bootstrap_data: Dict) -> List[Dict]:
        """
        Extract team data from bootstrap-static.

        Args:
            bootstrap_data: Raw bootstrap data

        Returns:
            List of structured team dictionaries
        """
        return self.processor.extract_teams_from_bootstrap(bootstrap_data)

    async def get_player_data(self, player_id: int, use_cache: bool = True) -> Dict:
        """
        Fetch detailed data for a specific player from element-summary endpoint.
        Cached for 1 hour per DefCon requirements.

        Args:
            player_id: FPL player ID
            use_cache: Whether to use cached data if available (default: True)

        Returns:
            Dictionary containing history, history_past, fixtures, explain
        """
        cache_key = f"element-summary-{player_id}"

        # Check cache first
        if use_cache:
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Using cached player data for {player_id} (1h TTL)")
                return cached_data

        # Fetch from API
        data = await self.client.get_player_data(player_id)

        # Cache the data for 1 hour
        await self.cache.set(cache_key, data, self.ELEMENT_SUMMARY_CACHE_TTL)

        return data

    def extract_player_history(self, player_summary: Dict) -> List[Dict]:
        """
        Extract match history statistics from element-summary.

        Args:
            player_summary: Player summary data from FPL API

        Returns:
            List of match statistics with normalized points for DGW
        """
        return self.processor.extract_player_history(player_summary)

    async def get_fixtures(
        self, gameweek: Optional[int] = None, future_only: bool = False
    ) -> List[Dict]:
        """
        Fetch fixtures data with rate limiting.

        Args:
            gameweek: Optional gameweek filter
            future_only: If True, only return unfinished fixtures

        Returns:
            List of fixture dictionaries
        """
        # Fetch all fixtures
        fixtures_data = await self.client.get_fixtures(gameweek)
        fixtures = (
            fixtures_data
            if isinstance(fixtures_data, list)
            else fixtures_data.get("fixtures", [])
        )

        # Filter by gameweek if specified
        if gameweek:
            fixtures = [f for f in fixtures if f.get("event") == gameweek]

        # Filter out finished fixtures if future_only is True
        if future_only:
            fixtures = [f for f in fixtures if not f.get("finished", False)]

        logger.info(
            f"Fetched {len(fixtures)} fixtures (gameweek={gameweek}, future_only={future_only})"
        )
        return fixtures

    def extract_fixtures_with_difficulty(
        self, fixtures: List[Dict], teams_data: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Extract and structure fixtures with difficulty ratings.

        Args:
            fixtures: List of raw fixture dictionaries from FPL API
            teams_data: Optional list of team data for strength lookup

        Returns:
            List of structured fixture dictionaries with difficulty ratings
        """
        team_lookup = {}
        if teams_data:
            team_lookup = {team["id"]: team for team in teams_data}

        structured_fixtures = []
        for fixture in fixtures:
            home_team_id = fixture.get("team_h")
            away_team_id = fixture.get("team_a")

            home_team = team_lookup.get(home_team_id, {})
            away_team = team_lookup.get(away_team_id, {})

            # Get strength values, default to 1000 if not found or 0
            home_strength = home_team.get("strength", 0) or 1000
            away_strength = away_team.get("strength", 0) or 1000

            # Ensure strength is at least 100 to avoid division issues
            home_strength = max(home_strength, 100)
            away_strength = max(away_strength, 100)

            # Calculate difficulty: opponent's strength determines difficulty
            # Higher opponent strength = higher difficulty (1-5 scale)
            # Normalize: strength ranges roughly 1000-1500, map to 1-5
            home_difficulty = min(5, max(1, int(round((away_strength / 1000.0) * 5))))
            away_difficulty = min(5, max(1, int(round((home_strength / 1000.0) * 5))))

            fixture_data = {
                "id": fixture.get("id"),
                "gameweek": fixture.get("event"),
                "kickoff_time": fixture.get("kickoff_time"),
                "home_team_id": home_team_id,
                "home_team_name": home_team.get("name", ""),
                "away_team_id": away_team_id,
                "away_team_name": away_team.get("name", ""),
                "home_difficulty": home_difficulty,
                "away_difficulty": away_difficulty,
                "home_score": fixture.get("team_h_score"),
                "away_score": fixture.get("team_a_score"),
                "finished": fixture.get("finished", False),
                "home_strength": home_strength,
                "away_strength": away_strength,
            }
            structured_fixtures.append(fixture_data)

        return structured_fixtures

    # ==================== FBref DefCon Metrics ====================

    async def fetch_fbref_defcon_metrics(
        self, season: str = "2025-2026", player_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch DefCon metrics (tackles, interceptions, blocks) from FBref.
        Based on worldfootballR approach for FBref data extraction.

        Args:
            season: Season string (e.g., "2025-2026")
            player_name: Optional player name filter

        Returns:
            List of player defensive statistics with DefCon metrics
        """
        try:
            # FBref URL structure (similar to worldfootballR)
            url = f"{self.FBREF_BASE_URL}/en/comps/9/{season}/{season}-Premier-League-Stats"

            # Use client's HTTP client for the request
            import httpx

            async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=True
            ) as http_client:
                response = await http_client.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                defensive_stats = []

                # Find defensive stats table
                # FBref uses table IDs like "stats_standard_defense" or "stats_defense"
                tables = soup.find_all(
                    "table", {"id": re.compile(r".*defense.*", re.I)}
                )

                if not tables:
                    # Fallback to standard stats table
                    tables = soup.find_all("table", {"class": "stats_table"})

                for table in tables:
                    rows = table.find_all("tr")[1:]  # Skip header

                    for row in rows:
                        cells = row.find_all(["td", "th"])
                        if len(cells) < 5:
                            continue

                        try:
                            name = cells[0].get_text(strip=True)

                            stats = {
                                "name": name,
                                "team": cells[1].get_text(strip=True)
                                if len(cells) > 1
                                else "",
                                "position": cells[2].get_text(strip=True)
                                if len(cells) > 2
                                else "",
                                "games": self._parse_int(cells[3].get_text(strip=True))
                                if len(cells) > 3
                                else 0,
                                "minutes": self._parse_int(
                                    cells[4].get_text(strip=True)
                                )
                                if len(cells) > 4
                                else 0,
                            }

                            # Find defensive metrics columns
                            header_row = table.find("tr")
                            if header_row:
                                headers = [
                                    h.get_text(strip=True).lower()
                                    for h in header_row.find_all(["th", "td"])
                                ]

                                # Blocks
                                blocks_idx = self._find_column_index(
                                    headers, ["blocks", "blocked"]
                                )
                                if blocks_idx is not None:
                                    stats["blocks"] = self._parse_int(
                                        cells[blocks_idx].get_text(strip=True)
                                    )

                                # Tackles
                                tackles_idx = self._find_column_index(
                                    headers, ["tackles", "tkl", "tackles_won"]
                                )
                                if tackles_idx is not None:
                                    stats["tackles"] = self._parse_int(
                                        cells[tackles_idx].get_text(strip=True)
                                    )

                                # Interceptions
                                int_idx = self._find_column_index(
                                    headers, ["interceptions", "int", "intercepted"]
                                )
                                if int_idx is not None:
                                    stats["interceptions"] = self._parse_int(
                                        cells[int_idx].get_text(strip=True)
                                    )

                                # Passes
                                passes_idx = self._find_column_index(
                                    headers, ["passes", "pass", "passes_completed"]
                                )
                                if passes_idx is not None:
                                    stats["passes"] = self._parse_int(
                                        cells[passes_idx].get_text(strip=True)
                                    )

                            # Calculate per-90 and interventions
                            minutes = stats.get("minutes", 0)
                            if minutes > 0:
                                stats["blocks_per_90"] = (
                                    stats.get("blocks", 0) / minutes
                                ) * 90
                                stats["tackles_per_90"] = (
                                    stats.get("tackles", 0) / minutes
                                ) * 90
                                stats["interceptions_per_90"] = (
                                    stats.get("interceptions", 0) / minutes
                                ) * 90
                                stats["passes_per_90"] = (
                                    stats.get("passes", 0) / minutes
                                ) * 90

                                # Interventions = tackles + interceptions (2025/26 DefCon rule)
                                interventions = stats.get("tackles", 0) + stats.get(
                                    "interceptions", 0
                                )
                                stats["interventions"] = interventions
                                stats["interventions_per_90"] = (
                                    interventions / minutes
                                ) * 90
                            else:
                                stats["blocks_per_90"] = 0.0
                                stats["tackles_per_90"] = 0.0
                                stats["interceptions_per_90"] = 0.0
                                stats["passes_per_90"] = 0.0
                                stats["interventions"] = 0
                                stats["interventions_per_90"] = 0.0

                            defensive_stats.append(stats)

                        except Exception as e:
                            logger.warning(f"Error parsing FBref row: {str(e)}")
                            continue

                if player_name:
                    defensive_stats = [
                        p
                        for p in defensive_stats
                        if player_name.lower() in p["name"].lower()
                    ]

                return defensive_stats

        except Exception as e:
            logger.error(f"Error fetching FBref DefCon metrics: {str(e)}")
            return []

    def _parse_int(self, value: str) -> int:
        """
        Parse integer from string, handling commas and formatting.

        Args:
            value: String value to parse

        Returns:
            Parsed integer, or 0 if parsing fails
        """
        try:
            return int(re.sub(r"[^\d]", "", str(value)))
        except Exception:
            return 0

    def _find_column_index(
        self, headers: List[str], keywords: List[str]
    ) -> Optional[int]:
        """
        Find column index by header keywords.

        Args:
            headers: List of header strings
            keywords: List of keywords to search for

        Returns:
            Column index if found, None otherwise
        """
        for idx, header in enumerate(headers):
            if any(keyword.lower() in header.lower() for keyword in keywords):
                return idx
        return None

    # ==================== ID Mapping with FuzzyWuzzy ====================

    async def map_players_with_fbref(
        self,
        fpl_players: List[Dict],
        fbref_players: List[Dict],
        use_fuzzy: bool = True,
        threshold: int = 80,
    ) -> Dict[int, Dict]:
        """
        Map FPL players to FBref players using FPL-ID-Map and FuzzyWuzzy.

        Args:
            fpl_players: List of FPL player dictionaries
            fbref_players: List of FBref player dictionaries
            use_fuzzy: Whether to use FuzzyWuzzy for unmatched players
            threshold: Minimum similarity score (0-100) for fuzzy matching

        Returns:
            Dictionary mapping FPL ID to FBref data
        """
        mappings = {}

        # First, try Master ID Map
        for fpl_player in fpl_players:
            fpl_id = fpl_player.get("fpl_id") or fpl_player.get("id")
            fpl_name = fpl_player.get("web_name") or fpl_player.get("full_name", "")

            # Try entity resolution first
            resolution = self.entity_resolution.resolve_player_entity(
                fpl_id=fpl_id, fpl_name=fpl_name, fpl_team=fpl_player.get("team_name")
            )

            if resolution.get("matched") and resolution.get("fbref_id"):
                # Found in Master Map
                fbref_id = resolution.get("fbref_id")
                # Find matching FBref player
                fbref_player = next(
                    (
                        p
                        for p in fbref_players
                        if str(fbref_id) in p.get("name", "")
                        or fbref_id == p.get("fbref_id")
                    ),
                    None,
                )
                if fbref_player:
                    mappings[fpl_id] = {
                        "fbref_data": fbref_player,
                        "match_method": "master_map",
                        "confidence": 1.0,
                    }
                    continue

            # If not found, try FuzzyWuzzy matching
            if use_fuzzy:
                fbref_names = [p.get("name", "") for p in fbref_players]

                # Use FuzzyWuzzy to find best match
                best_match = process.extractOne(
                    fpl_name, fbref_names, scorer=fuzz.token_sort_ratio
                )

                if best_match and best_match[1] >= threshold:
                    matched_name = best_match[0]
                    fbref_player = next(
                        (p for p in fbref_players if p.get("name") == matched_name),
                        None,
                    )
                    if fbref_player:
                        mappings[fpl_id] = {
                            "fbref_data": fbref_player,
                            "match_method": "fuzzywuzzy",
                            "confidence": best_match[1] / 100.0,
                            "matched_name": matched_name,
                        }
                        logger.info(
                            f"Fuzzy matched: {fpl_name} -> {matched_name} (score: {best_match[1]})"
                        )

        return mappings

    # ==================== Comprehensive Data Fetching ====================

    async def fetch_comprehensive_player_data(
        self,
        player_id: int,
        season: str = "2025-26",
        include_fbref: bool = True,
        normalize_dgw: bool = True,
    ) -> Dict:
        """
        Fetch comprehensive player data from FPL API and FBref.
        Includes ID mapping, normalization, and DefCon metrics.

        Args:
            player_id: FPL player ID
            season: Season string
            include_fbref: Whether to fetch FBref DefCon metrics
            normalize_dgw: Whether to normalize DGW points

        Returns:
            Comprehensive player data dictionary
        """
        try:
            # 1. Fetch FPL data
            player_summary = await self.get_player_data(player_id)
            bootstrap = await self.get_bootstrap_data()
            players = self.extract_players_from_bootstrap(bootstrap)
            fpl_player = next((p for p in players if p["id"] == player_id), None)

            if not fpl_player:
                raise ValueError(f"Player {player_id} not found in bootstrap data")

            # 2. Extract history with DGW normalization
            history = self.extract_player_history(player_summary)

            # 3. Fetch FBref DefCon metrics if requested
            fbref_data = None
            if include_fbref:
                fbref_players = await self.fetch_fbref_defcon_metrics(season)
                mappings = await self.map_players_with_fbref(
                    [fpl_player], fbref_players, use_fuzzy=True, threshold=80
                )

                if player_id in mappings:
                    fbref_data = mappings[player_id]["fbref_data"]
                    # Add DefCon metrics to player data
                    fpl_player.update(
                        {
                            "fbref_blocks": fbref_data.get("blocks", 0),
                            "fbref_blocks_per_90": fbref_data.get("blocks_per_90", 0.0),
                            "fbref_tackles": fbref_data.get("tackles", 0),
                            "fbref_interceptions": fbref_data.get("interceptions", 0),
                            "fbref_interventions": fbref_data.get("interventions", 0),
                            "fbref_interventions_per_90": fbref_data.get(
                                "interventions_per_90", 0.0
                            ),
                            "fbref_passes": fbref_data.get("passes", 0),
                            "fbref_passes_per_90": fbref_data.get("passes_per_90", 0.0),
                        }
                    )

            # 4. Calculate DefCon floor points
            position = fpl_player.get("position", "MID")
            defcon_metrics = self.data_cleaning.get_defcon_metrics(fpl_player, position)
            fpl_player["defcon_floor_points"] = defcon_metrics["floor_points"]

            # 5. Combine all data
            comprehensive_data = {
                "fpl_data": fpl_player,
                "history": history,
                "history_past": player_summary.get("history_past", []),
                "fixtures": player_summary.get("fixtures", []),
                "fbref_data": fbref_data,
                "defcon_metrics": defcon_metrics,
            }

            return comprehensive_data

        except Exception as e:
            logger.error(
                f"Error fetching comprehensive data for player {player_id}: {str(e)}"
            )
            raise

    # ==================== PostgreSQL Integration ====================

    async def save_player_gameweek_stats(
        self,
        player_id: int,
        gameweek: int,
        season: str = "2025-26",
        include_fbref: bool = True,
    ) -> Dict:
        """
        Fetch player data and save to PostgreSQL player_gameweek_stats table.

        Args:
            player_id: FPL player ID
            gameweek: Gameweek number
            season: Season string
            include_fbref: Whether to include FBref DefCon metrics

        Returns:
            Dictionary with save status
        """
        try:
            # Fetch comprehensive data
            comprehensive_data = await self.fetch_comprehensive_player_data(
                player_id, season, include_fbref=include_fbref, normalize_dgw=True
            )

            # Find gameweek stats
            history = comprehensive_data.get("history", [])
            gameweek_stats = next(
                (h for h in history if h.get("gameweek") == gameweek), None
            )

            if not gameweek_stats:
                return {
                    "status": "not_found",
                    "message": f"No stats found for player {player_id} in gameweek {gameweek}",
                }

            # Prepare stats for database with FBref data
            _fpl_data = comprehensive_data.get("fpl_data", {})
            fbref_data = comprehensive_data.get("fbref_data", {})

            stats_for_db = {
                **gameweek_stats,
                "blocks": fbref_data.get("blocks", 0) if fbref_data else 0,
                "interventions": fbref_data.get("interventions", 0)
                if fbref_data
                else 0,
                "passes": fbref_data.get("passes", 0) if fbref_data else 0,
                "defcon_floor_points": comprehensive_data.get("defcon_metrics", {}).get(
                    "floor_points", 0.0
                ),
            }

            # Save to database using repository
            result = await self.repository.save_player_gameweek_stats(
                player_id=player_id,
                gameweek=gameweek,
                season=season,
                stats_data=stats_for_db,
            )

            return result

        except Exception as e:
            logger.error(
                f"Error saving gameweek stats for player {player_id}, GW {gameweek}: {str(e)}"
            )
            raise

    async def bulk_save_gameweek_stats(
        self, gameweek: int, season: str = "2025-26", max_players: Optional[int] = None
    ) -> Dict:
        """
        Bulk fetch and save all players' gameweek stats to PostgreSQL.

        Args:
            gameweek: Gameweek number
            season: Season string
            max_players: Maximum number of players to process (for testing)

        Returns:
            Dictionary with bulk save results
        """
        try:
            # Get all players
            bootstrap = await self.get_bootstrap_data()
            players = self.extract_players_from_bootstrap(bootstrap)

            if max_players:
                players = players[:max_players]

            # Prepare stats for bulk save (Task 3.2: Process one at a time to reduce memory footprint)
            players_stats = []
            for player in players:
                try:
                    player_summary = await self.get_player_data(
                        player["id"], use_cache=True
                    )
                    history = self.extract_player_history(player_summary)
                    gameweek_stats = next(
                        (h for h in history if h.get("gameweek") == gameweek), None
                    )

                    if gameweek_stats:
                        players_stats.append(
                            {"player_id": player["id"], "stats_data": gameweek_stats}
                        )

                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)

                    # Memory management: periodically collect garbage for large batches
                    if len(players_stats) % 50 == 0:
                        import gc

                        gc.collect()

                except Exception as e:
                    logger.warning(f"Error processing player {player['id']}: {str(e)}")
                    continue

            # Bulk save using repository
            return await self.repository.bulk_save_gameweek_stats(
                gameweek=gameweek, season=season, players_stats=players_stats
            )

        except Exception as e:
            logger.error(f"Error in bulk save: {str(e)}")
            raise

    async def clear_cache(self, key: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            key: Optional cache key to clear. If None, clears all cache.
        """
        await self.cache.clear(key)

    async def close(self) -> None:
        """Close HTTP client and services."""
        await self.client.close()
        await self.repository.close()
        await self.entity_resolution.close()
