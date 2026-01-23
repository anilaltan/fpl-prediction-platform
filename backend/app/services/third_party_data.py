"""
Third-Party Data Integration Service
Integrates data from Understat and FBref for advanced metrics:
- Understat: xG, xA, NPxG (Non-Penalty Expected Goals)
- FBref: DefCon metrics (blocks, interventions, passes) for 2025/26 rules
"""
import httpx
import asyncio
import re
from typing import Dict, List, Optional
import logging
from bs4 import BeautifulSoup
import json
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class PlayerNameMatcher:
    """
    Entity Resolution: Maps player names across different data sources.
    Handles name variations (e.g., "Bruno Fernandes" vs "Bruno F.")
    Note: For production use, prefer EntityResolutionService with Master ID Map.
    """

    def __init__(self):
        self.name_mappings: Dict[str, Dict[str, str]] = {}

    def normalize_name(self, name: str) -> str:
        """
        Normalize player name for matching.
        Removes accents, converts to lowercase, handles common variations.
        """
        # Remove accents and special characters
        name = name.lower().strip()
        name = re.sub(r"[àáâãäå]", "a", name)
        name = re.sub(r"[èéêë]", "e", name)
        name = re.sub(r"[ìíîï]", "i", name)
        name = re.sub(r"[òóôõö]", "o", name)
        name = re.sub(r"[ùúûü]", "u", name)
        name = re.sub(r"[ç]", "c", name)
        name = re.sub(r"[ñ]", "n", name)

        # Remove common suffixes
        name = re.sub(r"\s+[jr]\.?\s*$", "", name)  # Jr., Jr, jr
        name = re.sub(r"\s+[ivx]+$", "", name)  # Roman numerals

        return name.strip()

    def match_player(
        self, fpl_name: str, external_name: str, team: Optional[str] = None
    ) -> float:
        """
        Calculate similarity score between two player names.

        Returns:
            Similarity score (0-1)
        """
        fpl_norm = self.normalize_name(fpl_name)
        ext_norm = self.normalize_name(external_name)

        # Exact match
        if fpl_norm == ext_norm:
            return 1.0

        # Check if one contains the other
        if fpl_norm in ext_norm or ext_norm in fpl_norm:
            return 0.9

        # Split into words and check overlap
        fpl_words = set(fpl_norm.split())
        ext_words = set(ext_norm.split())

        if not fpl_words or not ext_words:
            return 0.0

        # Jaccard similarity
        intersection = len(fpl_words & ext_words)
        union = len(fpl_words | ext_words)

        similarity = intersection / union if union > 0 else 0.0

        return similarity

    def create_mapping(
        self,
        fpl_players: List[Dict],
        external_players: List[Dict],
        threshold: float = 0.7,
    ) -> Dict[int, Dict]:
        """
        Create mapping between FPL players and external data sources.

        Returns:
            Dictionary mapping FPL ID to external player data
        """
        mappings = {}

        for fpl_player in fpl_players:
            fpl_id = fpl_player.get("id")
            fpl_name = fpl_player.get("web_name", "") or fpl_player.get("name", "")
            fpl_team = fpl_player.get("team_name", "")

            best_match = None
            best_score = 0.0

            for ext_player in external_players:
                ext_name = ext_player.get("name", "")
                ext_team = ext_player.get("team", "")

                # Team match bonus
                team_bonus = 0.1 if fpl_team.lower() == ext_team.lower() else 0.0

                score = self.match_player(fpl_name, ext_name) + team_bonus

                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = ext_player

            if best_match:
                mappings[fpl_id] = {
                    "external_data": best_match,
                    "similarity": best_score,
                    "fpl_name": fpl_name,
                    "external_name": best_match.get("name", ""),
                }

        return mappings


class UnderstatService:
    """
    Service for fetching data from Understat.com
    Provides xG, xA, NPxG (Non-Penalty Expected Goals) metrics
    """

    BASE_URL = "https://understat.com"
    LEAGUE = "EPL"  # English Premier League

    def __init__(
        self, entity_resolution_service=None, db_session: Optional[Session] = None
    ):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self.name_matcher = PlayerNameMatcher()  # Fallback for simple matching
        self.entity_resolution = entity_resolution_service
        self.db_session = db_session

    async def get_player_stats(
        self, season: str = "2025", player_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch player statistics from Understat.

        Args:
            season: Season year (e.g., "2025" for 2025/26)
            player_name: Optional player name filter

        Returns:
            List of player statistics with xG, xA, NPxG
        """
        try:
            # Understat uses JavaScript-rendered content, so we need to parse the page
            url = f"{self.BASE_URL}/league/{self.LEAGUE}/{season}"

            logger.info(f"Fetching Understat data from {url}")
            response = await self.client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Understat stores data in JavaScript variables
            # Look for script tags with player data
            scripts = soup.find_all("script")
            players_data = []

            for script in scripts:
                if not script.string:
                    continue

                script_text = script.string

                # Try multiple patterns to find the data
                patterns = [
                    r"var\s+playersData\s*=\s*(\[.*?\]);",
                    r"playersData\s*=\s*(\[.*?\]);",
                    r"var\s+playersData\s*=\s*(\{.*?\});",
                    r"playersData\s*=\s*(\{.*?\});",
                ]

                for pattern in patterns:
                    try:
                        json_match = re.search(pattern, script_text, re.DOTALL)
                        if json_match:
                            data_str = json_match.group(1)
                            # Clean up the JSON string
                            data_str = data_str.strip()
                            # Handle potential trailing commas
                            data_str = re.sub(r",\s*}", "}", data_str)
                            data_str = re.sub(r",\s*]", "]", data_str)

                            parsed_data = json.loads(data_str)

                            # Handle both array and dict formats
                            if isinstance(parsed_data, list):
                                players_data = parsed_data
                            elif (
                                isinstance(parsed_data, dict)
                                and "players" in parsed_data
                            ):
                                players_data = parsed_data["players"]
                            elif isinstance(parsed_data, dict):
                                # If it's a dict with player data, try to extract
                                players_data = (
                                    list(parsed_data.values()) if parsed_data else []
                                )

                            if players_data:
                                break
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.debug(
                            f"Failed to parse with pattern {pattern}: {str(e)}"
                        )
                        continue

                if players_data:
                    break

            if not players_data:
                logger.warning("Could not extract playersData from Understat page")
                return []

            # Structure and clean the data
            structured_data = []
            for player in players_data:
                try:
                    # Handle different field name variations
                    name = (
                        player.get("player_name")
                        or player.get("name")
                        or player.get("Player")
                        or ""
                    )
                    team = (
                        player.get("team_title")
                        or player.get("team")
                        or player.get("Team")
                        or ""
                    )
                    position = player.get("position") or player.get("Position") or ""

                    # Parse numeric fields with error handling
                    games = self._safe_int(
                        player.get("games") or player.get("Games") or 0
                    )
                    time = self._safe_int(
                        player.get("time")
                        or player.get("Time")
                        or player.get("minutes")
                        or 0
                    )
                    goals = self._safe_int(
                        player.get("goals") or player.get("Goals") or 0
                    )
                    assists = self._safe_int(
                        player.get("assists") or player.get("Assists") or 0
                    )
                    xg = self._safe_float(player.get("xG") or player.get("xg") or 0.0)
                    xa = self._safe_float(player.get("xA") or player.get("xa") or 0.0)
                    npxg = self._safe_float(
                        player.get("npxG")
                        or player.get("npxg")
                        or player.get("NPxG")
                        or 0.0
                    )

                    # Calculate per-90 metrics
                    minutes_played = time if time > 0 else 0
                    if minutes_played > 0:
                        xg_per_90 = (xg / minutes_played) * 90.0
                        xa_per_90 = (xa / minutes_played) * 90.0
                        npxg_per_90 = (npxg / minutes_played) * 90.0
                    else:
                        xg_per_90 = 0.0
                        xa_per_90 = 0.0
                        npxg_per_90 = 0.0

                    structured_data.append(
                        {
                            "name": name.strip(),
                            "team": team.strip(),
                            "position": position.strip(),
                            "games": games,
                            "time": time,
                            "goals": goals,
                            "assists": assists,
                            "xg": xg,
                            "xa": xa,
                            "npxg": npxg,
                            "xg_per_90": round(xg_per_90, 3),
                            "xa_per_90": round(xa_per_90, 3),
                            "npxg_per_90": round(npxg_per_90, 3),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error parsing player data: {str(e)}")
                    continue

            if player_name:
                structured_data = [
                    p
                    for p in structured_data
                    if player_name.lower() in p["name"].lower()
                ]

            logger.info(
                f"Successfully extracted {len(structured_data)} players from Understat"
            )
            return structured_data

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching Understat data: {e.response.status_code}"
            )
            return []
        except Exception as e:
            logger.error(f"Error fetching Understat data: {str(e)}", exc_info=True)
            return []

    def _safe_int(self, value) -> int:
        """Safely convert value to int, handling various formats."""
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            # Remove commas and other formatting
            cleaned = re.sub(r"[^\d.-]", "", str(value))
            return int(float(cleaned))
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, value) -> float:
        """Safely convert value to float, handling various formats."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            # Remove commas and other formatting
            cleaned = re.sub(r"[^\d.-]", "", str(value))
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0

    async def get_match_stats(
        self, match_id: Optional[int] = None, season: str = "2025"
    ) -> List[Dict]:
        """
        Fetch match-by-match statistics from Understat.

        Returns:
            List of match statistics with player xG, xA per match
        """
        # This would require parsing individual match pages
        # For now, return empty list - can be implemented based on Understat's structure
        return []

    async def map_to_fpl_players(
        self,
        understat_data: List[Dict],
        fpl_players: List[Dict],
        use_entity_resolution: bool = True,
    ) -> Dict[int, Dict]:
        """
        Map Understat data to FPL players using Entity Resolution Engine.
        Falls back to simple name matching if Entity Resolution is not available.

        Args:
            understat_data: List of Understat player data
            fpl_players: List of FPL players
            use_entity_resolution: Whether to use Entity Resolution Engine (default: True)

        Returns:
            Dictionary mapping FPL ID to Understat data with confidence scores
        """
        mappings = {}

        # Use Entity Resolution if available
        if use_entity_resolution and self.entity_resolution:
            # Load master map if not already loaded
            if self.entity_resolution.master_map is None:
                await self.entity_resolution.load_master_map()

            # Extract Understat names
            understat_names = [
                p.get("name", "") for p in understat_data if p.get("name")
            ]

            for fpl_player in fpl_players:
                fpl_id = fpl_player.get("id")
                fpl_name = fpl_player.get("web_name", "") or fpl_player.get("name", "")
                fpl_team = fpl_player.get("team_name", "")

                if not fpl_id or not fpl_name:
                    continue

                # Try to resolve using Entity Resolution
                resolution = self.entity_resolution.resolve_player_entity(
                    fpl_id=fpl_id, fpl_name=fpl_name, fpl_team=fpl_team
                )

                # If found in master map, try to match by Understat name
                if resolution.get("matched"):
                    # Find matching Understat player by name
                    _understat_name = None
                    if "Understat_Name" in resolution:
                        _understat_name = resolution.get("Understat_Name")

                    # Try fuzzy matching against Understat names
                    if understat_names:
                        matches = self.entity_resolution.fuzzy_match_multi_source(
                            target_name=fpl_name,
                            understat_names=understat_names,
                            threshold=0.85,
                        )

                        if matches.get("understat"):
                            best_match_name, confidence = matches["understat"][0]
                            # Find the Understat player data
                            understat_player = next(
                                (
                                    p
                                    for p in understat_data
                                    if p.get("name") == best_match_name
                                ),
                                None,
                            )

                            if understat_player:
                                mappings[fpl_id] = {
                                    "external_data": understat_player,
                                    "similarity": confidence,
                                    "fpl_name": fpl_name,
                                    "external_name": best_match_name,
                                    "match_method": "entity_resolution",
                                    "confidence": confidence,
                                }
                                continue

                # Fallback: Try direct fuzzy matching
                if understat_names:
                    matches = self.entity_resolution.fuzzy_match_multi_source(
                        target_name=fpl_name,
                        understat_names=understat_names,
                        threshold=0.7,  # Lower threshold for fallback
                    )

                    if matches.get("understat"):
                        best_match_name, confidence = matches["understat"][0]
                        understat_player = next(
                            (
                                p
                                for p in understat_data
                                if p.get("name") == best_match_name
                            ),
                            None,
                        )

                        if understat_player:
                            mappings[fpl_id] = {
                                "external_data": understat_player,
                                "similarity": confidence,
                                "fpl_name": fpl_name,
                                "external_name": best_match_name,
                                "match_method": "fuzzy_fallback",
                                "confidence": confidence,
                            }

        # If no Entity Resolution or no matches, fall back to simple name matching
        if not mappings:
            mappings = self.name_matcher.create_mapping(
                fpl_players, understat_data, threshold=0.7
            )
            # Update match_method for simple matches
            for fpl_id in mappings:
                if "match_method" not in mappings[fpl_id]:
                    mappings[fpl_id]["match_method"] = "simple_name_match"

        return mappings


class FBrefService:
    """
    Service for fetching data from FBref.com
    Provides DefCon metrics: blocks, interventions (tackles/interceptions), passes
    """

    BASE_URL = "https://fbref.com"
    LEAGUE = "Premier-League"

    def __init__(
        self, entity_resolution_service=None, db_session: Optional[Session] = None
    ):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self.name_matcher = PlayerNameMatcher()  # Fallback for simple matching
        self.entity_resolution = entity_resolution_service
        self.db_session = db_session

    async def get_player_defensive_stats(
        self, season: str = "2025-2026", player_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch defensive statistics from FBref.
        Includes blocks, tackles, interceptions, passes for DefCon metrics.

        Args:
            season: Season string (e.g., "2025-2026")
            player_name: Optional player name filter

        Returns:
            List of player defensive statistics
        """
        try:
            # FBref URL structure
            url = f"{self.BASE_URL}/en/comps/9/{season}/{season}-Premier-League-Stats"

            logger.info(f"Fetching FBref data from {url}")
            response = await self.client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Find defensive stats table
            # FBref uses specific table IDs for different stat types
            defensive_stats = []

            # Try multiple table identification strategies
            tables = []

            # Strategy 1: Look for defensive action stats table by ID
            defensive_table = soup.find(
                "table", {"id": re.compile(r".*defensive.*", re.I)}
            )
            if defensive_table:
                tables.append(defensive_table)

            # Strategy 2: Look for standard stats table
            if not tables:
                stats_tables = soup.find_all(
                    "table", {"class": re.compile(r"stats.*table", re.I)}
                )
                tables.extend(stats_tables)

            # Strategy 3: Look for any table with defensive headers
            if not tables:
                all_tables = soup.find_all("table")
                for table in all_tables:
                    headers = table.find_all(["th", "thead"])
                    header_text = " ".join([h.get_text() for h in headers]).lower()
                    if any(
                        keyword in header_text
                        for keyword in ["tackle", "interception", "block", "defensive"]
                    ):
                        tables.append(table)
                        break

            if not tables:
                logger.warning("Could not find defensive stats table on FBref page")
                return []

            for table in tables:
                # Get header row to map column indices
                header_row = table.find("thead")
                if not header_row:
                    header_row = table.find("tr")

                if not header_row:
                    continue

                headers = header_row.find_all(["th", "td"])
                header_map = {}
                for idx, header in enumerate(headers):
                    header_text = header.get_text(strip=True).lower()
                    header_map[idx] = header_text

                # Find column indices for key metrics
                name_idx = self._find_column_by_keywords(header_map, ["player", "name"])
                team_idx = self._find_column_by_keywords(header_map, ["team", "squad"])
                pos_idx = self._find_column_by_keywords(header_map, ["pos", "position"])
                games_idx = self._find_column_by_keywords(
                    header_map, ["games", "mp", "matches"]
                )
                minutes_idx = self._find_column_by_keywords(
                    header_map, ["minutes", "min", "mins"]
                )
                blocks_idx = self._find_column_by_keywords(
                    header_map, ["blocks", "blocked", "blk"]
                )
                tackles_idx = self._find_column_by_keywords(
                    header_map, ["tackles", "tkl", "tklw"]
                )
                interceptions_idx = self._find_column_by_keywords(
                    header_map, ["interceptions", "int", "inter"]
                )
                passes_idx = self._find_column_by_keywords(
                    header_map, ["passes", "pass", "passes_completed"]
                )

                # Parse data rows
                rows = table.find_all("tr")[1:]  # Skip header

                for row in rows:
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 3:
                        continue

                    try:
                        # Extract player name (handle links)
                        name_cell = (
                            cells[name_idx] if name_idx is not None else cells[0]
                        )
                        name_link = name_cell.find("a")
                        name = (
                            name_link.get_text(strip=True)
                            if name_link
                            else name_cell.get_text(strip=True)
                        )

                        if not name or name.lower() in ["player", "name", ""]:
                            continue

                        # Extract basic stats
                        stats = {
                            "name": name,
                            "team": cells[team_idx].get_text(strip=True)
                            if team_idx is not None and len(cells) > team_idx
                            else "",
                            "position": cells[pos_idx].get_text(strip=True)
                            if pos_idx is not None and len(cells) > pos_idx
                            else "",
                            "games": self._parse_int(
                                cells[games_idx].get_text(strip=True)
                            )
                            if games_idx is not None and len(cells) > games_idx
                            else 0,
                            "minutes": self._parse_int(
                                cells[minutes_idx].get_text(strip=True)
                            )
                            if minutes_idx is not None and len(cells) > minutes_idx
                            else 0,
                        }

                        # Extract defensive metrics
                        stats["blocks"] = (
                            self._parse_int(cells[blocks_idx].get_text(strip=True))
                            if blocks_idx is not None and len(cells) > blocks_idx
                            else 0
                        )
                        stats["tackles"] = (
                            self._parse_int(cells[tackles_idx].get_text(strip=True))
                            if tackles_idx is not None and len(cells) > tackles_idx
                            else 0
                        )
                        stats["interceptions"] = (
                            self._parse_int(
                                cells[interceptions_idx].get_text(strip=True)
                            )
                            if interceptions_idx is not None
                            and len(cells) > interceptions_idx
                            else 0
                        )
                        stats["passes"] = (
                            self._parse_int(cells[passes_idx].get_text(strip=True))
                            if passes_idx is not None and len(cells) > passes_idx
                            else 0
                        )

                        # Calculate per-90 metrics
                        minutes = stats.get("minutes", 0)
                        if minutes > 0:
                            stats["blocks_per_90"] = round(
                                (stats.get("blocks", 0) / minutes) * 90, 3
                            )
                            stats["tackles_per_90"] = round(
                                (stats.get("tackles", 0) / minutes) * 90, 3
                            )
                            stats["interceptions_per_90"] = round(
                                (stats.get("interceptions", 0) / minutes) * 90, 3
                            )
                            stats["passes_per_90"] = round(
                                (stats.get("passes", 0) / minutes) * 90, 3
                            )

                            # Interventions = tackles + interceptions
                            interventions = stats.get("tackles", 0) + stats.get(
                                "interceptions", 0
                            )
                            stats["interventions"] = interventions
                            stats["interventions_per_90"] = round(
                                (interventions / minutes) * 90, 3
                            )
                        else:
                            stats["blocks_per_90"] = 0.0
                            stats["tackles_per_90"] = 0.0
                            stats["interceptions_per_90"] = 0.0
                            stats["passes_per_90"] = 0.0
                            stats["interventions"] = 0
                            stats["interventions_per_90"] = 0.0

                        defensive_stats.append(stats)

                    except Exception as e:
                        logger.debug(f"Error parsing row: {str(e)}")
                        continue

            if player_name:
                defensive_stats = [
                    p
                    for p in defensive_stats
                    if player_name.lower() in p["name"].lower()
                ]

            logger.info(
                f"Successfully extracted {len(defensive_stats)} players from FBref"
            )
            return defensive_stats

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching FBref data: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Error fetching FBref data: {str(e)}", exc_info=True)
            return []

    def _parse_int(self, value: str) -> int:
        """Parse integer from string, handling commas and other formatting"""
        try:
            return int(re.sub(r"[^\d]", "", value))
        except Exception:
            return 0

    def _find_column_index(self, row, keywords: List[str]) -> Optional[int]:
        """Find column index by header keywords (legacy method for backward compatibility)"""
        header_row = row.find_parent("table").find("tr")
        if not header_row:
            return None

        headers = header_row.find_all(["th", "td"])
        for idx, header in enumerate(headers):
            header_text = header.get_text(strip=True).lower()
            if any(keyword.lower() in header_text for keyword in keywords):
                return idx
        return None

    def _find_column_by_keywords(
        self, header_map: Dict[int, str], keywords: List[str]
    ) -> Optional[int]:
        """Find column index by keywords in header map"""
        for idx, header_text in header_map.items():
            if any(keyword.lower() in header_text for keyword in keywords):
                return idx
        return None

    async def map_to_fpl_players(
        self,
        fbref_data: List[Dict],
        fpl_players: List[Dict],
        use_entity_resolution: bool = True,
    ) -> Dict[int, Dict]:
        """
        Map FBref data to FPL players using Entity Resolution Engine.
        Falls back to simple name matching if Entity Resolution is not available.

        Args:
            fbref_data: List of FBref player data
            fpl_players: List of FPL players
            use_entity_resolution: Whether to use Entity Resolution Engine (default: True)

        Returns:
            Dictionary mapping FPL ID to FBref data with confidence scores
        """
        mappings = {}

        # Use Entity Resolution if available
        if use_entity_resolution and self.entity_resolution:
            # Load master map if not already loaded
            if self.entity_resolution.master_map is None:
                await self.entity_resolution.load_master_map()

            # Extract FBref names
            fbref_names = [p.get("name", "") for p in fbref_data if p.get("name")]

            for fpl_player in fpl_players:
                fpl_id = fpl_player.get("id")
                fpl_name = fpl_player.get("web_name", "") or fpl_player.get("name", "")
                fpl_team = fpl_player.get("team_name", "")

                if not fpl_id or not fpl_name:
                    continue

                # Try to resolve using Entity Resolution
                resolution = self.entity_resolution.resolve_player_entity(
                    fpl_id=fpl_id, fpl_name=fpl_name, fpl_team=fpl_team
                )

                # If found in master map, try to match by FBref name
                if resolution.get("matched"):
                    # Find matching FBref player by name
                    _fbref_name = None
                    if "FBref_Name" in resolution:
                        _fbref_name = resolution.get("FBref_Name")

                    # Try fuzzy matching against FBref names
                    if fbref_names:
                        matches = self.entity_resolution.fuzzy_match_multi_source(
                            target_name=fpl_name,
                            fbref_names=fbref_names,
                            threshold=0.85,
                        )

                        if matches.get("fbref"):
                            best_match_name, confidence = matches["fbref"][0]
                            # Find the FBref player data
                            fbref_player = next(
                                (
                                    p
                                    for p in fbref_data
                                    if p.get("name") == best_match_name
                                ),
                                None,
                            )

                            if fbref_player:
                                mappings[fpl_id] = {
                                    "external_data": fbref_player,
                                    "similarity": confidence,
                                    "fpl_name": fpl_name,
                                    "external_name": best_match_name,
                                    "match_method": "entity_resolution",
                                    "confidence": confidence,
                                }
                                continue

                # Fallback: Try direct fuzzy matching
                if fbref_names:
                    matches = self.entity_resolution.fuzzy_match_multi_source(
                        target_name=fpl_name,
                        fbref_names=fbref_names,
                        threshold=0.7,  # Lower threshold for fallback
                    )

                    if matches.get("fbref"):
                        best_match_name, confidence = matches["fbref"][0]
                        fbref_player = next(
                            (p for p in fbref_data if p.get("name") == best_match_name),
                            None,
                        )

                        if fbref_player:
                            mappings[fpl_id] = {
                                "external_data": fbref_player,
                                "similarity": confidence,
                                "fpl_name": fpl_name,
                                "external_name": best_match_name,
                                "match_method": "fuzzy_fallback",
                                "confidence": confidence,
                            }

        # If no Entity Resolution or no matches, fall back to simple name matching
        if not mappings:
            mappings = self.name_matcher.create_mapping(
                fpl_players, fbref_data, threshold=0.7
            )
            # Update match_method for simple matches
            for fpl_id in mappings:
                if "match_method" not in mappings[fpl_id]:
                    mappings[fpl_id]["match_method"] = "simple_name_match"

        return mappings


class ThirdPartyDataService:
    """
    Main service orchestrating third-party data integration.
    Uses Entity Resolution Engine for accurate player mapping.
    """

    def __init__(
        self, entity_resolution_service=None, db_session: Optional[Session] = None
    ):
        self.entity_resolution = entity_resolution_service
        self.db_session = db_session
        self.understat = UnderstatService(
            entity_resolution_service=entity_resolution_service, db_session=db_session
        )
        self.fbref = FBrefService(
            entity_resolution_service=entity_resolution_service, db_session=db_session
        )
        self.name_matcher = PlayerNameMatcher()

    async def enrich_player_data(self, fpl_player: Dict, season: str = "2025") -> Dict:
        """
        Enrich FPL player data with third-party metrics.

        Args:
            fpl_player: FPL player data
            season: Season string

        Returns:
            Enriched player data with Understat and FBref metrics
        """
        enriched = fpl_player.copy()

        player_name = fpl_player.get("web_name", "") or fpl_player.get("name", "")

        # Fetch Understat data
        try:
            understat_data = await self.understat.get_player_stats(season, player_name)
            if understat_data:
                # Match and merge using Entity Resolution
                mapping = await self.understat.map_to_fpl_players(
                    understat_data, [fpl_player]
                )
                if fpl_player.get("id") in mapping:
                    mapping_data = mapping[fpl_player["id"]]
                    ext_data = mapping_data["external_data"]
                    enriched.update(
                        {
                            "understat_xg": ext_data.get("xg", 0.0),
                            "understat_xa": ext_data.get("xa", 0.0),
                            "understat_npxg": ext_data.get("npxg", 0.0),
                            "understat_xg_per_90": ext_data.get("xg_per_90", 0.0),
                            "understat_xa_per_90": ext_data.get("xa_per_90", 0.0),
                            "understat_npxg_per_90": ext_data.get("npxg_per_90", 0.0),
                        }
                    )

                    # Store mapping in database if Entity Resolution is available
                    if self.entity_resolution and self.db_session:
                        try:
                            self.entity_resolution.upsert_mapping(
                                db=self.db_session,
                                fpl_id=fpl_player.get("id"),
                                fpl_name=player_name,
                                understat_name=mapping_data.get("external_name"),
                                confidence_score=mapping_data.get("confidence", 0.0),
                                manually_verified=False,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to store Understat mapping: {str(e)}"
                            )
        except Exception as e:
            logger.warning(
                f"Failed to fetch Understat data for {player_name}: {str(e)}"
            )

        # Fetch FBref data
        try:
            fbref_data = await self.fbref.get_player_defensive_stats(
                season, player_name
            )
            if fbref_data:
                # Match and merge using Entity Resolution
                mapping = await self.fbref.map_to_fpl_players(fbref_data, [fpl_player])
                if fpl_player.get("id") in mapping:
                    mapping_data = mapping[fpl_player["id"]]
                    ext_data = mapping_data["external_data"]
                    enriched.update(
                        {
                            "fbref_blocks": ext_data.get("blocks", 0),
                            "fbref_blocks_per_90": ext_data.get("blocks_per_90", 0.0),
                            "fbref_interventions": ext_data.get("interventions", 0),
                            "fbref_interventions_per_90": ext_data.get(
                                "interventions_per_90", 0.0
                            ),
                            "fbref_tackles": ext_data.get("tackles", 0),
                            "fbref_interceptions": ext_data.get("interceptions", 0),
                            "fbref_passes": ext_data.get("passes", 0),
                            "fbref_passes_per_90": ext_data.get("passes_per_90", 0.0),
                        }
                    )

                    # Store mapping in database if Entity Resolution is available
                    if self.entity_resolution and self.db_session:
                        try:
                            self.entity_resolution.upsert_mapping(
                                db=self.db_session,
                                fpl_id=fpl_player.get("id"),
                                fpl_name=player_name,
                                fbref_name=mapping_data.get("external_name"),
                                confidence_score=mapping_data.get("confidence", 0.0),
                                manually_verified=False,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to store FBref mapping: {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to fetch FBref data for {player_name}: {str(e)}")

        return enriched

    async def enrich_players_bulk(
        self,
        fpl_players: List[Dict],
        season: str = "2025",
        max_players: Optional[int] = None,
    ) -> List[Dict]:
        """
        Enrich multiple players with third-party data.
        Implements rate limiting.

        Args:
            fpl_players: List of FPL players
            season: Season string
            max_players: Maximum players to process

        Returns:
            List of enriched players
        """
        if max_players:
            fpl_players = fpl_players[:max_players]

        # Task 3.2: Process one at a time with periodic garbage collection for memory efficiency
        enriched_players = []
        import gc

        for i, player in enumerate(fpl_players):
            try:
                enriched = await self.enrich_player_data(player, season)
                enriched_players.append(enriched)

                # Rate limiting
                await asyncio.sleep(0.2)

                if (i + 1) % 10 == 0:
                    logger.info(f"Enriched {i + 1}/{len(fpl_players)} players")
                    # Memory management: collect garbage periodically
                    gc.collect()

            except Exception as e:
                logger.warning(f"Failed to enrich player {player.get('id')}: {str(e)}")
                enriched_players.append(player)

        return enriched_players
