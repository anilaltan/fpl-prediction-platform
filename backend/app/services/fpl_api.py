"""
Enhanced FPL API Service
Comprehensive data fetching, ID mapping, normalization, and PostgreSQL integration
Integrates:
- FPL Official API
- FBref DefCon metrics (2025/26 rules)
- FPL-ID-Map for entity resolution
- FuzzyWuzzy for name matching
- DGW normalization
- Async PostgreSQL saving
"""
import httpx
import os
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import logging
from fuzzywuzzy import fuzz, process
from bs4 import BeautifulSoup
import re

from app.services.entity_resolution import EntityResolutionService
from app.services.data_cleaning import DataCleaningService
from app.services.etl_service import ETLService

load_dotenv()
logger = logging.getLogger(__name__)


class FPLAPIService:
    """
    Enhanced FPL API Service with comprehensive data integration.
    Orchestrates data fetching, ID mapping, normalization, and database saving.
    """
    
    BASE_URL = "https://fantasy.premierleague.com/api"
    FBREF_BASE_URL = "https://fbref.com"
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """
        Initialize FPL API service with integrated services.
        
        Args:
            rate_limit_delay: Delay between requests in seconds (default: 0.1s)
        """
        self.email = os.getenv("FPL_EMAIL")
        self.password = os.getenv("FPL_PASSWORD")
        self.rate_limit_delay = rate_limit_delay
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        
        # Integrated services
        self.entity_resolution = EntityResolutionService()
        self.data_cleaning = DataCleaningService()
        self.etl_service = ETLService()
        
        # Load Master ID Map on initialization
        asyncio.create_task(self.entity_resolution.load_master_map())
    
    # ==================== FPL Official API Methods ====================
    
    async def get_bootstrap_data(self) -> Dict:
        """
        Fetch bootstrap-static data containing all players, teams, and fixtures.
        
        Returns:
            Dictionary with keys:
            - elements: List of all players
            - teams: List of all teams
            - events: List of gameweeks
            - element_types: Position types
        """
        try:
            response = await self.client.get(f"{self.BASE_URL}/bootstrap-static/")
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Fetched bootstrap data: {len(data.get('elements', []))} players, {len(data.get('teams', []))} teams")
            return data
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch bootstrap data: {str(e)}")
            raise Exception(f"Failed to fetch bootstrap data: {str(e)}")
    
    async def get_current_gameweek(self) -> Optional[int]:
        """
        Get the current active gameweek from FPL API.
        
        Returns:
            Current gameweek number, or None if not available
        """
        try:
            bootstrap = await self.get_bootstrap_data()
            events = bootstrap.get('events', [])
            
            # Find current gameweek (is_current = True)
            current_event = next((e for e in events if e.get('is_current')), None)
            if current_event:
                gameweek = current_event.get('id')
                logger.info(f"Current gameweek: {gameweek}")
                return gameweek
            
            # Fallback: Find next gameweek if current not found
            next_event = next((e for e in events if e.get('is_next')), None)
            if next_event:
                gameweek = next_event.get('id')
                logger.info(f"Using next gameweek as current: {gameweek}")
                return gameweek
            
            # Last resort: Use the latest finished gameweek
            finished_events = [e for e in events if e.get('finished')]
            if finished_events:
                latest_finished = max(finished_events, key=lambda x: x.get('id', 0))
                gameweek = latest_finished.get('id')
                logger.warning(f"No current gameweek found, using latest finished: {gameweek}")
                return gameweek
            
            logger.warning("No gameweek found in bootstrap data")
            return None
        except Exception as e:
            logger.error(f"Failed to get current gameweek: {str(e)}")
            return None
    
    async def get_next_gameweek(self) -> Optional[int]:
        """
        Get the next upcoming gameweek from FPL API.
        Prioritizes is_next=True, then finds first unfinished gameweek.
        
        Returns:
            Next gameweek number, or None if not available
        """
        try:
            bootstrap = await self.get_bootstrap_data()
            events = bootstrap.get('events', [])
            
            # Priority 1: Find next gameweek (is_next = True)
            next_event = next((e for e in events if e.get('is_next')), None)
            if next_event:
                gameweek = next_event.get('id')
                logger.info(f"Next gameweek (is_next=True): {gameweek}")
                return gameweek
            
            # Priority 2: Find first unfinished gameweek (finished = False)
            unfinished_events = [e for e in events if not e.get('finished', True)]
            if unfinished_events:
                # Sort by ID to get the earliest unfinished gameweek
                unfinished_events.sort(key=lambda x: x.get('id', 999))
                gameweek = unfinished_events[0].get('id')
                logger.info(f"Next gameweek (first unfinished): {gameweek}")
                return gameweek
            
            # Fallback: Use current gameweek if no next found
            current_event = next((e for e in events if e.get('is_current')), None)
            if current_event:
                gameweek = current_event.get('id')
                logger.warning(f"No next gameweek found, using current: {gameweek}")
                return gameweek
            
            logger.warning("No next gameweek found in bootstrap data")
            return None
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
        elements = bootstrap_data.get('elements', [])
        teams = bootstrap_data.get('teams', [])
        element_types = bootstrap_data.get('element_types', [])
        
        # Create lookup dictionaries
        team_lookup = {team['id']: team for team in teams}
        position_lookup = {et['id']: et['singular_name_short'] for et in element_types}
        
        players = []
        for element in elements:
            team_id = element.get('team')
            team_data = team_lookup.get(team_id, {})
            position_id = element.get('element_type')
            position = position_lookup.get(position_id, 'UNKNOWN')
            
            player = {
                'id': element.get('id'),
                'fpl_id': element.get('id'),
                'web_name': element.get('web_name', ''),
                'first_name': element.get('first_name', ''),
                'second_name': element.get('second_name', ''),
                'full_name': f"{element.get('first_name', '')} {element.get('second_name', '')}".strip(),
                'position': position,
                'position_id': position_id,
                'team_id': team_id,
                'team_name': team_data.get('name', ''),
                'team_short_name': team_data.get('short_name', ''),
                'price': element.get('now_cost', 0) / 10.0,
                'price_start': element.get('cost_change_start', 0) / 10.0,
                'selected_by_percent': float(element.get('selected_by_percent', 0.0)),
                'form': float(element.get('form', 0.0)),
                'points_per_game': float(element.get('points_per_game', 0.0)),
                'total_points': element.get('total_points', 0),
                'goals_scored': element.get('goals_scored', 0),
                'assists': element.get('assists', 0),
                'clean_sheets': element.get('clean_sheets', 0),
                'saves': element.get('saves', 0),
                'bonus': element.get('bonus', 0),
                'bps': element.get('bps', 0),
                'influence': float(element.get('influence', 0.0)),
                'creativity': float(element.get('creativity', 0.0)),
                'threat': float(element.get('threat', 0.0)),
                'ict_index': float(element.get('ict_index', 0.0)),
                'status': element.get('status', 'a'),
                'news': element.get('news', ''),
                'news_added': element.get('news_added'),
                'transfers_in': element.get('transfers_in', 0),
                'transfers_out': element.get('transfers_out', 0),
                'transfers_in_event': element.get('transfers_in_event', 0),
                'transfers_out_event': element.get('transfers_out_event', 0),
                'value_form': float(element.get('value_form', 0.0)),
                'value_season': float(element.get('value_season', 0.0)),
                'minutes': element.get('minutes', 0),
                'goals_conceded': element.get('goals_conceded', 0),
                'yellow_cards': element.get('yellow_cards', 0),
                'red_cards': element.get('red_cards', 0),
                'penalties_saved': element.get('penalties_saved', 0),
                'penalties_missed': element.get('penalties_missed', 0),
                'expected_goals': float(element.get('expected_goals', 0.0)),
                'expected_assists': float(element.get('expected_assists', 0.0)),
                'expected_goal_involvements': float(element.get('expected_goal_involvements', 0.0)),
                'expected_goals_conceded': float(element.get('expected_goals_conceded', 0.0)),
            }
            players.append(player)
        
        return players
    
    def extract_teams_from_bootstrap(self, bootstrap_data: Dict) -> List[Dict]:
        """Extract team data from bootstrap-static."""
        teams = bootstrap_data.get('teams', [])
        return [
            {
                'id': team.get('id'),
                'name': team.get('name', ''),
                'short_name': team.get('short_name', ''),
                'code': team.get('code', 0),
                'strength': team.get('strength', 0),
                'strength_attack_home': team.get('strength_attack_home', 0),
                'strength_attack_away': team.get('strength_attack_away', 0),
                'strength_defence_home': team.get('strength_defence_home', 0),
                'strength_defence_away': team.get('strength_defence_away', 0),
                'pulse_id': team.get('pulse_id', 0),
            }
            for team in teams
        ]
    
    async def get_player_data(self, player_id: int) -> Dict:
        """
        Fetch detailed data for a specific player from element-summary endpoint.
        
        Args:
            player_id: FPL player ID
        
        Returns:
            Dictionary containing history, history_past, fixtures, explain
        """
        try:
            await asyncio.sleep(self.rate_limit_delay)
            response = await self.client.get(f"{self.BASE_URL}/element-summary/{player_id}/")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch player data for {player_id}: {str(e)}")
            raise Exception(f"Failed to fetch player data: {str(e)}")
    
    def extract_player_history(self, player_summary: Dict) -> List[Dict]:
        """
        Extract match history statistics from element-summary.
        
        Returns:
            List of match statistics with normalized points for DGW
        """
        history = player_summary.get('history', [])
        
        structured_history = []
        for match in history:
            # Detect DGW (Double Gameweek) - if minutes > 90, likely 2 matches
            minutes = match.get('minutes', 0)
            matches_played = 2 if minutes > 90 else 1
            points = match.get('total_points', 0)
            
            # Normalize DGW points
            normalized_points = self.data_cleaning.normalize_dgw_points(
                points,
                matches_played,
                "dgw" if matches_played > 1 else "normal"
            )
            
            match_data = {
                'gameweek': match.get('round', 0),
                'fixture': match.get('fixture', 0),
                'opponent_team': match.get('opponent_team', 0),
                'was_home': match.get('was_home', False),
                'minutes': minutes,
                'points': points,
                'normalized_points': normalized_points,  # DGW normalized
                'matches_played': matches_played,
                'goals_scored': match.get('goals_scored', 0),
                'assists': match.get('assists', 0),
                'clean_sheets': match.get('clean_sheets', 0),
                'goals_conceded': match.get('goals_conceded', 0),
                'saves': match.get('saves', 0),
                'bonus': match.get('bonus', 0),
                'bps': match.get('bps', 0),
                'influence': float(match.get('influence', 0.0)),
                'creativity': float(match.get('creativity', 0.0)),
                'threat': float(match.get('threat', 0.0)),
                'ict_index': float(match.get('ict_index', 0.0)),
                'expected_goals': float(match.get('expected_goals', 0.0)),
                'expected_assists': float(match.get('expected_assists', 0.0)),
                'expected_goal_involvements': float(match.get('expected_goal_involvements', 0.0)),
                'expected_goals_conceded': float(match.get('expected_goals_conceded', 0.0)),
                'value': match.get('value', 0) / 10.0,
                'transfers_balance': match.get('transfers_balance', 0),
                'selected': match.get('selected', 0),
                'transfers_in': match.get('transfers_in', 0),
                'transfers_out': match.get('transfers_out', 0),
            }
            structured_history.append(match_data)
        
        return structured_history
    
    async def get_fixtures(self, gameweek: Optional[int] = None, future_only: bool = False) -> List[Dict]:
        """
        Fetch fixtures data.
        
        Args:
            gameweek: Optional gameweek filter. If None, uses next gameweek for future-focused queries.
            future_only: If True, only return unfinished fixtures (finished=False)
        
        Returns:
            List of fixture dictionaries
        """
        try:
            response = await self.client.get(f"{self.BASE_URL}/fixtures/")
            response.raise_for_status()
            fixtures = response.json()
            
            # If gameweek not provided, use next gameweek for future-focused queries
            if gameweek is None:
                next_gw = await self.get_next_gameweek()
                if next_gw:
                    gameweek = next_gw
                    logger.info(f"No gameweek provided, using next gameweek: {gameweek}")
            
            # Filter by gameweek if specified
            if gameweek:
                fixtures = [f for f in fixtures if f.get('event') == gameweek]
            
            # Filter out finished fixtures if future_only is True
            if future_only:
                fixtures = [f for f in fixtures if not f.get('finished', False)]
            
            logger.info(f"Fetched {len(fixtures)} fixtures (gameweek={gameweek}, future_only={future_only})")
            return fixtures
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch fixtures: {str(e)}")
            raise Exception(f"Failed to fetch fixtures: {str(e)}")
    
    def extract_fixtures_with_difficulty(
        self,
        fixtures: List[Dict],
        teams_data: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Extract and structure fixtures with difficulty ratings."""
        team_lookup = {}
        if teams_data:
            team_lookup = {team['id']: team for team in teams_data}
        
        structured_fixtures = []
        for fixture in fixtures:
            home_team_id = fixture.get('team_h')
            away_team_id = fixture.get('team_a')
            
            home_team = team_lookup.get(home_team_id, {})
            away_team = team_lookup.get(away_team_id, {})
            
            # Get strength values, default to 1000 if not found or 0
            home_strength = home_team.get('strength', 0) or 1000
            away_strength = away_team.get('strength', 0) or 1000
            
            # Ensure strength is at least 100 to avoid division issues
            home_strength = max(home_strength, 100)
            away_strength = max(away_strength, 100)
            
            # Calculate difficulty: opponent's strength determines difficulty
            # Higher opponent strength = higher difficulty (1-5 scale)
            # Normalize: strength ranges roughly 1000-1500, map to 1-5
            home_difficulty = min(5, max(1, int(round((away_strength / 1000.0) * 5))))
            away_difficulty = min(5, max(1, int(round((home_strength / 1000.0) * 5))))
            
            fixture_data = {
                'id': fixture.get('id'),
                'gameweek': fixture.get('event'),
                'kickoff_time': fixture.get('kickoff_time'),
                'home_team_id': home_team_id,
                'home_team_name': home_team.get('name', ''),
                'away_team_id': away_team_id,
                'away_team_name': away_team.get('name', ''),
                'home_difficulty': home_difficulty,
                'away_difficulty': away_difficulty,
                'home_score': fixture.get('team_h_score'),
                'away_score': fixture.get('team_a_score'),
                'finished': fixture.get('finished', False),
                'home_strength': home_strength,
                'away_strength': away_strength,
            }
            structured_fixtures.append(fixture_data)
        
        return structured_fixtures
    
    # ==================== FBref DefCon Metrics ====================
    
    async def fetch_fbref_defcon_metrics(
        self,
        season: str = "2025-2026",
        player_name: Optional[str] = None
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
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            defensive_stats = []
            
            # Find defensive stats table
            # FBref uses table IDs like "stats_standard_defense" or "stats_defense"
            tables = soup.find_all('table', {'id': re.compile(r'.*defense.*', re.I)})
            
            if not tables:
                # Fallback to standard stats table
                tables = soup.find_all('table', {'class': 'stats_table'})
            
            for table in tables:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 5:
                        continue
                    
                    try:
                        name = cells[0].get_text(strip=True)
                        
                        stats = {
                            'name': name,
                            'team': cells[1].get_text(strip=True) if len(cells) > 1 else '',
                            'position': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                            'games': self._parse_int(cells[3].get_text(strip=True)) if len(cells) > 3 else 0,
                            'minutes': self._parse_int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0,
                        }
                        
                        # Find defensive metrics columns
                        header_row = table.find('tr')
                        if header_row:
                            headers = [h.get_text(strip=True).lower() for h in header_row.find_all(['th', 'td'])]
                            
                            # Blocks
                            blocks_idx = self._find_column_index(headers, ['blocks', 'blocked'])
                            if blocks_idx is not None:
                                stats['blocks'] = self._parse_int(cells[blocks_idx].get_text(strip=True))
                            
                            # Tackles
                            tackles_idx = self._find_column_index(headers, ['tackles', 'tkl', 'tackles_won'])
                            if tackles_idx is not None:
                                stats['tackles'] = self._parse_int(cells[tackles_idx].get_text(strip=True))
                            
                            # Interceptions
                            int_idx = self._find_column_index(headers, ['interceptions', 'int', 'intercepted'])
                            if int_idx is not None:
                                stats['interceptions'] = self._parse_int(cells[int_idx].get_text(strip=True))
                            
                            # Passes
                            passes_idx = self._find_column_index(headers, ['passes', 'pass', 'passes_completed'])
                            if passes_idx is not None:
                                stats['passes'] = self._parse_int(cells[passes_idx].get_text(strip=True))
                        
                        # Calculate per-90 and interventions
                        minutes = stats.get('minutes', 0)
                        if minutes > 0:
                            stats['blocks_per_90'] = (stats.get('blocks', 0) / minutes) * 90
                            stats['tackles_per_90'] = (stats.get('tackles', 0) / minutes) * 90
                            stats['interceptions_per_90'] = (stats.get('interceptions', 0) / minutes) * 90
                            stats['passes_per_90'] = (stats.get('passes', 0) / minutes) * 90
                            
                            # Interventions = tackles + interceptions (2025/26 DefCon rule)
                            interventions = stats.get('tackles', 0) + stats.get('interceptions', 0)
                            stats['interventions'] = interventions
                            stats['interventions_per_90'] = (interventions / minutes) * 90
                        else:
                            stats['blocks_per_90'] = 0.0
                            stats['tackles_per_90'] = 0.0
                            stats['interceptions_per_90'] = 0.0
                            stats['passes_per_90'] = 0.0
                            stats['interventions'] = 0
                            stats['interventions_per_90'] = 0.0
                        
                        defensive_stats.append(stats)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing FBref row: {str(e)}")
                        continue
            
            if player_name:
                defensive_stats = [p for p in defensive_stats if player_name.lower() in p['name'].lower()]
            
            return defensive_stats
            
        except Exception as e:
            logger.error(f"Error fetching FBref DefCon metrics: {str(e)}")
            return []
    
    def _parse_int(self, value: str) -> int:
        """Parse integer from string, handling commas and formatting"""
        try:
            return int(re.sub(r'[^\d]', '', str(value)))
        except:
            return 0
    
    def _find_column_index(self, headers: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index by header keywords"""
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
        threshold: int = 80
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
            fpl_id = fpl_player.get('fpl_id') or fpl_player.get('id')
            fpl_name = fpl_player.get('web_name') or fpl_player.get('full_name', '')
            
            # Try entity resolution first
            resolution = self.entity_resolution.resolve_player_entity(
                fpl_id=fpl_id,
                fpl_name=fpl_name,
                fpl_team=fpl_player.get('team_name')
            )
            
            if resolution.get('matched') and resolution.get('fbref_id'):
                # Found in Master Map
                fbref_id = resolution.get('fbref_id')
                # Find matching FBref player
                fbref_player = next(
                    (p for p in fbref_players if str(fbref_id) in p.get('name', '') or fbref_id == p.get('fbref_id')),
                    None
                )
                if fbref_player:
                    mappings[fpl_id] = {
                        'fbref_data': fbref_player,
                        'match_method': 'master_map',
                        'confidence': 1.0
                    }
                    continue
            
            # If not found, try FuzzyWuzzy matching
            if use_fuzzy:
                fbref_names = [p.get('name', '') for p in fbref_players]
                
                # Use FuzzyWuzzy to find best match
                best_match = process.extractOne(
                    fpl_name,
                    fbref_names,
                    scorer=fuzz.token_sort_ratio
                )
                
                if best_match and best_match[1] >= threshold:
                    matched_name = best_match[0]
                    fbref_player = next(
                        (p for p in fbref_players if p.get('name') == matched_name),
                        None
                    )
                    if fbref_player:
                        mappings[fpl_id] = {
                            'fbref_data': fbref_player,
                            'match_method': 'fuzzywuzzy',
                            'confidence': best_match[1] / 100.0,
                            'matched_name': matched_name
                        }
                        logger.info(f"Fuzzy matched: {fpl_name} -> {matched_name} (score: {best_match[1]})")
        
        return mappings
    
    # ==================== Comprehensive Data Fetching ====================
    
    async def fetch_comprehensive_player_data(
        self,
        player_id: int,
        season: str = "2025-26",
        include_fbref: bool = True,
        normalize_dgw: bool = True
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
            fpl_player = next((p for p in players if p['id'] == player_id), None)
            
            if not fpl_player:
                raise ValueError(f"Player {player_id} not found in bootstrap data")
            
            # 2. Extract history with DGW normalization
            history = self.extract_player_history(player_summary)
            
            # 3. Fetch FBref DefCon metrics if requested
            fbref_data = None
            if include_fbref:
                fbref_players = await self.fetch_fbref_defcon_metrics(season)
                mappings = await self.map_players_with_fbref(
                    [fpl_player],
                    fbref_players,
                    use_fuzzy=True,
                    threshold=80
                )
                
                if player_id in mappings:
                    fbref_data = mappings[player_id]['fbref_data']
                    # Add DefCon metrics to player data
                    fpl_player.update({
                        'fbref_blocks': fbref_data.get('blocks', 0),
                        'fbref_blocks_per_90': fbref_data.get('blocks_per_90', 0.0),
                        'fbref_tackles': fbref_data.get('tackles', 0),
                        'fbref_interceptions': fbref_data.get('interceptions', 0),
                        'fbref_interventions': fbref_data.get('interventions', 0),
                        'fbref_interventions_per_90': fbref_data.get('interventions_per_90', 0.0),
                        'fbref_passes': fbref_data.get('passes', 0),
                        'fbref_passes_per_90': fbref_data.get('passes_per_90', 0.0),
                    })
            
            # 4. Calculate DefCon floor points
            position = fpl_player.get('position', 'MID')
            defcon_metrics = self.data_cleaning.get_defcon_metrics(fpl_player, position)
            fpl_player['defcon_floor_points'] = defcon_metrics['floor_points']
            
            # 5. Combine all data
            comprehensive_data = {
                'fpl_data': fpl_player,
                'history': history,
                'history_past': player_summary.get('history_past', []),
                'fixtures': player_summary.get('fixtures', []),
                'fbref_data': fbref_data,
                'defcon_metrics': defcon_metrics
            }
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive data for player {player_id}: {str(e)}")
            raise
    
    # ==================== PostgreSQL Integration ====================
    
    async def save_player_gameweek_stats(
        self,
        player_id: int,
        gameweek: int,
        season: str = "2025-26",
        include_fbref: bool = True
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
                player_id,
                season,
                include_fbref=include_fbref,
                normalize_dgw=True
            )
            
            # Find gameweek stats
            history = comprehensive_data.get('history', [])
            gameweek_stats = next(
                (h for h in history if h.get('gameweek') == gameweek),
                None
            )
            
            if not gameweek_stats:
                return {
                    'status': 'not_found',
                    'message': f"No stats found for player {player_id} in gameweek {gameweek}"
                }
            
            # Prepare stats for database
            fpl_data = comprehensive_data.get('fpl_data', {})
            fbref_data = comprehensive_data.get('fbref_data', {})
            
            stats_for_db = {
                'fpl_id': player_id,
                'gameweek': gameweek,
                'season': season,
                'minutes': gameweek_stats.get('minutes', 0),
                'goals': gameweek_stats.get('goals_scored', 0),
                'assists': gameweek_stats.get('assists', 0),
                'clean_sheets': gameweek_stats.get('clean_sheets', 0),
                'goals_conceded': gameweek_stats.get('goals_conceded', 0),
                'saves': gameweek_stats.get('saves', 0),
                'bonus': gameweek_stats.get('bonus', 0),
                'bps': gameweek_stats.get('bps', 0),
                'total_points': gameweek_stats.get('points', 0),
                'normalized_points': gameweek_stats.get('normalized_points', 0.0),
                'xg': gameweek_stats.get('expected_goals', 0.0),
                'xa': gameweek_stats.get('expected_assists', 0.0),
                'xgi': gameweek_stats.get('expected_goal_involvements', 0.0),
                'xgc': gameweek_stats.get('expected_goals_conceded', 0.0),
                'influence': gameweek_stats.get('influence', 0.0),
                'creativity': gameweek_stats.get('creativity', 0.0),
                'threat': gameweek_stats.get('threat', 0.0),
                'ict_index': gameweek_stats.get('ict_index', 0.0),
                'blocks': fbref_data.get('blocks', 0) if fbref_data else 0,
                'interventions': fbref_data.get('interventions', 0) if fbref_data else 0,
                'passes': fbref_data.get('passes', 0) if fbref_data else 0,
                'defcon_floor_points': comprehensive_data.get('defcon_metrics', {}).get('floor_points', 0.0),
                'was_home': gameweek_stats.get('was_home', True),
                'opponent_team': gameweek_stats.get('opponent_team'),
            }
            
            # Save to database using ETL service
            await self.etl_service.upsert_player_gameweek_stats(stats_for_db)
            
            return {
                'status': 'success',
                'player_id': player_id,
                'gameweek': gameweek,
                'stats': stats_for_db
            }
            
        except Exception as e:
            logger.error(f"Error saving gameweek stats for player {player_id}, GW {gameweek}: {str(e)}")
            raise
    
    async def bulk_save_gameweek_stats(
        self,
        gameweek: int,
        season: str = "2025-26",
        max_players: Optional[int] = None
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
            
            results = {
                'total_players': len(players),
                'saved': 0,
                'errors': 0,
                'errors_list': []
            }
            
            # Process each player
            for i, player in enumerate(players):
                try:
                    await self.save_player_gameweek_stats(
                        player['id'],
                        gameweek,
                        season,
                        include_fbref=True
                    )
                    results['saved'] += 1
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Saved stats for {i + 1}/{len(players)} players")
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    results['errors'] += 1
                    results['errors_list'].append({
                        'player_id': player['id'],
                        'error': str(e)
                    })
                    logger.warning(f"Error saving stats for player {player['id']}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk save: {str(e)}")
            raise
    
    async def close(self):
        """Close HTTP client and services"""
        await self.client.aclose()
        await self.entity_resolution.close()
        await self.etl_service.close()