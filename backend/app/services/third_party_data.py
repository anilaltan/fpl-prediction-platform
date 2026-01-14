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
from datetime import datetime
import logging
from bs4 import BeautifulSoup
import json

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
        name = re.sub(r'[àáâãäå]', 'a', name)
        name = re.sub(r'[èéêë]', 'e', name)
        name = re.sub(r'[ìíîï]', 'i', name)
        name = re.sub(r'[òóôõö]', 'o', name)
        name = re.sub(r'[ùúûü]', 'u', name)
        name = re.sub(r'[ç]', 'c', name)
        name = re.sub(r'[ñ]', 'n', name)
        
        # Remove common suffixes
        name = re.sub(r'\s+[jr]\.?\s*$', '', name)  # Jr., Jr, jr
        name = re.sub(r'\s+[ivx]+$', '', name)  # Roman numerals
        
        return name.strip()
    
    def match_player(
        self,
        fpl_name: str,
        external_name: str,
        team: Optional[str] = None
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
        threshold: float = 0.7
    ) -> Dict[int, Dict]:
        """
        Create mapping between FPL players and external data sources.
        
        Returns:
            Dictionary mapping FPL ID to external player data
        """
        mappings = {}
        
        for fpl_player in fpl_players:
            fpl_id = fpl_player.get('id')
            fpl_name = fpl_player.get('web_name', '') or fpl_player.get('name', '')
            fpl_team = fpl_player.get('team_name', '')
            
            best_match = None
            best_score = 0.0
            
            for ext_player in external_players:
                ext_name = ext_player.get('name', '')
                ext_team = ext_player.get('team', '')
                
                # Team match bonus
                team_bonus = 0.1 if fpl_team.lower() == ext_team.lower() else 0.0
                
                score = self.match_player(fpl_name, ext_name) + team_bonus
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = ext_player
            
            if best_match:
                mappings[fpl_id] = {
                    'external_data': best_match,
                    'similarity': best_score,
                    'fpl_name': fpl_name,
                    'external_name': best_match.get('name', '')
                }
        
        return mappings


class UnderstatService:
    """
    Service for fetching data from Understat.com
    Provides xG, xA, NPxG (Non-Penalty Expected Goals) metrics
    """
    
    BASE_URL = "https://understat.com"
    LEAGUE = "EPL"  # English Premier League
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self.name_matcher = PlayerNameMatcher()
    
    async def get_player_stats(
        self,
        season: str = "2025",
        player_name: Optional[str] = None
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
            # In production, you might need Selenium or use their API if available
            url = f"{self.BASE_URL}/league/{self.LEAGUE}/{season}"
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Understat stores data in JavaScript variables
            # Look for script tags with player data
            scripts = soup.find_all('script')
            players_data = []
            
            for script in scripts:
                if script.string and 'playersData' in script.string:
                    # Extract JSON data from JavaScript
                    # This is a simplified version - actual implementation may vary
                    try:
                        # Find JSON-like data
                        json_match = re.search(r'var\s+playersData\s*=\s*(\[.*?\]);', script.string, re.DOTALL)
                        if json_match:
                            data_str = json_match.group(1)
                            players_data = json.loads(data_str)
                            break
                    except:
                        continue
            
            # Structure the data
            structured_data = []
            for player in players_data:
                structured_data.append({
                    'name': player.get('player_name', ''),
                    'team': player.get('team_title', ''),
                    'position': player.get('position', ''),
                    'games': int(player.get('games', 0)),
                    'time': int(player.get('time', 0)),
                    'goals': int(player.get('goals', 0)),
                    'assists': int(player.get('assists', 0)),
                    'xg': float(player.get('xG', 0.0)),
                    'xa': float(player.get('xA', 0.0)),
                    'npxg': float(player.get('npxG', 0.0)),  # Non-penalty xG
                    'xg_per_90': float(player.get('xG', 0.0)) / (int(player.get('time', 0)) / 90.0) if player.get('time', 0) > 0 else 0.0,
                    'xa_per_90': float(player.get('xA', 0.0)) / (int(player.get('time', 0)) / 90.0) if player.get('time', 0) > 0 else 0.0,
                    'npxg_per_90': float(player.get('npxG', 0.0)) / (int(player.get('time', 0)) / 90.0) if player.get('time', 0) > 0 else 0.0,
                })
            
            if player_name:
                structured_data = [p for p in structured_data if player_name.lower() in p['name'].lower()]
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error fetching Understat data: {str(e)}")
            return []
    
    async def get_match_stats(
        self,
        match_id: Optional[int] = None,
        season: str = "2025"
    ) -> List[Dict]:
        """
        Fetch match-by-match statistics from Understat.
        
        Returns:
            List of match statistics with player xG, xA per match
        """
        # This would require parsing individual match pages
        # For now, return empty list - can be implemented based on Understat's structure
        return []
    
    def map_to_fpl_players(
        self,
        understat_data: List[Dict],
        fpl_players: List[Dict]
    ) -> Dict[int, Dict]:
        """
        Map Understat data to FPL players using name matching.
        
        Returns:
            Dictionary mapping FPL ID to Understat data
        """
        return self.name_matcher.create_mapping(fpl_players, understat_data, threshold=0.7)


class FBrefService:
    """
    Service for fetching data from FBref.com
    Provides DefCon metrics: blocks, interventions (tackles/interceptions), passes
    """
    
    BASE_URL = "https://fbref.com"
    LEAGUE = "Premier-League"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self.name_matcher = PlayerNameMatcher()
    
    async def get_player_defensive_stats(
        self,
        season: str = "2025-2026",
        player_name: Optional[str] = None
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
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find defensive stats table
            # FBref uses specific table IDs for different stat types
            defensive_stats = []
            
            # Look for defensive action stats table
            tables = soup.find_all('table', {'id': re.compile(r'.*defensive.*', re.I)})
            
            if not tables:
                # Try to find standard stats table
                tables = soup.find_all('table', {'class': 'stats_table'})
            
            for table in tables:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 5:
                        continue
                    
                    try:
                        player_name_cell = cells[0]
                        name = player_name_cell.get_text(strip=True)
                        
                        # Extract stats (column positions may vary)
                        stats = {
                            'name': name,
                            'team': cells[1].get_text(strip=True) if len(cells) > 1 else '',
                            'position': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                            'games': self._parse_int(cells[3].get_text(strip=True)) if len(cells) > 3 else 0,
                            'minutes': self._parse_int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0,
                        }
                        
                        # Look for defensive metrics (columns vary)
                        # Blocks
                        blocks_idx = self._find_column_index(row, ['blocks', 'blocked'])
                        if blocks_idx:
                            stats['blocks'] = self._parse_int(cells[blocks_idx].get_text(strip=True))
                        
                        # Tackles
                        tackles_idx = self._find_column_index(row, ['tackles', 'tkl'])
                        if tackles_idx:
                            stats['tackles'] = self._parse_int(cells[tackles_idx].get_text(strip=True))
                        
                        # Interceptions
                        interceptions_idx = self._find_column_index(row, ['interceptions', 'int'])
                        if interceptions_idx:
                            stats['interceptions'] = self._parse_int(cells[interceptions_idx].get_text(strip=True))
                        
                        # Passes
                        passes_idx = self._find_column_index(row, ['passes', 'pass'])
                        if passes_idx:
                            stats['passes'] = self._parse_int(cells[passes_idx].get_text(strip=True))
                        
                        # Calculate per-90
                        minutes = stats.get('minutes', 0)
                        if minutes > 0:
                            stats['blocks_per_90'] = (stats.get('blocks', 0) / minutes) * 90
                            stats['tackles_per_90'] = (stats.get('tackles', 0) / minutes) * 90
                            stats['interceptions_per_90'] = (stats.get('interceptions', 0) / minutes) * 90
                            stats['passes_per_90'] = (stats.get('passes', 0) / minutes) * 90
                            
                            # Interventions = tackles + interceptions
                            interventions = stats.get('tackles', 0) + stats.get('interceptions', 0)
                            stats['interventions'] = interventions
                            stats['interventions_per_90'] = (interventions / minutes) * 90
                        else:
                            stats['blocks_per_90'] = 0.0
                            stats['tackles_per_90'] = 0.0
                            stats['interventions_per_90'] = 0.0
                            stats['passes_per_90'] = 0.0
                            stats['interventions'] = 0
                            stats['interventions_per_90'] = 0.0
                        
                        defensive_stats.append(stats)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing row: {str(e)}")
                        continue
            
            if player_name:
                defensive_stats = [p for p in defensive_stats if player_name.lower() in p['name'].lower()]
            
            return defensive_stats
            
        except Exception as e:
            logger.error(f"Error fetching FBref data: {str(e)}")
            return []
    
    def _parse_int(self, value: str) -> int:
        """Parse integer from string, handling commas and other formatting"""
        try:
            return int(re.sub(r'[^\d]', '', value))
        except:
            return 0
    
    def _find_column_index(self, row, keywords: List[str]) -> Optional[int]:
        """Find column index by header keywords"""
        header_row = row.find_parent('table').find('tr')
        if not header_row:
            return None
        
        headers = header_row.find_all(['th', 'td'])
        for idx, header in enumerate(headers):
            header_text = header.get_text(strip=True).lower()
            if any(keyword.lower() in header_text for keyword in keywords):
                return idx
        return None
    
    def map_to_fpl_players(
        self,
        fbref_data: List[Dict],
        fpl_players: List[Dict]
    ) -> Dict[int, Dict]:
        """
        Map FBref data to FPL players using name matching.
        
        Returns:
            Dictionary mapping FPL ID to FBref data
        """
        return self.name_matcher.create_mapping(fpl_players, fbref_data, threshold=0.7)


class ThirdPartyDataService:
    """
    Main service orchestrating third-party data integration.
    """
    
    def __init__(self):
        self.understat = UnderstatService()
        self.fbref = FBrefService()
        self.name_matcher = PlayerNameMatcher()
    
    async def enrich_player_data(
        self,
        fpl_player: Dict,
        season: str = "2025"
    ) -> Dict:
        """
        Enrich FPL player data with third-party metrics.
        
        Args:
            fpl_player: FPL player data
            season: Season string
        
        Returns:
            Enriched player data with Understat and FBref metrics
        """
        enriched = fpl_player.copy()
        
        player_name = fpl_player.get('web_name', '') or fpl_player.get('name', '')
        
        # Fetch Understat data
        try:
            understat_data = await self.understat.get_player_stats(season, player_name)
            if understat_data:
                # Match and merge
                mapping = self.understat.map_to_fpl_players(understat_data, [fpl_player])
                if fpl_player.get('id') in mapping:
                    ext_data = mapping[fpl_player['id']]['external_data']
                    enriched.update({
                        'understat_xg': ext_data.get('xg', 0.0),
                        'understat_xa': ext_data.get('xa', 0.0),
                        'understat_npxg': ext_data.get('npxg', 0.0),
                        'understat_xg_per_90': ext_data.get('xg_per_90', 0.0),
                        'understat_xa_per_90': ext_data.get('xa_per_90', 0.0),
                        'understat_npxg_per_90': ext_data.get('npxg_per_90', 0.0),
                    })
        except Exception as e:
            logger.warning(f"Failed to fetch Understat data for {player_name}: {str(e)}")
        
        # Fetch FBref data
        try:
            fbref_data = await self.fbref.get_player_defensive_stats(season, player_name)
            if fbref_data:
                # Match and merge
                mapping = self.fbref.map_to_fpl_players(fbref_data, [fpl_player])
                if fpl_player.get('id') in mapping:
                    ext_data = mapping[fpl_player['id']]['external_data']
                    enriched.update({
                        'fbref_blocks': ext_data.get('blocks', 0),
                        'fbref_blocks_per_90': ext_data.get('blocks_per_90', 0.0),
                        'fbref_interventions': ext_data.get('interventions', 0),
                        'fbref_interventions_per_90': ext_data.get('interventions_per_90', 0.0),
                        'fbref_tackles': ext_data.get('tackles', 0),
                        'fbref_interceptions': ext_data.get('interceptions', 0),
                        'fbref_passes': ext_data.get('passes', 0),
                        'fbref_passes_per_90': ext_data.get('passes_per_90', 0.0),
                    })
        except Exception as e:
            logger.warning(f"Failed to fetch FBref data for {player_name}: {str(e)}")
        
        return enriched
    
    async def enrich_players_bulk(
        self,
        fpl_players: List[Dict],
        season: str = "2025",
        max_players: Optional[int] = None
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
        
        enriched_players = []
        
        for i, player in enumerate(fpl_players):
            try:
                enriched = await self.enrich_player_data(player, season)
                enriched_players.append(enriched)
                
                # Rate limiting
                await asyncio.sleep(0.2)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Enriched {i + 1}/{len(fpl_players)} players")
                    
            except Exception as e:
                logger.warning(f"Failed to enrich player {player.get('id')}: {str(e)}")
                enriched_players.append(player)
        
        return enriched_players