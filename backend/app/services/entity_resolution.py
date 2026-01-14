"""
Entity Resolution Service for FPL Player ID Mapping
Integrates Master ID Map from ChrisMusson/FPL-ID-Map GitHub repository
Handles name variations and fuzzy matching for unmatched players
"""
import httpx
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
import logging
import re
from difflib import SequenceMatcher
from datetime import datetime
import json
from io import StringIO

logger = logging.getLogger(__name__)


class EntityResolutionService:
    """
    Entity Resolution service using Master ID Map from GitHub.
    Maps players across FPL, Understat, FBref, and other sources.
    """
    
    MASTER_MAP_URL = "https://raw.githubusercontent.com/ChrisMusson/FPL-ID-Map/main/Master.csv"
    LOCAL_MAP_PATH = "data/master_id_map.csv"
    
    def __init__(self):
        self.master_map: Optional[pd.DataFrame] = None
        self.unmatched_players: List[Dict] = []
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def load_master_map(self, force_reload: bool = False) -> bool:
        """
        Load Master ID Map from GitHub or local cache.
        
        Args:
            force_reload: Force reload from GitHub even if local exists
        
        Returns:
            True if loaded successfully
        """
        try:
            # Check if local file exists and is recent
            if not force_reload and os.path.exists(self.LOCAL_MAP_PATH):
                file_age = datetime.now().timestamp() - os.path.getmtime(self.LOCAL_MAP_PATH)
                # Use local if less than 24 hours old
                if file_age < 86400:
                    logger.info(f"Loading Master ID Map from local cache: {self.LOCAL_MAP_PATH}")
                    self.master_map = pd.read_csv(self.LOCAL_MAP_PATH)
                    return True
            
            # Download from GitHub
            logger.info(f"Downloading Master ID Map from GitHub: {self.MASTER_MAP_URL}")
            response = await self.client.get(self.MASTER_MAP_URL)
            response.raise_for_status()
            
            # Parse CSV
            csv_data = StringIO(response.text)
            self.master_map = pd.read_csv(csv_data)
            
            # Save to local cache
            os.makedirs(os.path.dirname(self.LOCAL_MAP_PATH), exist_ok=True)
            self.master_map.to_csv(self.LOCAL_MAP_PATH, index=False)
            
            logger.info(f"Loaded Master ID Map: {len(self.master_map)} mappings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Master ID Map: {str(e)}")
            # Try to load local if available
            if os.path.exists(self.LOCAL_MAP_PATH):
                logger.info("Falling back to local Master ID Map")
                self.master_map = pd.read_csv(self.LOCAL_MAP_PATH)
                return True
            return False
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize player name for matching.
        More comprehensive than basic normalization.
        """
        if not name or pd.isna(name):
            return ""
        
        name = str(name).lower().strip()
        
        # Remove accents
        name = re.sub(r'[àáâãäå]', 'a', name)
        name = re.sub(r'[èéêë]', 'e', name)
        name = re.sub(r'[ìíîï]', 'i', name)
        name = re.sub(r'[òóôõö]', 'o', name)
        name = re.sub(r'[ùúûü]', 'u', name)
        name = re.sub(r'[ç]', 'c', name)
        name = re.sub(r'[ñ]', 'n', name)
        name = re.sub(r'[ýÿ]', 'y', name)
        
        # Remove common suffixes and prefixes
        name = re.sub(r'\s+[jr]\.?\s*$', '', name)  # Jr., Jr
        name = re.sub(r'\s+[ivx]+$', '', name)  # Roman numerals
        name = re.sub(r'^\s*[ivx]+\s+', '', name)  # Prefix roman numerals
        
        # Remove dots and special characters (keep spaces)
        name = re.sub(r'[^\w\s]', '', name)
        
        # Normalize whitespace
        name = ' '.join(name.split())
        
        return name.strip()
    
    def find_player_in_master_map(
        self,
        fpl_id: Optional[int] = None,
        fpl_name: Optional[str] = None,
        understat_name: Optional[str] = None,
        fbref_name: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Find player in Master ID Map using various identifiers.
        
        Returns:
            Dictionary with all IDs if found, None otherwise
        """
        if self.master_map is None:
            logger.warning("Master ID Map not loaded")
            return None
        
        # Search by FPL ID
        if fpl_id is not None:
            match = self.master_map[self.master_map.get('FPL', pd.NA) == fpl_id]
            if not match.empty:
                return match.iloc[0].to_dict()
        
        # Search by name variations
        if fpl_name:
            fpl_norm = self.normalize_name(fpl_name)
            for col in ['FPL_Name', 'Understat_Name', 'FBref_Name']:
                if col in self.master_map.columns:
                    matches = self.master_map[
                        self.master_map[col].apply(lambda x: self.normalize_name(x) == fpl_norm)
                    ]
                    if not matches.empty:
                        return matches.iloc[0].to_dict()
        
        # Search by Understat name
        if understat_name:
            understat_norm = self.normalize_name(understat_name)
            if 'Understat_Name' in self.master_map.columns:
                matches = self.master_map[
                    self.master_map['Understat_Name'].apply(
                        lambda x: self.normalize_name(x) == understat_norm
                    )
                ]
                if not matches.empty:
                    return matches.iloc[0].to_dict()
        
        # Search by FBref name
        if fbref_name:
            fbref_norm = self.normalize_name(fbref_name)
            if 'FBref_Name' in self.master_map.columns:
                matches = self.master_map[
                    self.master_map['FBref_Name'].apply(
                        lambda x: self.normalize_name(x) == fbref_norm
                    )
                ]
                if not matches.empty:
                    return matches.iloc[0].to_dict()
        
        return None
    
    def fuzzy_match(
        self,
        target_name: str,
        candidate_names: List[str],
        threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        Fuzzy matching using SequenceMatcher (Ratcliff-Obershelp algorithm).
        
        Args:
            target_name: Name to match
            candidate_names: List of candidate names
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of (name, similarity_score) tuples, sorted by score
        """
        target_norm = self.normalize_name(target_name)
        matches = []
        
        for candidate in candidate_names:
            candidate_norm = self.normalize_name(candidate)
            
            # SequenceMatcher similarity
            similarity = SequenceMatcher(None, target_norm, candidate_norm).ratio()
            
            # Word-based similarity (Jaccard)
            target_words = set(target_norm.split())
            candidate_words = set(candidate_norm.split())
            
            if target_words and candidate_words:
                jaccard = len(target_words & candidate_words) / len(target_words | candidate_words)
                # Combine both metrics
                combined_score = (similarity * 0.7) + (jaccard * 0.3)
            else:
                combined_score = similarity
            
            if combined_score >= threshold:
                matches.append((candidate, combined_score))
        
        # Sort by score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def resolve_player_entity(
        self,
        fpl_id: int,
        fpl_name: str,
        fpl_team: Optional[str] = None,
        understat_name: Optional[str] = None,
        fbref_name: Optional[str] = None
    ) -> Dict:
        """
        Resolve player entity across all data sources.
        
        Returns:
            Dictionary with resolved IDs and matching status
        """
        result = {
            'fpl_id': fpl_id,
            'fpl_name': fpl_name,
            'fpl_team': fpl_team,
            'matched': False,
            'match_method': None,
            'global_id': None,
            'understat_id': None,
            'fbref_id': None,
            'confidence': 0.0
        }
        
        # Try to find in Master Map
        master_entry = self.find_player_in_master_map(
            fpl_id=fpl_id,
            fpl_name=fpl_name,
            understat_name=understat_name,
            fbref_name=fbref_name
        )
        
        if master_entry:
            result['matched'] = True
            result['match_method'] = 'master_map'
            result['confidence'] = 1.0
            
            # Extract IDs from master map
            if 'FPL' in master_entry and pd.notna(master_entry['FPL']):
                result['global_id'] = int(master_entry['FPL'])
            
            if 'Understat' in master_entry and pd.notna(master_entry['Understat']):
                result['understat_id'] = int(master_entry['Understat'])
            
            if 'FBref' in master_entry and pd.notna(master_entry['FBref']):
                result['fbref_id'] = master_entry['FBref']
            
            return result
        
        # If not found, try fuzzy matching
        if self.master_map is not None:
            # Get all names from master map for fuzzy matching
            all_names = []
            if 'FPL_Name' in self.master_map.columns:
                all_names.extend(self.master_map['FPL_Name'].dropna().tolist())
            if 'Understat_Name' in self.master_map.columns:
                all_names.extend(self.master_map['Understat_Name'].dropna().tolist())
            
            fuzzy_matches = self.fuzzy_match(fpl_name, all_names, threshold=0.75)
            
            if fuzzy_matches:
                best_match = fuzzy_matches[0]
                result['matched'] = True
                result['match_method'] = 'fuzzy_match'
                result['confidence'] = best_match[1]
                result['fuzzy_match_name'] = best_match[0]
                
                # Try to find the matched name in master map
                matched_entry = self.find_player_in_master_map(fpl_name=best_match[0])
                if matched_entry:
                    if 'FPL' in matched_entry and pd.notna(matched_entry['FPL']):
                        result['global_id'] = int(matched_entry['FPL'])
                    if 'Understat' in matched_entry and pd.notna(matched_entry['Understat']):
                        result['understat_id'] = int(matched_entry['Understat'])
                    if 'FBref' in matched_entry and pd.notna(matched_entry['FBref']):
                        result['fbref_id'] = matched_entry['FBref']
        
        # If still not matched, log for manual review
        if not result['matched']:
            self.log_unmatched_player(result)
        
        return result
    
    def log_unmatched_player(self, player_data: Dict):
        """
        Log unmatched player for manual review.
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'fpl_id': player_data.get('fpl_id'),
            'fpl_name': player_data.get('fpl_name'),
            'fpl_team': player_data.get('fpl_team'),
            'understat_name': player_data.get('understat_name'),
            'fbref_name': player_data.get('fbref_name'),
            'suggested_matches': player_data.get('fuzzy_matches', [])
        }
        
        self.unmatched_players.append(log_entry)
        
        # Save to file for manual review
        log_file = "data/unmatched_players.json"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing = json.load(f)
            else:
                existing = []
            
            existing.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save unmatched player log: {str(e)}")
    
    def resolve_players_bulk(
        self,
        players: List[Dict],
        include_fuzzy: bool = True
    ) -> Dict[int, Dict]:
        """
        Resolve multiple players at once.
        
        Args:
            players: List of player dictionaries with fpl_id, fpl_name, etc.
            include_fuzzy: Whether to use fuzzy matching for unmatched players
        
        Returns:
            Dictionary mapping FPL ID to resolved entity data
        """
        resolved = {}
        
        for player in players:
            fpl_id = player.get('id') or player.get('fpl_id')
            fpl_name = player.get('web_name') or player.get('name') or player.get('fpl_name')
            fpl_team = player.get('team_name') or player.get('team')
            
            resolution = self.resolve_player_entity(
                fpl_id=fpl_id,
                fpl_name=fpl_name,
                fpl_team=fpl_team,
                understat_name=player.get('understat_name'),
                fbref_name=player.get('fbref_name')
            )
            
            resolved[fpl_id] = resolution
        
        return resolved
    
    def get_unmatched_players(self) -> List[Dict]:
        """Get list of unmatched players for manual review"""
        return self.unmatched_players
    
    def add_manual_mapping(
        self,
        fpl_id: int,
        fpl_name: str,
        understat_id: Optional[int] = None,
        fbref_id: Optional[str] = None,
        understat_name: Optional[str] = None,
        fbref_name: Optional[str] = None
    ):
        """
        Add manual mapping for unmatched players.
        This can be used to update the master map or create custom mappings.
        """
        mapping = {
            'FPL': fpl_id,
            'FPL_Name': fpl_name,
            'Understat': understat_id if understat_id else pd.NA,
            'Understat_Name': understat_name if understat_name else pd.NA,
            'FBref': fbref_id if fbref_id else pd.NA,
            'FBref_Name': fbref_name if fbref_name else pd.NA,
            'Source': 'Manual',
            'Date_Added': datetime.now().isoformat()
        }
        
        # Add to master map if loaded
        if self.master_map is not None:
            # Check if FPL ID already exists
            if 'FPL' in self.master_map.columns:
                existing = self.master_map[self.master_map['FPL'] == fpl_id]
                if existing.empty:
                    # Add new row
                    new_row = pd.DataFrame([mapping])
                    self.master_map = pd.concat([self.master_map, new_row], ignore_index=True)
                    logger.info(f"Added manual mapping for {fpl_name} (FPL ID: {fpl_id})")
                else:
                    # Update existing
                    idx = existing.index[0]
                    for key, value in mapping.items():
                        if pd.isna(self.master_map.at[idx, key]) and pd.notna(value):
                            self.master_map.at[idx, key] = value
                    logger.info(f"Updated mapping for {fpl_name} (FPL ID: {fpl_id})")
        
        # Save updated map
        if self.master_map is not None:
            os.makedirs(os.path.dirname(self.LOCAL_MAP_PATH), exist_ok=True)
            self.master_map.to_csv(self.LOCAL_MAP_PATH, index=False)
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()