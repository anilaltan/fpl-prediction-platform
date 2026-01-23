"""
Entity Resolution Service for FPL Player ID Mapping
Integrates Master ID Map from ChrisMusson/FPL-ID-Map GitHub repository
Handles name variations and fuzzy matching for unmatched players
"""
import httpx
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging
import re
from datetime import datetime, timezone
import json
from io import StringIO
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from decimal import Decimal
from .fuzzy_matching import FuzzyMatcher
from app.models import EntityMapping

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
    
    def generate_canonical_name(
        self,
        fpl_name: Optional[str] = None,
        understat_name: Optional[str] = None,
        fbref_name: Optional[str] = None
    ) -> str:
        """
        Generate a canonical (standardized) player name from multiple sources.
        
        Strategy:
        1. Prefer FPL name if available (most authoritative source)
        2. If FPL name not available, use the most common name among available sources
        3. Handle special characters, accents, and common variations
        4. Ensure consistency: same player always gets same canonical name
        
        Args:
            fpl_name: FPL player name (preferred source)
            understat_name: Understat player name
            fbref_name: FBref player name
        
        Returns:
            Canonical player name (standardized, title-cased)
        """
        # Collect all available names (non-empty, non-NaN)
        available_names = []
        if fpl_name and not pd.isna(fpl_name) and str(fpl_name).strip():
            available_names.append(('fpl', str(fpl_name).strip()))
        if understat_name and not pd.isna(understat_name) and str(understat_name).strip():
            available_names.append(('understat', str(understat_name).strip()))
        if fbref_name and not pd.isna(fbref_name) and str(fbref_name).strip():
            available_names.append(('fbref', str(fbref_name).strip()))
        
        if not available_names:
            logger.warning("No valid names provided for canonical name generation")
            return ""
        
        # Strategy 1: Prefer FPL name if available
        fpl_entry = next((name for source, name in available_names if source == 'fpl'), None)
        if fpl_entry:
            canonical = self._clean_canonical_name(fpl_entry)
            if canonical:
                return canonical
        
        # Strategy 2: Use most common name (if multiple sources agree)
        # Count occurrences of normalized names
        normalized_counts = Counter()
        name_map = {}
        
        for source, name in available_names:
            normalized = self.normalize_name(name)
            if normalized:
                normalized_counts[normalized] += 1
                # Keep the first (most complete) original name for each normalized version
                if normalized not in name_map:
                    name_map[normalized] = name
        
        # If we have a name that appears in multiple sources, use it
        if normalized_counts:
            most_common_normalized, count = normalized_counts.most_common(1)[0]
            if count > 1 or len(available_names) == 1:
                # Use the original name from the map
                canonical = self._clean_canonical_name(name_map[most_common_normalized])
                if canonical:
                    return canonical
        
        # Strategy 3: Use the longest/most complete name (likely most accurate)
        longest_name = max(available_names, key=lambda x: len(x[1]))[1]
        canonical = self._clean_canonical_name(longest_name)
        if canonical:
            return canonical
        
        # Fallback: use first available name
        return self._clean_canonical_name(available_names[0][1])
    
    def _clean_canonical_name(self, name: str) -> str:
        """
        Clean and standardize a name for canonical use.
        Preserves proper capitalization and handles special characters.
        
        Args:
            name: Raw player name
        
        Returns:
            Cleaned, title-cased canonical name
        """
        if not name or pd.isna(name):
            return ""
        
        name = str(name).strip()
        if not name:
            return ""
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Handle common abbreviations and variations
        # Remove trailing dots from abbreviations (e.g., "B. Fernandes" -> "B Fernandes")
        name = re.sub(r'\b([A-Z])\.\s+', r'\1 ', name)
        
        # Remove common suffixes that shouldn't be in canonical name
        # But keep them if they're part of the actual name (like "Jr.")
        # Only remove if it's clearly a suffix pattern
        name = re.sub(r'\s+[Jr]\.?\s*$', '', name, flags=re.IGNORECASE)
        
        # Title case the name (capitalize first letter of each word)
        # But preserve existing capitalization for known patterns
        words = name.split()
        cleaned_words = []
        
        for word in words:
            # If word is all caps (like "DE" in "Kevin DE Bruyne"), keep it
            if word.isupper() and len(word) <= 3:
                cleaned_words.append(word)
            # If word starts with lowercase (like "van", "de", "da"), keep it lowercase
            elif word.lower() in ['van', 'de', 'da', 'del', 'der', 'den', 'von', 'le', 'la', 'el']:
                cleaned_words.append(word.lower())
            # Otherwise, title case
            else:
                cleaned_words.append(word.capitalize())
        
        canonical = ' '.join(cleaned_words)
        
        # Final cleanup: ensure proper spacing
        canonical = ' '.join(canonical.split())
        
        return canonical.strip()
    
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
        Fuzzy matching using RapidFuzz (faster and more accurate than SequenceMatcher).
        
        Args:
            target_name: Name to match
            candidate_names: List of candidate names
            threshold: Minimum similarity score (0-1, converted to 0-100 for RapidFuzz)
        
        Returns:
            List of (name, similarity_score) tuples, sorted by score (0-1 scale)
        """
        if not candidate_names:
            return []
        
        # Normalize target name
        target_norm = self.normalize_name(target_name)
        
        # Convert threshold from 0-1 scale to 0-100 scale for RapidFuzz
        threshold_100 = threshold * 100.0
        
        # Use RapidFuzz for efficient batch matching with weighted ratio
        # This combines multiple algorithms for best accuracy
        matches = FuzzyMatcher.find_best_matches(
            target=target_norm,
            candidates=[self.normalize_name(c) for c in candidate_names],
            limit=len(candidate_names),  # Get all matches above threshold
            threshold=threshold_100,
            use_weighted=True  # Use weighted ratio for best accuracy
        )
        
        # Map back to original candidate names and convert scores to 0-1 scale
        # Create a mapping from normalized to original names
        norm_to_original = {
            self.normalize_name(c): c for c in candidate_names
        }
        
        result = []
        for norm_candidate, score_100 in matches:
            original_candidate = norm_to_original.get(norm_candidate, norm_candidate)
            score_01 = score_100 / 100.0  # Convert back to 0-1 scale
            result.append((original_candidate, score_01))
        
        return result
    
    def fuzzy_match_multi_source(
        self,
        target_name: str,
        understat_names: Optional[List[str]] = None,
        fbref_names: Optional[List[str]] = None,
        threshold: float = 0.85
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Enhanced fuzzy matching against multiple data sources (Understat and FBref).
        Uses RapidFuzz with weighted ratio for best accuracy.
        
        Args:
            target_name: Name to match (typically FPL player name)
            understat_names: List of Understat player names to match against
            fbref_names: List of FBref player names to match against
            threshold: Minimum similarity score (0-1 scale). Default 0.85 for high-confidence matches.
        
        Returns:
            Dictionary with keys 'understat' and 'fbref', each containing:
            List of (matched_name, confidence_score) tuples, sorted by score (descending).
            Confidence scores are in 0.0-1.0 scale.
        """
        result = {
            'understat': [],
            'fbref': []
        }
        
        # Normalize target name
        target_norm = self.normalize_name(target_name)
        
        # Convert threshold from 0-1 scale to 0-100 scale for RapidFuzz
        threshold_100 = threshold * 100.0
        
        # Match against Understat names
        if understat_names:
            understat_matches = FuzzyMatcher.find_best_matches(
                target=target_norm,
                candidates=[self.normalize_name(c) for c in understat_names],
                limit=len(understat_names),
                threshold=threshold_100,
                use_weighted=True
            )
            
            # Map back to original names and convert scores to 0-1 scale
            norm_to_original_understat = {
                self.normalize_name(c): c for c in understat_names
            }
            
            for norm_candidate, score_100 in understat_matches:
                original_candidate = norm_to_original_understat.get(norm_candidate, norm_candidate)
                score_01 = score_100 / 100.0
                result['understat'].append((original_candidate, score_01))
        
        # Match against FBref names
        if fbref_names:
            fbref_matches = FuzzyMatcher.find_best_matches(
                target=target_norm,
                candidates=[self.normalize_name(c) for c in fbref_names],
                limit=len(fbref_names),
                threshold=threshold_100,
                use_weighted=True
            )
            
            # Map back to original names and convert scores to 0-1 scale
            norm_to_original_fbref = {
                self.normalize_name(c): c for c in fbref_names
            }
            
            for norm_candidate, score_100 in fbref_matches:
                original_candidate = norm_to_original_fbref.get(norm_candidate, norm_candidate)
                score_01 = score_100 / 100.0
                result['fbref'].append((original_candidate, score_01))
        
        return result
    
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
                try:
                    result['global_id'] = int(master_entry['FPL'])
                except (ValueError, TypeError):
                    pass
            
            if 'Understat' in master_entry and pd.notna(master_entry['Understat']):
                try:
                    result['understat_id'] = int(master_entry['Understat'])
                except (ValueError, TypeError):
                    pass
            
            if 'FBref' in master_entry and pd.notna(master_entry['FBref']):
                try:
                    # FBref ID might be string or number
                    fbref_val = master_entry['FBref']
                    if pd.notna(fbref_val):
                        result['fbref_id'] = str(fbref_val) if not isinstance(fbref_val, str) else fbref_val
                except (ValueError, TypeError):
                    pass
            
            return result
        
        # If not found, try fuzzy matching with multi-source matching
        if self.master_map is not None:
            # Get separate name lists for Understat and FBref
            understat_names = []
            fbref_names = []
            all_names = []
            
            if 'FPL_Name' in self.master_map.columns:
                all_names.extend(self.master_map['FPL_Name'].dropna().tolist())
            if 'Understat_Name' in self.master_map.columns:
                understat_names = self.master_map['Understat_Name'].dropna().tolist()
                all_names.extend(understat_names)
            if 'FBref_Name' in self.master_map.columns:
                fbref_names = self.master_map['FBref_Name'].dropna().tolist()
                all_names.extend(fbref_names)
            
            # Use multi-source matching with 0.85 threshold for high-confidence matches
            multi_source_matches = self.fuzzy_match_multi_source(
                target_name=fpl_name,
                understat_names=understat_names if understat_names else None,
                fbref_names=fbref_names if fbref_names else None,
                threshold=0.85  # High-confidence threshold
            )
            
            # Check for matches in any source
            best_match = None
            best_score = 0.0
            match_source = None
            
            # Check Understat matches
            if multi_source_matches['understat']:
                understat_match = multi_source_matches['understat'][0]
                if understat_match[1] > best_score:
                    best_match = understat_match[0]
                    best_score = understat_match[1]
                    match_source = 'understat'
            
            # Check FBref matches
            if multi_source_matches['fbref']:
                fbref_match = multi_source_matches['fbref'][0]
                if fbref_match[1] > best_score:
                    best_match = fbref_match[0]
                    best_score = fbref_match[1]
                    match_source = 'fbref'
            
            # Fallback to general fuzzy matching if no multi-source match found
            if not best_match:
                fuzzy_matches = self.fuzzy_match(fpl_name, all_names, threshold=0.75)
                if fuzzy_matches:
                    best_match = fuzzy_matches[0][0]
                    best_score = fuzzy_matches[0][1]
                    match_source = 'general'
            
            if best_match:
                result['matched'] = True
                result['match_method'] = f'fuzzy_match_{match_source}' if match_source else 'fuzzy_match'
                result['confidence'] = best_score
                result['fuzzy_match_name'] = best_match
                
                # Try to find the matched name in master map
                matched_entry = self.find_player_in_master_map(fpl_name=best_match)
                if matched_entry:
                    if 'FPL' in matched_entry and pd.notna(matched_entry['FPL']):
                        try:
                            result['global_id'] = int(matched_entry['FPL'])
                        except (ValueError, TypeError):
                            pass
                    if 'Understat' in matched_entry and pd.notna(matched_entry['Understat']):
                        try:
                            result['understat_id'] = int(matched_entry['Understat'])
                        except (ValueError, TypeError):
                            pass
                    if 'FBref' in matched_entry and pd.notna(matched_entry['FBref']):
                        try:
                            fbref_val = matched_entry['FBref']
                            if pd.notna(fbref_val):
                                result['fbref_id'] = str(fbref_val) if not isinstance(fbref_val, str) else fbref_val
                        except (ValueError, TypeError):
                            pass
        
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
    
    # ============================================================================
    # DATABASE PERSISTENCE METHODS
    # ============================================================================
    
    def upsert_mapping(
        self,
        db: Session,
        fpl_id: int,
        canonical_name: Optional[str] = None,
        understat_name: Optional[str] = None,
        fbref_name: Optional[str] = None,
        fpl_name: Optional[str] = None,
        confidence_score: Optional[float] = None,
        manually_verified: bool = False
    ) -> EntityMapping:
        """
        Insert or update an entity mapping in the database.
        Automatically generates canonical name if not provided.
        
        Args:
            db: Database session
            fpl_id: FPL player ID (required, unique)
            canonical_name: Canonical player name (optional, will be generated if not provided)
            understat_name: Understat player name (optional)
            fbref_name: FBref player name (optional)
            fpl_name: FPL player name (optional, used for canonical name generation)
            confidence_score: Confidence score 0.0-1.0 (optional)
            manually_verified: Whether mapping is manually verified (default: False)
        
        Returns:
            EntityMapping object (created or updated)
        
        Raises:
            IntegrityError: If fpl_id constraint is violated
            ValueError: If canonical name cannot be generated (no names provided)
        """
        try:
            # Generate canonical name if not provided
            if not canonical_name:
                canonical_name = self.generate_canonical_name(
                    fpl_name=fpl_name,
                    understat_name=understat_name,
                    fbref_name=fbref_name
                )
                if not canonical_name:
                    raise ValueError(
                        f"Cannot generate canonical name for FPL ID {fpl_id}: "
                        f"no valid names provided (fpl_name={fpl_name}, "
                        f"understat_name={understat_name}, fbref_name={fbref_name})"
                    )
            
            # Try to find existing mapping
            existing = db.query(EntityMapping).filter(EntityMapping.fpl_id == fpl_id).first()
            
            if existing:
                # Update existing mapping
                existing.understat_name = understat_name if understat_name else existing.understat_name
                existing.fbref_name = fbref_name if fbref_name else existing.fbref_name
                existing.canonical_name = canonical_name
                if confidence_score is not None:
                    existing.confidence_score = Decimal(str(confidence_score))
                existing.manually_verified = manually_verified
                existing.updated_at = datetime.now(timezone.utc)
                
                db.commit()
                db.refresh(existing)
                logger.info(f"Updated entity mapping for FPL ID {fpl_id}: {canonical_name}")
                return existing
            else:
                # Create new mapping
                new_mapping = EntityMapping(
                    fpl_id=fpl_id,
                    canonical_name=canonical_name,
                    understat_name=understat_name,
                    fbref_name=fbref_name,
                    confidence_score=Decimal(str(confidence_score)) if confidence_score is not None else None,
                    manually_verified=manually_verified
                )
                
                db.add(new_mapping)
                db.commit()
                db.refresh(new_mapping)
                logger.info(f"Created entity mapping for FPL ID {fpl_id}: {canonical_name}")
                return new_mapping
                
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Integrity error upserting mapping for FPL ID {fpl_id}: {str(e)}")
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Error upserting mapping for FPL ID {fpl_id}: {str(e)}")
            raise
    
    def get_mapping_by_fpl_id(
        self,
        db: Session,
        fpl_id: int
    ) -> Optional[EntityMapping]:
        """
        Get entity mapping by FPL ID.
        
        Args:
            db: Database session
            fpl_id: FPL player ID
        
        Returns:
            EntityMapping object if found, None otherwise
        """
        return db.query(EntityMapping).filter(EntityMapping.fpl_id == fpl_id).first()
    
    def get_mapping_by_canonical_name(
        self,
        db: Session,
        canonical_name: str
    ) -> Optional[EntityMapping]:
        """
        Get entity mapping by canonical name.
        
        Args:
            db: Database session
            canonical_name: Canonical player name
        
        Returns:
            EntityMapping object if found, None otherwise
        """
        return db.query(EntityMapping).filter(
            EntityMapping.canonical_name == canonical_name
        ).first()
    
    def get_mapping_by_understat_name(
        self,
        db: Session,
        understat_name: str
    ) -> Optional[EntityMapping]:
        """
        Get entity mapping by Understat name.
        
        Args:
            db: Database session
            understat_name: Understat player name
        
        Returns:
            EntityMapping object if found, None otherwise
        """
        return db.query(EntityMapping).filter(
            EntityMapping.understat_name == understat_name
        ).first()
    
    def get_mapping_by_fbref_name(
        self,
        db: Session,
        fbref_name: str
    ) -> Optional[EntityMapping]:
        """
        Get entity mapping by FBref name.
        
        Args:
            db: Database session
            fbref_name: FBref player name
        
        Returns:
            EntityMapping object if found, None otherwise
        """
        return db.query(EntityMapping).filter(
            EntityMapping.fbref_name == fbref_name
        ).first()
    
    def get_all_mappings(
        self,
        db: Session,
        limit: Optional[int] = None,
        offset: int = 0,
        manually_verified_only: bool = False
    ) -> List[EntityMapping]:
        """
        Get all entity mappings with optional filtering.
        
        Args:
            db: Database session
            limit: Maximum number of results to return
            offset: Number of results to skip
            manually_verified_only: If True, only return manually verified mappings
        
        Returns:
            List of EntityMapping objects
        """
        query = db.query(EntityMapping)
        
        if manually_verified_only:
            query = query.filter(EntityMapping.manually_verified == True)
        
        query = query.order_by(EntityMapping.fpl_id)
        
        if offset > 0:
            query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def update_confidence_score(
        self,
        db: Session,
        fpl_id: int,
        confidence_score: float
    ) -> Optional[EntityMapping]:
        """
        Update confidence score for an existing mapping.
        
        Args:
            db: Database session
            fpl_id: FPL player ID
            confidence_score: New confidence score (0.0-1.0)
        
        Returns:
            Updated EntityMapping object if found, None otherwise
        
        Raises:
            ValueError: If confidence_score is not in range 0.0-1.0
        """
        if not (0.0 <= confidence_score <= 1.0):
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {confidence_score}")
        
        mapping = self.get_mapping_by_fpl_id(db, fpl_id)
        if mapping:
            mapping.confidence_score = Decimal(str(confidence_score))
            mapping.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(mapping)
            logger.info(f"Updated confidence score for FPL ID {fpl_id}: {confidence_score}")
            return mapping
        else:
            logger.warning(f"No mapping found for FPL ID {fpl_id}")
            return None
    
    def mark_manually_verified(
        self,
        db: Session,
        fpl_id: int,
        verified: bool = True
    ) -> Optional[EntityMapping]:
        """
        Mark a mapping as manually verified (or unverified).
        
        Args:
            db: Database session
            fpl_id: FPL player ID
            verified: Whether to mark as verified (default: True)
        
        Returns:
            Updated EntityMapping object if found, None otherwise
        """
        mapping = self.get_mapping_by_fpl_id(db, fpl_id)
        if mapping:
            mapping.manually_verified = verified
            if verified:
                # Set confidence score to 1.0 when manually verified
                mapping.confidence_score = Decimal('1.0')
            mapping.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(mapping)
            logger.info(f"Marked mapping for FPL ID {fpl_id} as {'verified' if verified else 'unverified'}")
            return mapping
        else:
            logger.warning(f"No mapping found for FPL ID {fpl_id}")
            return None
    
    def override_mapping(
        self,
        db: Session,
        fpl_id: int,
        understat_name: Optional[str] = None,
        fbref_name: Optional[str] = None,
        fpl_name: Optional[str] = None,
        canonical_name: Optional[str] = None
    ) -> EntityMapping:
        """
        Manually override/correct an entity mapping.
        Used for low-confidence matches (score < 0.85) that need manual correction.
        
        This method:
        - Updates the mapping with corrected names
        - Marks as manually_verified=True
        - Sets confidence_score to 1.0
        - Validates to prevent duplicate mappings
        - Generates canonical name if not provided
        
        Args:
            db: Database session
            fpl_id: FPL player ID (required)
            understat_name: Corrected Understat player name (optional)
            fbref_name: Corrected FBref player name (optional)
            fpl_name: FPL player name (optional, used for canonical name generation)
            canonical_name: Canonical player name (optional, will be generated if not provided)
        
        Returns:
            Updated EntityMapping object
        
        Raises:
            ValueError: If mapping not found or duplicate mapping detected
            IntegrityError: If database constraint violation
        """
        # Get existing mapping
        existing_mapping = self.get_mapping_by_fpl_id(db, fpl_id)
        if not existing_mapping:
            raise ValueError(f"No mapping found for FPL ID {fpl_id}. Create mapping first using upsert_mapping.")
        
        # Validate: Check for duplicate mappings (same names but different FPL ID)
        if understat_name:
            duplicate_understat = self.get_mapping_by_understat_name(db, understat_name)
            if duplicate_understat and duplicate_understat.fpl_id != fpl_id:
                raise ValueError(
                    f"Duplicate mapping detected: Understat name '{understat_name}' "
                    f"already mapped to FPL ID {duplicate_understat.fpl_id}"
                )
        
        if fbref_name:
            duplicate_fbref = self.get_mapping_by_fbref_name(db, fbref_name)
            if duplicate_fbref and duplicate_fbref.fpl_id != fpl_id:
                raise ValueError(
                    f"Duplicate mapping detected: FBref name '{fbref_name}' "
                    f"already mapped to FPL ID {duplicate_fbref.fpl_id}"
                )
        
        # Update mapping fields
        if understat_name is not None:
            existing_mapping.understat_name = understat_name
        if fbref_name is not None:
            existing_mapping.fbref_name = fbref_name
        
        # Generate canonical name if not provided
        if not canonical_name:
            canonical_name = self.generate_canonical_name(
                fpl_name=fpl_name or existing_mapping.canonical_name,
                understat_name=understat_name or existing_mapping.understat_name,
                fbref_name=fbref_name or existing_mapping.fbref_name
            )
            if not canonical_name:
                # Fallback to existing canonical name
                canonical_name = existing_mapping.canonical_name
        
        existing_mapping.canonical_name = canonical_name
        
        # Mark as manually verified and set confidence to 1.0
        existing_mapping.manually_verified = True
        existing_mapping.confidence_score = Decimal('1.0')
        existing_mapping.updated_at = datetime.now(timezone.utc)
        
        try:
            db.commit()
            db.refresh(existing_mapping)
            logger.info(
                f"Manually overridden mapping for FPL ID {fpl_id}: "
                f"canonical_name={canonical_name}, "
                f"understat_name={understat_name or existing_mapping.understat_name}, "
                f"fbref_name={fbref_name or existing_mapping.fbref_name}"
            )
            return existing_mapping
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Integrity error overriding mapping for FPL ID {fpl_id}: {str(e)}")
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Error overriding mapping for FPL ID {fpl_id}: {str(e)}")
            raise
    
    def delete_mapping(
        self,
        db: Session,
        fpl_id: int
    ) -> bool:
        """
        Delete an entity mapping from the database.
        
        Args:
            db: Database session
            fpl_id: FPL player ID
        
        Returns:
            True if deleted, False if not found
        """
        mapping = self.get_mapping_by_fpl_id(db, fpl_id)
        if mapping:
            db.delete(mapping)
            db.commit()
            logger.info(f"Deleted entity mapping for FPL ID {fpl_id}")
            return True
        else:
            logger.warning(f"No mapping found for FPL ID {fpl_id}")
            return False
    
    def resolve_all_players(
        self,
        db: Session,
        fpl_players: List[Dict],
        store_mappings: bool = True
    ) -> Dict:
        """
        Process all FPL players and resolve entity mappings.
        Stores mappings in database and generates a comprehensive report.
        
        Args:
            db: Database session
            fpl_players: List of FPL player dictionaries (from bootstrap data)
            store_mappings: Whether to store mappings in database (default: True)
        
        Returns:
            Dictionary with resolution report containing:
            - total_players: Total number of players processed
            - matched_count: Number of successfully matched players
            - unmatched_count: Number of unmatched players
            - high_confidence_count: Number of high-confidence matches (>= 0.85)
            - low_confidence_count: Number of low-confidence matches (< 0.85)
            - manually_verified_count: Number of manually verified mappings
            - low_confidence_mappings: List of low-confidence mappings for review
            - unmatched_players: List of unmatched players
            - match_accuracy: Overall match accuracy percentage
        """
        report = {
            'total_players': len(fpl_players),
            'matched_count': 0,
            'unmatched_count': 0,
            'high_confidence_count': 0,
            'low_confidence_count': 0,
            'manually_verified_count': 0,
            'low_confidence_mappings': [],
            'unmatched_players': [],
            'match_accuracy': 0.0,
            'mappings_stored': 0
        }
        
        logger.info(f"Starting bulk resolution for {len(fpl_players)} players")
        
        # Ensure master map is loaded
        if self.master_map is None:
            logger.warning("Master ID Map not loaded. Resolution may be incomplete.")
        
        for player in fpl_players:
            fpl_id = player.get('id') or player.get('fpl_id')
            fpl_name = player.get('web_name') or player.get('name') or player.get('fpl_name')
            fpl_team = player.get('team_name') or player.get('team')
            
            if not fpl_id or not fpl_name:
                logger.warning(f"Skipping player with missing ID or name: {player}")
                continue
            
            try:
                # Resolve player entity
                resolution = self.resolve_player_entity(
                    fpl_id=fpl_id,
                    fpl_name=fpl_name,
                    fpl_team=fpl_team,
                    understat_name=player.get('understat_name'),
                    fbref_name=player.get('fbref_name')
                )
                
                if resolution.get('matched'):
                    report['matched_count'] += 1
                    confidence = resolution.get('confidence', 0.0)
                    
                    if confidence >= 0.85:
                        report['high_confidence_count'] += 1
                    else:
                        report['low_confidence_count'] += 1
                        # Add to low-confidence list for manual review
                        report['low_confidence_mappings'].append({
                            'fpl_id': fpl_id,
                            'fpl_name': fpl_name,
                            'fpl_team': fpl_team,
                            'match_method': resolution.get('match_method'),
                            'confidence': confidence,
                            'fuzzy_match_name': resolution.get('fuzzy_match_name'),
                            'understat_id': resolution.get('understat_id'),
                            'fbref_id': resolution.get('fbref_id')
                        })
                    
                    # Store mapping in database if requested
                    if store_mappings:
                        try:
                            # Extract names from resolution
                            understat_name = None
                            fbref_name = None
                            
                            # Try to get names from master map if available
                            if self.master_map is not None:
                                master_entry = self.find_player_in_master_map(
                                    fpl_id=fpl_id,
                                    fpl_name=fpl_name
                                )
                                if master_entry:
                                    if 'Understat_Name' in master_entry and pd.notna(master_entry['Understat_Name']):
                                        understat_name = str(master_entry['Understat_Name'])
                                    if 'FBref_Name' in master_entry and pd.notna(master_entry['FBref_Name']):
                                        fbref_name = str(master_entry['FBref_Name'])
                            
                            # Generate canonical name
                            canonical_name = self.generate_canonical_name(
                                fpl_name=fpl_name,
                                understat_name=understat_name,
                                fbref_name=fbref_name
                            )
                            
                            # Upsert mapping
                            mapping = self.upsert_mapping(
                                db=db,
                                fpl_id=fpl_id,
                                fpl_name=fpl_name,
                                understat_name=understat_name,
                                fbref_name=fbref_name,
                                canonical_name=canonical_name,
                                confidence_score=confidence,
                                manually_verified=False
                            )
                            
                            report['mappings_stored'] += 1
                            
                            # Check if manually verified
                            if mapping.manually_verified:
                                report['manually_verified_count'] += 1
                            
                        except Exception as e:
                            logger.error(f"Error storing mapping for FPL ID {fpl_id}: {str(e)}")
                else:
                    report['unmatched_count'] += 1
                    report['unmatched_players'].append({
                        'fpl_id': fpl_id,
                        'fpl_name': fpl_name,
                        'fpl_team': fpl_team,
                        'suggested_matches': resolution.get('fuzzy_matches', [])
                    })
                    
            except Exception as e:
                error_msg = str(e) if e else "Unknown error"
                # Handle pandas NA values in error messages
                if pd.isna(error_msg) or error_msg == '<NA>':
                    error_msg = "Data conversion error (pandas NA value)"
                logger.error(f"Error resolving player {fpl_name} (FPL ID {fpl_id}): {error_msg}")
                report['unmatched_count'] += 1
                report['unmatched_players'].append({
                    'fpl_id': fpl_id,
                    'fpl_name': fpl_name,
                    'fpl_team': fpl_team,
                    'error': error_msg
                })
        
        # Calculate match accuracy
        if report['total_players'] > 0:
            report['match_accuracy'] = (report['matched_count'] / report['total_players']) * 100.0
        
        logger.info(
            f"Bulk resolution completed: "
            f"{report['matched_count']}/{report['total_players']} matched "
            f"({report['match_accuracy']:.2f}%), "
            f"{report['high_confidence_count']} high-confidence, "
            f"{report['low_confidence_count']} low-confidence, "
            f"{report['unmatched_count']} unmatched"
        )
        
        return report
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()