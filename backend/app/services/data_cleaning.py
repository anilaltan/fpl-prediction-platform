"""
Data Cleaning and Normalization Service
Handles:
- DGW/BGW normalization (Double/Blank Gameweek point normalization)
- DefCon floor points calculation for 2025/26 rules
- Type conversion for metrics (ICT, xG, etc.)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCleaningService:
    """
    Service for cleaning and normalizing FPL data.
    Prevents overfitting by normalizing DGW points and ensuring proper data types.
    """
    
    def __init__(self):
        self.defcon_rules = {
            'defender': {
                'block': 1,  # 1 point per block
                'intervention': 1,  # 1 point per intervention (tackle + interception)
                'pass_bonus': 0.1  # 0.1 points per 10 successful passes
            },
            'midfielder': {
                'block': 1,
                'intervention': 1,
                'pass_bonus': 0.1
            },
            'forward': {
                'block': 0,  # Forwards don't get block points
                'intervention': 1,
                'pass_bonus': 0.1
            }
        }
    
    def normalize_dgw_points(
        self,
        points: Union[float, int],
        matches_played: int,
        gameweek_type: str = "normal"  # "normal", "dgw", "bgw"
    ) -> float:
        """
        Normalize points for Double Gameweeks (DGW) and Blank Gameweeks (BGW).
        
        DGW normalization: Divide points by number of matches to prevent
        overfitting where model thinks DGW players are more skilled.
        
        Args:
            points: Total points scored in the gameweek
            matches_played: Number of matches played (1 for normal, 2 for DGW, 0 for BGW)
            gameweek_type: Type of gameweek ("normal", "dgw", "bgw")
        
        Returns:
            Normalized points per match
        """
        if matches_played == 0:
            # Blank Gameweek - return 0 or handle separately
            return 0.0
        
        if matches_played == 1:
            # Normal gameweek - no normalization needed
            return float(points)
        
        # Double Gameweek - normalize by dividing by matches
        normalized = float(points) / matches_played
        return normalized
    
    def normalize_historical_points(
        self,
        historical_data: pd.DataFrame,
        points_column: str = 'points',
        gameweek_column: str = 'gameweek',
        matches_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Normalize historical points data for DGW/BGW.
        
        Args:
            historical_data: DataFrame with historical player data
            points_column: Column name for points
            gameweek_column: Column name for gameweek
            matches_column: Column name for matches played (if available)
        
        Returns:
            DataFrame with normalized points
        """
        df = historical_data.copy()
        
        # Detect DGW/BGW if matches_column not provided
        if matches_column is None:
            # Estimate matches based on minutes (90+ minutes = likely 2 matches)
            if 'minutes' in df.columns:
                df['estimated_matches'] = (df['minutes'] / 90.0).round().astype(int)
                df['estimated_matches'] = df['estimated_matches'].clip(0, 2)
            else:
                # Default: assume 1 match per gameweek
                df['estimated_matches'] = 1
        
        # Normalize points
        if matches_column:
            df['normalized_points'] = df.apply(
                lambda row: self.normalize_dgw_points(
                    row[points_column],
                    row[matches_column],
                    "dgw" if row[matches_column] > 1 else "normal"
                ),
                axis=1
            )
        else:
            df['normalized_points'] = df.apply(
                lambda row: self.normalize_dgw_points(
                    row[points_column],
                    row.get('estimated_matches', 1),
                    "dgw" if row.get('estimated_matches', 1) > 1 else "normal"
                ),
                axis=1
            )
        
        return df
    
    def calculate_defcon_floor_points(
        self,
        player_data: Dict,
        position: str,
        minutes: int = 90,
        use_raw_data: bool = True
    ) -> float:
        """
        Calculate DefCon floor points based on 2025/26 FPL rules.
        
        Floor points represent minimum expected points from defensive contributions:
        - Blocks (defenders/midfielders only)
        - Interventions (tackles + interceptions)
        - Pass bonus (per 10 successful passes)
        
        Args:
            player_data: Player statistics dictionary
            position: Player position ("DEF", "MID", "FWD")
            minutes: Expected minutes played
            use_raw_data: If True, use raw defensive stats; if False, use per-90 averages
        
        Returns:
            Floor points estimate
        """
        if minutes == 0:
            return 0.0
        
        position_lower = position.lower()
        if 'def' in position_lower:
            rules = self.defcon_rules['defender']
        elif 'mid' in position_lower:
            rules = self.defcon_rules['midfielder']
        else:
            rules = self.defcon_rules['forward']
        
        # Extract DefCon stats
        if use_raw_data:
            # Use raw defensive statistics
            blocks = self._safe_float(player_data.get('blocks', player_data.get('total_blocks', 0)))
            tackles = self._safe_float(player_data.get('tackles', player_data.get('total_tackles', 0)))
            interceptions = self._safe_float(player_data.get('interceptions', player_data.get('total_interceptions', 0)))
            passes = self._safe_float(player_data.get('passes', player_data.get('total_passes', player_data.get('successful_passes', 0))))
            
            # Calculate per-90 from raw data
            total_minutes = self._safe_float(player_data.get('minutes', player_data.get('minutes_played', 0)))
            games_played = self._safe_float(player_data.get('games', player_data.get('games_played', 1)))
            
            if total_minutes > 0 and games_played > 0:
                avg_minutes_per_game = total_minutes / games_played
                blocks_per_90 = (blocks / games_played) * (90.0 / max(avg_minutes_per_game, 1.0))
                interventions_per_90 = ((tackles + interceptions) / games_played) * (90.0 / max(avg_minutes_per_game, 1.0))
                passes_per_90 = (passes / games_played) * (90.0 / max(avg_minutes_per_game, 1.0))
            else:
                # Fallback to per-90 if available
                blocks_per_90 = self._safe_float(player_data.get('blocks_per_90', 0.0))
                interventions_per_90 = self._safe_float(player_data.get('interventions_per_90', 0.0))
                passes_per_90 = self._safe_float(player_data.get('passes_per_90', 0.0))
        else:
            # Use per-90 averages directly
            blocks_per_90 = self._safe_float(player_data.get('blocks_per_90', 0.0))
            interventions_per_90 = self._safe_float(player_data.get('interventions_per_90', 0.0))
            passes_per_90 = self._safe_float(player_data.get('passes_per_90', 0.0))
        
        # Scale to expected minutes
        minutes_factor = minutes / 90.0
        
        # Calculate floor points
        floor_points = 0.0
        
        # Block points (defenders and midfielders only)
        if rules['block'] > 0:
            floor_points += blocks_per_90 * rules['block'] * minutes_factor
        
        # Intervention points (tackles + interceptions)
        floor_points += interventions_per_90 * rules['intervention'] * minutes_factor
        
        # Pass bonus (per 10 successful passes)
        floor_points += (passes_per_90 / 10.0) * rules['pass_bonus'] * minutes_factor
        
        return float(max(0.0, floor_points))
    
    def convert_metrics_to_float(
        self,
        data: Union[Dict, pd.DataFrame, List[Dict]],
        metric_columns: Optional[List[str]] = None
    ) -> Union[Dict, pd.DataFrame, List[Dict]]:
        """
        Convert metric columns to float type to prevent calculation errors.
        
        Common metrics that need conversion:
        - ICT Index (influence, creativity, threat)
        - Expected stats (xG, xA, xGI, xGC)
        - Form, PPG, value
        - All numeric statistics
        
        Args:
            data: Dictionary, DataFrame, or list of dictionaries
            metric_columns: Optional list of specific columns to convert.
                          If None, converts all numeric columns.
        
        Returns:
            Data with converted float types
        """
        if isinstance(data, dict):
            return self._convert_dict_to_float(data, metric_columns)
        elif isinstance(data, pd.DataFrame):
            return self._convert_dataframe_to_float(data, metric_columns)
        elif isinstance(data, list):
            return [self._convert_dict_to_float(item, metric_columns) for item in data]
        else:
            return data
    
    def _convert_dict_to_float(
        self,
        data: Dict,
        metric_columns: Optional[List[str]] = None
    ) -> Dict:
        """Convert dictionary values to float"""
        converted = {}
        
        for key, value in data.items():
            if metric_columns and key not in metric_columns:
                converted[key] = value
                continue
            
            # Convert numeric values
            if isinstance(value, (int, float)):
                converted[key] = float(value)
            elif isinstance(value, str):
                # Try to convert string numbers
                try:
                    # Remove commas and whitespace
                    cleaned = value.replace(',', '').strip()
                    converted[key] = float(cleaned)
                except (ValueError, AttributeError):
                    converted[key] = value
            elif value is None or pd.isna(value):
                converted[key] = 0.0
            else:
                converted[key] = value
        
        return converted
    
    def _convert_dataframe_to_float(
        self,
        df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Convert DataFrame columns to float"""
        df_copy = df.copy()
        
        if metric_columns:
            # Convert specific columns
            for col in metric_columns:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0).astype(float)
        else:
            # Convert all numeric columns
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0).astype(float)
        
        return df_copy
    
    def _safe_float(self, value: Any) -> float:
        """
        Safely convert value to float.
        
        Args:
            value: Value to convert
        
        Returns:
            Float value or 0.0 if conversion fails
        """
        if value is None:
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            try:
                # Remove commas and whitespace
                cleaned = value.replace(',', '').strip()
                return float(cleaned)
            except (ValueError, AttributeError):
                return 0.0
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def clean_player_data(
        self,
        player_data: Dict,
        normalize_dgw: bool = True,
        calculate_defcon: bool = True,
        convert_types: bool = True,
        position: Optional[str] = None
    ) -> Dict:
        """
        Comprehensive player data cleaning.
        
        Args:
            player_data: Raw player data dictionary
            normalize_dgw: Whether to normalize DGW points
            calculate_defcon: Whether to calculate DefCon floor points
            convert_types: Whether to convert metrics to float
            position: Player position (required for DefCon calculation)
        
        Returns:
            Cleaned player data dictionary
        """
        cleaned = player_data.copy()
        
        # Type conversion (do this first)
        if convert_types:
            cleaned = self.convert_metrics_to_float(cleaned)
        
        # DGW normalization
        if normalize_dgw and 'points' in cleaned:
            matches_played = cleaned.get('matches_played', cleaned.get('matches', 1))
            cleaned['normalized_points'] = self.normalize_dgw_points(
                cleaned['points'],
                matches_played
            )
            cleaned['original_points'] = cleaned['points']  # Keep original
            cleaned['points'] = cleaned['normalized_points']  # Use normalized
        
        # DefCon floor points
        if calculate_defcon and position:
            minutes = cleaned.get('minutes', cleaned.get('expected_minutes', 90))
            floor_points = self.calculate_defcon_floor_points(
                cleaned,
                position,
                minutes,
                use_raw_data=True
            )
            cleaned['defcon_floor_points'] = floor_points
            cleaned['defcon_blocks_per_90'] = self._safe_float(cleaned.get('blocks_per_90', 0.0))
            cleaned['defcon_interventions_per_90'] = self._safe_float(cleaned.get('interventions_per_90', 0.0))
            cleaned['defcon_passes_per_90'] = self._safe_float(cleaned.get('passes_per_90', 0.0))
        
        return cleaned
    
    def clean_bulk_player_data(
        self,
        players_data: List[Dict],
        normalize_dgw: bool = True,
        calculate_defcon: bool = True,
        convert_types: bool = True
    ) -> List[Dict]:
        """
        Clean multiple players' data at once.
        
        Args:
            players_data: List of player data dictionaries
            normalize_dgw: Whether to normalize DGW points
            calculate_defcon: Whether to calculate DefCon floor points
            convert_types: Whether to convert metrics to float
        
        Returns:
            List of cleaned player data dictionaries
        """
        cleaned_players = []
        
        for player in players_data:
            position = player.get('position', player.get('element_type', 'MID'))
            cleaned = self.clean_player_data(
                player,
                normalize_dgw=normalize_dgw,
                calculate_defcon=calculate_defcon,
                convert_types=convert_types,
                position=position
            )
            cleaned_players.append(cleaned)
        
        return cleaned_players
    
    def get_defcon_metrics(
        self,
        player_data: Dict,
        position: str
    ) -> Dict[str, float]:
        """
        Extract all DefCon-related metrics for a player.
        
        Returns:
            Dictionary with DefCon metrics
        """
        minutes = player_data.get('minutes', player_data.get('expected_minutes', 90))
        
        return {
            'floor_points': self.calculate_defcon_floor_points(player_data, position, minutes),
            'floor_points_90': self.calculate_defcon_floor_points(player_data, position, 90),
            'blocks_per_90': self._safe_float(player_data.get('blocks_per_90', 0.0)),
            'interventions_per_90': self._safe_float(player_data.get('interventions_per_90', 0.0)),
            'passes_per_90': self._safe_float(player_data.get('passes_per_90', 0.0)),
            'defcon_score': self.calculate_defcon_floor_points(player_data, position, 90)
        }