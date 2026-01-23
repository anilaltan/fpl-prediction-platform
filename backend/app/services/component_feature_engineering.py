"""
Component Model Feature Engineering Service
Prepares datasets for xMins, Attack, and Defense models with:
- DefCon features (blocks, tackles/interventions, interceptions)
- Lag features for xG, xA, and clean sheet history
- Entity mapping alignment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
import logging

from app.models import EntityMapping

logger = logging.getLogger(__name__)


class ComponentFeatureEngineering:
    """
    Feature engineering service for component-based ML models.
    Prepares features for:
    - xMins model (XGBoost Classifier)
    - Attack model (LightGBM for xG/xA)
    - Defense model (LightGBM for P_CS)
    """

    def __init__(self, db_session: Session):
        """
        Initialize feature engineering service.

        Args:
            db_session: Database session for querying player stats
        """
        self.db_session = db_session
        self.entity_mappings: Optional[Dict[int, EntityMapping]] = None

    def load_entity_mappings(self) -> Dict[int, EntityMapping]:
        """
        Load entity mappings to ensure feature alignment.

        Returns:
            Dictionary mapping fpl_id to EntityMapping
        """
        if self.entity_mappings is None:
            mappings = self.db_session.query(EntityMapping).all()
            self.entity_mappings = {mapping.fpl_id: mapping for mapping in mappings}
            logger.info(f"Loaded {len(self.entity_mappings)} entity mappings")
        return self.entity_mappings

    def extract_defcon_features(
        self,
        player_stats: pd.DataFrame,
        fpl_id: int,
        gameweek: int,
        season: str = "2025-26",
    ) -> Dict[str, float]:
        """
        Extract DefCon features (blocks, tackles/interventions, interceptions) from player_stats.

        Args:
            player_stats: DataFrame with player gameweek stats
            fpl_id: Player FPL ID
            gameweek: Current gameweek
            season: Season identifier

        Returns:
            Dictionary with DefCon features
        """
        # Get player's historical stats up to current gameweek
        player_data = player_stats[
            (player_stats["fpl_id"] == fpl_id)
            & (player_stats["gameweek"] < gameweek)
            & (player_stats["season"] == season)
        ].copy()

        if len(player_data) == 0:
            # No historical data, return defaults
            return {
                "blocks_per_90": 0.0,
                "interventions_per_90": 0.0,
                "passes_per_90": 0.0,
                "defcon_floor_points": 0.0,
                "avg_blocks": 0.0,
                "avg_interventions": 0.0,
                "avg_passes": 0.0,
            }

        # Calculate per-90 averages
        total_minutes = player_data["minutes"].sum()
        if total_minutes == 0:
            return {
                "blocks_per_90": 0.0,
                "interventions_per_90": 0.0,
                "passes_per_90": 0.0,
                "defcon_floor_points": 0.0,
                "avg_blocks": 0.0,
                "avg_interventions": 0.0,
                "avg_passes": 0.0,
            }

        total_blocks = player_data["blocks"].sum()
        total_interventions = player_data["interventions"].sum()
        total_passes = player_data["passes"].sum()

        # Per-90 calculations
        games_played = len(player_data[player_data["minutes"] > 0])
        if games_played == 0:
            games_played = 1

        blocks_per_90 = (total_blocks / games_played) * (
            90.0 / max(total_minutes / games_played, 1.0)
        )
        interventions_per_90 = (total_interventions / games_played) * (
            90.0 / max(total_minutes / games_played, 1.0)
        )
        passes_per_90 = (total_passes / games_played) * (
            90.0 / max(total_minutes / games_played, 1.0)
        )

        # Average per game
        avg_blocks = total_blocks / games_played if games_played > 0 else 0.0
        avg_interventions = (
            total_interventions / games_played if games_played > 0 else 0.0
        )
        avg_passes = total_passes / games_played if games_played > 0 else 0.0

        # DefCon floor points calculation (simplified)
        # Blocks: 1 point per block (DEF/MID only)
        # Interventions: 1 point per intervention
        # Passes: 0.1 points per 10 successful passes
        defcon_floor_points = (
            avg_blocks * 1.0  # Blocks
            + avg_interventions * 1.0  # Interventions
            + (avg_passes / 10.0) * 0.1  # Pass bonus
        )

        return {
            "blocks_per_90": float(blocks_per_90),
            "interventions_per_90": float(interventions_per_90),
            "passes_per_90": float(passes_per_90),
            "defcon_floor_points": float(defcon_floor_points),
            "avg_blocks": float(avg_blocks),
            "avg_interventions": float(avg_interventions),
            "avg_passes": float(avg_passes),
        }

    def create_lag_features(
        self,
        player_stats: pd.DataFrame,
        fpl_id: int,
        gameweek: int,
        season: str = "2025-26",
        lag_periods: List[int] = [1, 3, 5],
    ) -> Dict[str, float]:
        """
        Create lag features for xG, xA, and clean sheet history.

        Args:
            player_stats: DataFrame with player gameweek stats
            fpl_id: Player FPL ID
            gameweek: Current gameweek
            season: Season identifier
            lag_periods: List of lag periods to create (e.g., [1, 3, 5] for 1, 3, 5 gameweeks ago)

        Returns:
            Dictionary with lag features
        """
        # Get player's historical stats up to current gameweek (sorted by gameweek descending)
        player_data = player_stats[
            (player_stats["fpl_id"] == fpl_id)
            & (player_stats["gameweek"] < gameweek)
            & (player_stats["season"] == season)
        ].copy()

        player_data = player_data.sort_values("gameweek", ascending=False)

        features = {}

        # Lag features for xG
        for lag in lag_periods:
            if len(player_data) >= lag:
                xg_lag = (
                    player_data.iloc[lag - 1]["xg"] if lag <= len(player_data) else 0.0
                )
                features[f"xg_lag_{lag}"] = (
                    float(xg_lag) if not pd.isna(xg_lag) else 0.0
                )
            else:
                features[f"xg_lag_{lag}"] = 0.0

        # Rolling averages for xG (3, 5 gameweeks)
        for window in [3, 5]:
            if len(player_data) >= window:
                xg_rolling = player_data.head(window)["xg"].mean()
                features[f"xg_rolling_{window}"] = (
                    float(xg_rolling) if not pd.isna(xg_rolling) else 0.0
                )
            else:
                features[f"xg_rolling_{window}"] = 0.0

        # Lag features for xA
        for lag in lag_periods:
            if len(player_data) >= lag:
                xa_lag = (
                    player_data.iloc[lag - 1]["xa"] if lag <= len(player_data) else 0.0
                )
                features[f"xa_lag_{lag}"] = (
                    float(xa_lag) if not pd.isna(xa_lag) else 0.0
                )
            else:
                features[f"xa_lag_{lag}"] = 0.0

        # Rolling averages for xA (3, 5 gameweeks)
        for window in [3, 5]:
            if len(player_data) >= window:
                xa_rolling = player_data.head(window)["xa"].mean()
                features[f"xa_rolling_{window}"] = (
                    float(xa_rolling) if not pd.isna(xa_rolling) else 0.0
                )
            else:
                features[f"xa_rolling_{window}"] = 0.0

        # Lag features for clean sheets
        for lag in lag_periods:
            if len(player_data) >= lag:
                cs_lag = (
                    player_data.iloc[lag - 1]["clean_sheets"]
                    if lag <= len(player_data)
                    else 0.0
                )
                features[f"cs_lag_{lag}"] = (
                    float(cs_lag) if not pd.isna(cs_lag) else 0.0
                )
            else:
                features[f"cs_lag_{lag}"] = 0.0

        # Rolling averages for clean sheets (3, 5 gameweeks)
        for window in [3, 5]:
            if len(player_data) >= window:
                cs_rolling = player_data.head(window)["clean_sheets"].mean()
                features[f"cs_rolling_{window}"] = (
                    float(cs_rolling) if not pd.isna(cs_rolling) else 0.0
                )
            else:
                features[f"cs_rolling_{window}"] = 0.0

        # Clean sheet rate (percentage of games with clean sheet)
        if len(player_data) > 0:
            games_with_minutes = player_data[player_data["minutes"] > 0]
            if len(games_with_minutes) > 0:
                cs_rate = (games_with_minutes["clean_sheets"] > 0).sum() / len(
                    games_with_minutes
                )
                features["cs_rate"] = float(cs_rate)
            else:
                features["cs_rate"] = 0.0
        else:
            features["cs_rate"] = 0.0

        return features

    def prepare_xmins_features(
        self, training_data: pd.DataFrame, season: str = "2025-26"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for xMins model (XGBoost Classifier).
        Includes DefCon features and entity mapping alignment.

        Args:
            training_data: Training DataFrame with player gameweek stats
            season: Season identifier

        Returns:
            (features_df, labels) tuple where labels are binary (1=started, 0=didn't start)
        """
        logger.info(f"Preparing xMins features for {len(training_data)} rows")

        # Load entity mappings
        entity_mappings = self.load_entity_mappings()

        # Sort by player and gameweek
        training_data_sorted = training_data.sort_values(["fpl_id", "gameweek"])

        features_list = []
        labels = []

        for idx, row in training_data_sorted.iterrows():
            fpl_id = row["fpl_id"]
            gameweek = row["gameweek"]

            # Verify entity mapping exists (if required)
            if fpl_id not in entity_mappings:
                logger.debug(
                    f"No entity mapping for fpl_id={fpl_id}, continuing anyway"
                )

            # Get DefCon features
            defcon_features = self.extract_defcon_features(
                training_data_sorted, fpl_id, gameweek, season
            )

            # Get lag features
            lag_features = self.create_lag_features(
                training_data_sorted, fpl_id, gameweek, season
            )

            # Combine all features
            feature_row = {
                "fpl_id": fpl_id,
                "gameweek": gameweek,
                "position": row.get("position", "MID"),
                "price": row.get("price", 5.0),
                **defcon_features,
                **lag_features,
            }

            features_list.append(feature_row)

            # Label: 1 if started (minutes > 0), 0 otherwise
            labels.append(1 if row.get("minutes", 0) > 0 else 0)

        features_df = pd.DataFrame(features_list)

        # Verify no null values
        null_counts = features_df.isnull().sum()
        if null_counts.any():
            logger.warning(
                f"Found null values in xMins features: {null_counts[null_counts > 0].to_dict()}"
            )
            features_df = features_df.fillna(0.0)

        # Optimize DataFrame types for memory efficiency (Task 3.1)
        from app.utils.dataframe_optimizer import optimize_dataframe_types

        features_df = optimize_dataframe_types(
            features_df,
            int_columns=["fpl_id", "gameweek"],
            category_columns=["position"],
        )

        logger.info(
            f"Prepared {len(features_df)} xMins feature rows with {len(features_df.columns)} features"
        )

        return features_df, np.array(labels)

    def prepare_attack_features(
        self, training_data: pd.DataFrame, season: str = "2025-26"
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Prepare features for Attack model (LightGBM for xG/xA).
        Includes DefCon features, lag features, and entity mapping alignment.

        Args:
            training_data: Training DataFrame with player gameweek stats
            season: Season identifier

        Returns:
            (features_df, xg_labels, xa_labels) tuple
        """
        logger.info(f"Preparing Attack features for {len(training_data)} rows")

        # Load entity mappings
        entity_mappings = self.load_entity_mappings()

        # Sort by player and gameweek
        training_data_sorted = training_data.sort_values(["fpl_id", "gameweek"])

        features_list = []
        xg_labels = []
        xa_labels = []

        for idx, row in training_data_sorted.iterrows():
            fpl_id = row["fpl_id"]
            gameweek = row["gameweek"]

            # Verify entity mapping exists (if required)
            if fpl_id not in entity_mappings:
                logger.debug(
                    f"No entity mapping for fpl_id={fpl_id}, continuing anyway"
                )

            # Get DefCon features
            defcon_features = self.extract_defcon_features(
                training_data_sorted, fpl_id, gameweek, season
            )

            # Get lag features for xG and xA
            lag_features = self.create_lag_features(
                training_data_sorted, fpl_id, gameweek, season
            )

            # Additional attack-specific features
            player_historical = training_data_sorted[
                (training_data_sorted["fpl_id"] == fpl_id)
                & (training_data_sorted["gameweek"] < gameweek)
                & (training_data_sorted["season"] == season)
            ]

            # Calculate per-90 stats
            if len(player_historical) > 0:
                total_minutes = player_historical["minutes"].sum()
                if total_minutes > 0:
                    xg_per_90 = (player_historical["xg"].sum() / total_minutes) * 90.0
                    xa_per_90 = (player_historical["xa"].sum() / total_minutes) * 90.0
                    goals_per_90 = (
                        player_historical["goals"].sum() / total_minutes
                    ) * 90.0
                    assists_per_90 = (
                        player_historical["assists"].sum() / total_minutes
                    ) * 90.0
                else:
                    xg_per_90 = 0.0
                    xa_per_90 = 0.0
                    goals_per_90 = 0.0
                    assists_per_90 = 0.0
            else:
                xg_per_90 = 0.0
                xa_per_90 = 0.0
                goals_per_90 = 0.0
                assists_per_90 = 0.0

            # Combine all features
            feature_row = {
                "fpl_id": fpl_id,
                "gameweek": gameweek,
                "position": row.get("position", "MID"),
                "was_home": row.get("was_home", True),
                "opponent_team": row.get("opponent_team", 0),
                "xg_per_90": xg_per_90,
                "xa_per_90": xa_per_90,
                "goals_per_90": goals_per_90,
                "assists_per_90": assists_per_90,
                **defcon_features,
                **lag_features,
            }

            features_list.append(feature_row)

            # Labels: actual xG and xA for this gameweek
            xg_labels.append(
                float(row.get("xg", 0.0)) if not pd.isna(row.get("xg")) else 0.0
            )
            xa_labels.append(
                float(row.get("xa", 0.0)) if not pd.isna(row.get("xa")) else 0.0
            )

        features_df = pd.DataFrame(features_list)

        # Verify no null values
        null_counts = features_df.isnull().sum()
        if null_counts.any():
            logger.warning(
                f"Found null values in Attack features: {null_counts[null_counts > 0].to_dict()}"
            )
            features_df = features_df.fillna(0.0)

        # Optimize DataFrame types for memory efficiency (Task 3.1)
        from app.utils.dataframe_optimizer import optimize_dataframe_types

        features_df = optimize_dataframe_types(
            features_df,
            int_columns=["fpl_id", "gameweek", "opponent_team"],
            category_columns=["position"],
        )

        logger.info(
            f"Prepared {len(features_df)} Attack feature rows with {len(features_df.columns)} features"
        )

        return features_df, np.array(xg_labels), np.array(xa_labels)

    def prepare_defense_features(
        self, training_data: pd.DataFrame, season: str = "2025-26"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for Defense model (LightGBM for P_CS).
        Includes DefCon features, lag features, and entity mapping alignment.

        Args:
            training_data: Training DataFrame with player gameweek stats
            season: Season identifier

        Returns:
            (features_df, cs_labels) tuple where cs_labels are binary (1=clean sheet, 0=no clean sheet)
        """
        logger.info(f"Preparing Defense features for {len(training_data)} rows")

        # Load entity mappings
        entity_mappings = self.load_entity_mappings()

        # Sort by player and gameweek
        training_data_sorted = training_data.sort_values(["fpl_id", "gameweek"])

        features_list = []
        cs_labels = []

        for idx, row in training_data_sorted.iterrows():
            fpl_id = row["fpl_id"]
            gameweek = row["gameweek"]

            # Verify entity mapping exists (if required)
            if fpl_id not in entity_mappings:
                logger.debug(
                    f"No entity mapping for fpl_id={fpl_id}, continuing anyway"
                )

            # Get DefCon features
            defcon_features = self.extract_defcon_features(
                training_data_sorted, fpl_id, gameweek, season
            )

            # Get lag features for clean sheets
            lag_features = self.create_lag_features(
                training_data_sorted, fpl_id, gameweek, season
            )

            # Additional defense-specific features
            player_historical = training_data_sorted[
                (training_data_sorted["fpl_id"] == fpl_id)
                & (training_data_sorted["gameweek"] < gameweek)
                & (training_data_sorted["season"] == season)
            ]

            # Calculate defensive stats
            if len(player_historical) > 0:
                total_minutes = player_historical["minutes"].sum()
                if total_minutes > 0:
                    xgc_per_90 = (player_historical["xgc"].sum() / total_minutes) * 90.0
                    goals_conceded_per_90 = (
                        player_historical["goals_conceded"].sum() / total_minutes
                    ) * 90.0
                else:
                    xgc_per_90 = 0.0
                    goals_conceded_per_90 = 0.0

                # Clean sheet rate
                games_with_minutes = player_historical[player_historical["minutes"] > 0]
                if len(games_with_minutes) > 0:
                    cs_rate = (games_with_minutes["clean_sheets"] > 0).sum() / len(
                        games_with_minutes
                    )
                else:
                    cs_rate = 0.0
            else:
                xgc_per_90 = 0.0
                goals_conceded_per_90 = 0.0
                cs_rate = 0.0

            # Combine all features
            feature_row = {
                "fpl_id": fpl_id,
                "gameweek": gameweek,
                "position": row.get("position", "MID"),
                "was_home": row.get("was_home", True),
                "opponent_team": row.get("opponent_team", 0),
                "xgc_per_90": xgc_per_90,
                "goals_conceded_per_90": goals_conceded_per_90,
                "cs_rate": cs_rate,
                **defcon_features,
                **lag_features,
            }

            features_list.append(feature_row)

            # Label: 1 if clean sheet (clean_sheets > 0), 0 otherwise
            cs_labels.append(1 if row.get("clean_sheets", 0) > 0 else 0)

        features_df = pd.DataFrame(features_list)

        # Verify no null values
        null_counts = features_df.isnull().sum()
        if null_counts.any():
            logger.warning(
                f"Found null values in Defense features: {null_counts[null_counts > 0].to_dict()}"
            )
            features_df = features_df.fillna(0.0)

        # Optimize DataFrame types for memory efficiency (Task 3.1)
        from app.utils.dataframe_optimizer import optimize_dataframe_types

        features_df = optimize_dataframe_types(
            features_df,
            int_columns=["fpl_id", "gameweek", "opponent_team"],
            category_columns=["position"],
        )

        logger.info(
            f"Prepared {len(features_df)} Defense feature rows with {len(features_df.columns)} features"
        )

        return features_df, np.array(cs_labels)

    def verify_feature_distributions(
        self, features_df: pd.DataFrame, feature_type: str = "general"
    ) -> Dict[str, Dict]:
        """
        Verify feature distributions and ensure no null values.

        Args:
            features_df: DataFrame with features
            feature_type: Type of features (for logging)

        Returns:
            Dictionary with distribution statistics
        """
        stats = {}

        for col in features_df.columns:
            if col in ["fpl_id", "gameweek", "position"]:
                continue

            col_data = features_df[col]

            # Check for null values
            null_count = col_data.isnull().sum()
            if null_count > 0:
                logger.warning(
                    f"{feature_type} feature '{col}' has {null_count} null values"
                )

            # Calculate statistics
            stats[col] = {
                "mean": float(col_data.mean()) if not col_data.empty else 0.0,
                "std": float(col_data.std()) if not col_data.empty else 0.0,
                "min": float(col_data.min()) if not col_data.empty else 0.0,
                "max": float(col_data.max()) if not col_data.empty else 0.0,
                "null_count": int(null_count),
            }

        logger.info(
            f"Feature distribution verification for {feature_type}: {len(stats)} features checked"
        )

        return stats
