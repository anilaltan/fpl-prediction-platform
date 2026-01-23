"""
FPL Data Processors
Transform and normalize FPL API responses.
"""

from typing import Dict, List, Optional
import logging

from app.services.data_cleaning import DataCleaningService

logger = logging.getLogger(__name__)


class FPLDataProcessor:
    """
    Transform and normalize FPL API responses.

    Handles data extraction, transformation, and normalization
    from raw FPL API responses to structured formats.
    """

    def __init__(self, data_cleaning: Optional[DataCleaningService] = None) -> None:
        """
        Initialize data processor.

        Args:
            data_cleaning: Optional DataCleaningService instance
        """
        self.data_cleaning = data_cleaning or DataCleaningService()

    def extract_players_from_bootstrap(self, bootstrap_data: Dict) -> List[Dict]:
        """
        Extract and structure player data from bootstrap-static.

        Args:
            bootstrap_data: Raw bootstrap data from FPL API

        Returns:
            List of structured player dictionaries with all relevant fields
        """
        elements = bootstrap_data.get("elements", [])
        teams = bootstrap_data.get("teams", [])
        element_types = bootstrap_data.get("element_types", [])

        # Create lookup dictionaries
        team_lookup = {team["id"]: team for team in teams}
        position_lookup = {et["id"]: et["singular_name_short"] for et in element_types}

        players = []
        for element in elements:
            team_id = element.get("team")
            team_data = team_lookup.get(team_id, {})
            position_id = element.get("element_type")
            position = position_lookup.get(position_id, "UNKNOWN")

            player = {
                "id": element.get("id"),
                "fpl_id": element.get("id"),
                "web_name": element.get("web_name", ""),
                "first_name": element.get("first_name", ""),
                "second_name": element.get("second_name", ""),
                "full_name": f"{element.get('first_name', '')} {element.get('second_name', '')}".strip(),
                "position": position,
                "position_id": position_id,
                "team_id": team_id,
                "team_name": team_data.get("name", ""),
                "team_short_name": team_data.get("short_name", ""),
                "price": element.get("now_cost", 0) / 10.0,
                "price_start": element.get("cost_change_start", 0) / 10.0,
                "selected_by_percent": float(element.get("selected_by_percent", 0.0)),
                "form": float(element.get("form", 0.0)),
                "points_per_game": float(element.get("points_per_game", 0.0)),
                "total_points": element.get("total_points", 0),
                "goals_scored": element.get("goals_scored", 0),
                "assists": element.get("assists", 0),
                "clean_sheets": element.get("clean_sheets", 0),
                "saves": element.get("saves", 0),
                "bonus": element.get("bonus", 0),
                "bps": element.get("bps", 0),
                "influence": float(element.get("influence", 0.0)),
                "creativity": float(element.get("creativity", 0.0)),
                "threat": float(element.get("threat", 0.0)),
                "ict_index": float(element.get("ict_index", 0.0)),
                "status": element.get("status", "a"),
                "news": element.get("news", ""),
                "news_added": element.get("news_added"),
                "transfers_in": element.get("transfers_in", 0),
                "transfers_out": element.get("transfers_out", 0),
                "transfers_in_event": element.get("transfers_in_event", 0),
                "transfers_out_event": element.get("transfers_out_event", 0),
                "value_form": float(element.get("value_form", 0.0)),
                "value_season": float(element.get("value_season", 0.0)),
                "minutes": element.get("minutes", 0),
                "goals_conceded": element.get("goals_conceded", 0),
                "yellow_cards": element.get("yellow_cards", 0),
                "red_cards": element.get("red_cards", 0),
                "penalties_saved": element.get("penalties_saved", 0),
                "penalties_missed": element.get("penalties_missed", 0),
                "expected_goals": float(element.get("expected_goals", 0.0)),
                "expected_assists": float(element.get("expected_assists", 0.0)),
                "expected_goal_involvements": float(
                    element.get("expected_goal_involvements", 0.0)
                ),
                "expected_goals_conceded": float(
                    element.get("expected_goals_conceded", 0.0)
                ),
            }
            players.append(player)

        return players

    def extract_teams_from_bootstrap(self, bootstrap_data: Dict) -> List[Dict]:
        """
        Extract team data from bootstrap-static.

        Args:
            bootstrap_data: Raw bootstrap data from FPL API

        Returns:
            List of structured team dictionaries
        """
        teams = bootstrap_data.get("teams", [])
        return [
            {
                "id": team.get("id"),
                "name": team.get("name", ""),
                "short_name": team.get("short_name", ""),
                "code": team.get("code", 0),
                "strength": team.get("strength", 0),
                "strength_attack_home": team.get("strength_attack_home", 0),
                "strength_attack_away": team.get("strength_attack_away", 0),
                "strength_defence_home": team.get("strength_defence_home", 0),
                "strength_defence_away": team.get("strength_defence_away", 0),
                "pulse_id": team.get("pulse_id", 0),
            }
            for team in teams
        ]

    def extract_player_history(self, player_summary: Dict) -> List[Dict]:
        """
        Extract match history statistics from element-summary.

        Normalizes DGW (Double Gameweek) points automatically.

        Args:
            player_summary: Player summary data from FPL API

        Returns:
            List of match statistics with normalized points for DGW
        """
        history = player_summary.get("history", [])

        structured_history = []
        for match in history:
            # Detect DGW (Double Gameweek) - if minutes > 90, likely 2 matches
            minutes = match.get("minutes", 0)
            matches_played = 2 if minutes > 90 else 1
            points = match.get("total_points", 0)

            # Normalize DGW points
            normalized_points = self.data_cleaning.normalize_dgw_points(
                points, matches_played, "dgw" if matches_played > 1 else "normal"
            )

            match_data = {
                "gameweek": match.get("round", 0),
                "fixture": match.get("fixture", 0),
                "opponent_team": match.get("opponent_team", 0),
                "was_home": match.get("was_home", False),
                "minutes": minutes,
                "points": points,
                "normalized_points": normalized_points,  # DGW normalized
                "matches_played": matches_played,
                "goals_scored": match.get("goals_scored", 0),
                "assists": match.get("assists", 0),
                "clean_sheets": match.get("clean_sheets", 0),
                "goals_conceded": match.get("goals_conceded", 0),
                "saves": match.get("saves", 0),
                "bonus": match.get("bonus", 0),
                "bps": match.get("bps", 0),
                "influence": float(match.get("influence", 0.0)),
                "creativity": float(match.get("creativity", 0.0)),
                "threat": float(match.get("threat", 0.0)),
                "ict_index": float(match.get("ict_index", 0.0)),
                "expected_goals": float(match.get("expected_goals", 0.0)),
                "expected_assists": float(match.get("expected_assists", 0.0)),
                "expected_goal_involvements": float(
                    match.get("expected_goal_involvements", 0.0)
                ),
                "expected_goals_conceded": float(
                    match.get("expected_goals_conceded", 0.0)
                ),
                "value": match.get("value", 0) / 10.0,
                "transfers_balance": match.get("transfers_balance", 0),
                "selected": match.get("selected", 0),
                "transfers_in": match.get("transfers_in", 0),
                "transfers_out": match.get("transfers_out", 0),
            }
            structured_history.append(match_data)

        return structured_history

    def extract_gameweek_from_events(
        self, bootstrap_data: Dict, is_next: bool = False
    ) -> Optional[int]:
        """
        Extract current or next gameweek from bootstrap events.

        Args:
            bootstrap_data: Raw bootstrap data
            is_next: If True, get next gameweek; if False, get current

        Returns:
            Gameweek number or None if not found
        """
        events = bootstrap_data.get("events", [])

        if is_next:
            # Find next gameweek (is_next = True)
            next_event = next((e for e in events if e.get("is_next")), None)
            if next_event:
                return next_event.get("id")

            # Fallback: Find first unfinished gameweek
            unfinished_events = [e for e in events if not e.get("finished", True)]
            if unfinished_events:
                unfinished_events.sort(key=lambda x: x.get("id", 999))
                return unfinished_events[0].get("id")
        else:
            # Find current gameweek (is_current = True)
            current_event = next((e for e in events if e.get("is_current")), None)
            if current_event:
                return current_event.get("id")

            # Fallback: Find next gameweek if current not found
            next_event = next((e for e in events if e.get("is_next")), None)
            if next_event:
                return next_event.get("id")

            # Last resort: Use the latest finished gameweek
            finished_events = [e for e in events if e.get("finished")]
            if finished_events:
                latest_finished = max(finished_events, key=lambda x: x.get("id", 0))
                return latest_finished.get("id")

        return None
