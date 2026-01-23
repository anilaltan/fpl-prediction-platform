"""
Risk Management and Strategy Layer for FPL
Implements:
- Ownership Arbitrage: Identify overvalued players (low xP/Ownership ratio)
- C/VC Logic: Advanced captain selection with Expected Value formula
- Chip Trigger: Wildcard usage recommendations based on 5-week point gain
- Solver Integration: Add strategy notes to solver outputs
"""
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class StrategyService:
    """
    Main strategy service orchestrating risk management and decision support.
    Integrates with PLEngine and FPLSolver to provide strategic recommendations.
    """

    def __init__(self):
        """Initialize strategy service."""
        self.ownership_threshold = 0.15  # 15% ownership threshold
        self.xp_ownership_ratio_threshold = (
            0.3  # xP/Ownership ratio threshold for overvalued
        )
        self.wildcard_gain_threshold = 20.0  # +20 points threshold for wildcard

    def analyze_ownership_arbitrage(
        self, players: List[Dict], plengine, gameweek: int = 1
    ) -> Dict:
        """
        Analyze ownership arbitrage opportunities.
        Compares FPL API selected_by_percent with PLEngine xP values.
        Identifies 'Overvalued' players with low xP/Ownership ratio.

        Args:
            players: List of player dictionaries with FPL API data (selected_by_percent)
            plengine: PLEngine instance for xP predictions
            gameweek: Current gameweek

        Returns:
            Dictionary with overvalued players and arbitrage opportunities
        """
        overvalued_players = []

        for player in players:
            # Get ownership from FPL API (selected_by_percent)
            ownership_percent = player.get("selected_by_percent", 0.0)
            ownership_ratio = (
                ownership_percent / 100.0 if ownership_percent > 0 else 0.0
            )

            # Skip if ownership too low
            if ownership_ratio < self.ownership_threshold:
                continue

            # Get xP from PLEngine
            try:
                prediction = plengine.predict(
                    player_data=player, fixture_data=player.get("fixture_data")
                )
                xp = prediction.get("expected_points", 0.0)
            except Exception as e:
                logger.warning(
                    f"Error predicting xP for player {player.get('id')}: {str(e)}"
                )
                xp = 0.0

            # Calculate xP/Ownership ratio
            if ownership_ratio > 0:
                xp_ownership_ratio = xp / ownership_ratio
            else:
                xp_ownership_ratio = 0.0

            # Flag as overvalued if ratio is low
            if xp_ownership_ratio < self.xp_ownership_ratio_threshold:
                value_score = (
                    xp / player.get("price", 50.0)
                    if player.get("price", 0) > 0
                    else 0.0
                )

                overvalued_players.append(
                    {
                        "player_id": player.get("id"),
                        "name": player.get("web_name", player.get("name", "")),
                        "position": self._get_position_name(
                            player.get("element_type", 3)
                        ),
                        "team_id": player.get("team", 0),
                        "ownership_percent": ownership_percent,
                        "expected_points": xp,
                        "xp_ownership_ratio": xp_ownership_ratio,
                        "price": player.get("now_cost", 0)
                        / 10.0,  # Convert from FPL units
                        "value_score": value_score,
                        "overvalued": True,
                    }
                )

        # Sort by ownership (highest first) then by xP/Ownership ratio (lowest first)
        overvalued_players.sort(
            key=lambda x: (-x["ownership_percent"], x["xp_ownership_ratio"])
        )

        return {
            "gameweek": gameweek,
            "overvalued_count": len(overvalued_players),
            "overvalued_players": overvalued_players[:20],  # Top 20
            "threshold_ownership": self.ownership_threshold * 100,
            "threshold_ratio": self.xp_ownership_ratio_threshold,
        }

    def calculate_captain_vice_captain_value(
        self, captain: Dict, vice_captain: Dict, plengine, gameweek: int = 1
    ) -> Dict:
        """
        Calculate Expected Value for C/VC pair using formula:
        Expected_Value = (xP_Capt * P_start_Capt) + (xP_VC * (1 - P_start_Capt))

        This formula accounts for the probability that captain doesn't start,
        in which case vice-captain's points are used.

        Args:
            captain: Captain player data
            vice_captain: Vice-captain player data
            plengine: PLEngine instance for predictions
            gameweek: Current gameweek

        Returns:
            Dictionary with Expected Value and analysis
        """
        # Get captain predictions
        try:
            capt_prediction = plengine.predict(
                player_data=captain, fixture_data=captain.get("fixture_data")
            )
            xp_capt = capt_prediction.get("expected_points", 0.0)
            p_start_capt = capt_prediction.get("p_start", 0.7)
        except Exception as e:
            logger.warning(
                f"Error predicting for captain {captain.get('id')}: {str(e)}"
            )
            xp_capt = 0.0
            p_start_capt = 0.7

        # Get vice-captain predictions
        try:
            vc_prediction = plengine.predict(
                player_data=vice_captain, fixture_data=vice_captain.get("fixture_data")
            )
            xp_vc = vc_prediction.get("expected_points", 0.0)
            p_start_vc = vc_prediction.get("p_start", 0.7)
        except Exception as e:
            logger.warning(
                f"Error predicting for vice-captain {vice_captain.get('id')}: {str(e)}"
            )
            xp_vc = 0.0
            p_start_vc = 0.7

        # Calculate Expected Value
        # Expected_Value = (xP_Capt * P_start_Capt) + (xP_VC * (1 - P_start_Capt))
        expected_value = (xp_capt * p_start_capt) + (xp_vc * (1 - p_start_capt))

        # Calculate individual components
        capt_contribution = xp_capt * p_start_capt
        vc_contribution = xp_vc * (1 - p_start_capt)

        # Risk metrics
        capt_no_play_prob = 1 - p_start_capt
        vc_no_play_prob = 1 - p_start_vc

        return {
            "captain": {
                "id": captain.get("id"),
                "name": captain.get("web_name", captain.get("name", "")),
                "expected_points": xp_capt,
                "p_start": p_start_capt,
                "no_play_probability": capt_no_play_prob,
                "contribution": capt_contribution,
            },
            "vice_captain": {
                "id": vice_captain.get("id"),
                "name": vice_captain.get("web_name", vice_captain.get("name", "")),
                "expected_points": xp_vc,
                "p_start": p_start_vc,
                "no_play_probability": vc_no_play_prob,
                "contribution": vc_contribution,
            },
            "expected_value": expected_value,
            "captain_contribution": capt_contribution,
            "vice_captain_contribution": vc_contribution,
            "formula": "Expected_Value = (xP_Capt * P_start_Capt) + (xP_VC * (1 - P_start_Capt))",
        }

    def analyze_chip_triggers(
        self,
        current_squad: List[Dict],
        optimized_squad: Optional[List[Dict]] = None,
        plengine=None,
        horizon_weeks: int = 5,
    ) -> Dict:
        """
        Analyze chip usage triggers.
        Calculates 5-week point gain for Wildcard usage.
        Returns use_wildcard=True if gain >= +20 points.

        Args:
            current_squad: Current squad player data
            optimized_squad: Optimized squad from solver (optional)
            plengine: PLEngine instance for predictions (optional)
            horizon_weeks: Number of weeks to analyze (default: 5)

        Returns:
            Dictionary with chip recommendations
        """
        recommendations = {
            "use_wildcard": False,
            "use_bench_boost": False,
            "use_free_hit": False,
            "wildcard_gain": 0.0,
            "wildcard_threshold": self.wildcard_gain_threshold,
            "analysis": {},
        }

        if not optimized_squad:
            return recommendations

        # Calculate current squad expected points over 5 weeks
        current_squad_points = self._calculate_squad_points(
            current_squad, plengine, horizon_weeks
        )

        # Calculate optimized squad expected points over 5 weeks
        optimized_squad_points = self._calculate_squad_points(
            optimized_squad, plengine, horizon_weeks
        )

        # Calculate gain
        wildcard_gain = optimized_squad_points - current_squad_points

        # Check threshold
        if wildcard_gain >= self.wildcard_gain_threshold:
            recommendations["use_wildcard"] = True
            recommendations["wildcard_gain"] = wildcard_gain

        recommendations["analysis"] = {
            "current_squad_points": current_squad_points,
            "optimized_squad_points": optimized_squad_points,
            "wildcard_gain": wildcard_gain,
            "threshold": self.wildcard_gain_threshold,
            "recommendation": "Use Wildcard"
            if recommendations["use_wildcard"]
            else "Keep Current Squad",
        }

        return recommendations

    def _calculate_squad_points(
        self, squad: List[Dict], plengine, horizon_weeks: int = 5
    ) -> float:
        """
        Calculate total expected points for a squad over horizon weeks.

        Args:
            squad: Squad player data
            plengine: PLEngine instance
            horizon_weeks: Number of weeks

        Returns:
            Total expected points
        """
        if not plengine:
            # Fallback: use simple average if no PLEngine
            return sum(p.get("expected_points", 0.0) for p in squad) * horizon_weeks

        total_points = 0.0

        for player in squad:
            player_points = 0.0

            for week in range(1, horizon_weeks + 1):
                try:
                    prediction = plengine.predict(
                        player_data=player, fixture_data=player.get("fixture_data")
                    )
                    week_points = prediction.get("expected_points", 0.0)
                    player_points += week_points
                except Exception as e:
                    logger.warning(
                        f"Error predicting for player {player.get('id')} week {week}: {str(e)}"
                    )
                    # Use fallback
                    week_points = player.get("expected_points", 0.0)
                    player_points += week_points

            total_points += player_points

        return total_points

    def generate_strategy_notes(
        self,
        solver_solution: Dict,
        players: List[Dict],
        plengine,
        current_squad: Optional[List[Dict]] = None,
        gameweek: int = 1,
    ) -> Dict:
        """
        Generate strategy notes to add to solver solution.
        Integrates ownership arbitrage, C/VC analysis, and chip triggers.

        Args:
            solver_solution: Solution from FPLSolver
            players: All available players (for ownership analysis)
            plengine: PLEngine instance
            current_squad: Current squad (for chip analysis)
            gameweek: Current gameweek

        Returns:
            Dictionary with strategy notes to add to solution
        """
        strategy_notes = {
            "ownership_arbitrage": None,
            "captain_analysis": None,
            "chip_recommendations": None,
            "warnings": [],
            "recommendations": [],
        }

        # 1. Ownership Arbitrage Analysis
        try:
            ownership_analysis = self.analyze_ownership_arbitrage(
                players, plengine, gameweek
            )
            strategy_notes["ownership_arbitrage"] = ownership_analysis

            # Add warnings for overvalued players in optimized squad
            optimized_player_ids = {
                p.get("id") for p in solver_solution.get("squad_week1", [])
            }
            overvalued_ids = {
                p["player_id"] for p in ownership_analysis.get("overvalued_players", [])
            }

            overvalued_in_squad = optimized_player_ids.intersection(overvalued_ids)
            if overvalued_in_squad:
                strategy_notes["warnings"].append(
                    f"Warning: {len(overvalued_in_squad)} overvalued players in optimized squad. "
                    "Consider differential alternatives."
                )
        except Exception as e:
            logger.error(f"Error in ownership arbitrage analysis: {str(e)}")
            strategy_notes["warnings"].append(f"Ownership analysis failed: {str(e)}")

        # 2. C/VC Analysis
        try:
            captain = solver_solution.get("captain")
            vice_captain = solver_solution.get("vice_captain")

            if captain and vice_captain:
                # Get full player data
                capt_data = next(
                    (p for p in players if p.get("id") == captain.get("id")), captain
                )
                vc_data = next(
                    (p for p in players if p.get("id") == vice_captain.get("id")),
                    vice_captain,
                )

                cvc_analysis = self.calculate_captain_vice_captain_value(
                    capt_data, vc_data, plengine, gameweek
                )
                strategy_notes["captain_analysis"] = cvc_analysis

                # Add recommendation if expected value is low
                if cvc_analysis.get("expected_value", 0.0) < 10.0:
                    strategy_notes["recommendations"].append(
                        "Consider alternative C/VC pair: Current expected value is below optimal."
                    )
        except Exception as e:
            logger.error(f"Error in C/VC analysis: {str(e)}")
            strategy_notes["warnings"].append(f"C/VC analysis failed: {str(e)}")

        # 3. Chip Recommendations
        try:
            if current_squad:
                optimized_squad = solver_solution.get("squad_week1", [])
                chip_analysis = self.analyze_chip_triggers(
                    current_squad, optimized_squad, plengine, horizon_weeks=5
                )
                strategy_notes["chip_recommendations"] = chip_analysis

                if chip_analysis.get("use_wildcard", False):
                    strategy_notes["recommendations"].append(
                        f"Wildcard recommended: +{chip_analysis.get('wildcard_gain', 0.0):.1f} points "
                        f"over 5 weeks (threshold: {self.wildcard_gain_threshold})"
                    )
        except Exception as e:
            logger.error(f"Error in chip analysis: {str(e)}")
            strategy_notes["warnings"].append(f"Chip analysis failed: {str(e)}")

        # 4. General recommendations based on solution
        total_xp = solver_solution.get("total_expected_points", 0.0)
        transfer_cost = solver_solution.get("total_transfer_cost", 0)
        net_points = total_xp - transfer_cost

        if transfer_cost > 0:
            strategy_notes["recommendations"].append(
                f"Transfer cost: -{transfer_cost} points. "
                f"Net expected points: {net_points:.1f}"
            )

        if net_points < 50.0:
            strategy_notes["warnings"].append(
                "Low net expected points. Consider more aggressive transfers or chip usage."
            )

        return strategy_notes

    def add_strategy_notes_to_solution(
        self, solver_solution: Dict, strategy_notes: Dict
    ) -> Dict:
        """
        Add strategy notes to solver solution.

        Args:
            solver_solution: Original solver solution
            strategy_notes: Strategy notes from generate_strategy_notes

        Returns:
            Enhanced solution with strategy notes
        """
        enhanced_solution = solver_solution.copy()
        enhanced_solution["strategy_notes"] = strategy_notes
        enhanced_solution["has_strategy_notes"] = True

        return enhanced_solution

    def _get_position_name(self, element_type: int) -> str:
        """
        Convert FPL element_type to position name.

        Args:
            element_type: FPL element type (1=GK, 2=DEF, 3=MID, 4=FWD)

        Returns:
            Position name string
        """
        position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        return position_map.get(element_type, "MID")
