"""
Risk Management and Smart Money Strategies for FPL
Implements:
- Ownership Arbitrage: Find differential opportunities
- C/VC Paradox: Advanced captain selection
- Chip Timing: Optimal chip usage triggers
"""
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class OwnershipArbitrage:
    """
    Identifies ownership arbitrage opportunities.
    Finds players with high ownership but low expected points (xP),
    suggesting differential alternatives with better value.
    """

    def __init__(self, ownership_threshold: float = 0.20, xp_threshold: float = 5.0):
        """
        Initialize arbitrage detector.

        Args:
            ownership_threshold: Minimum ownership % to consider (default: 20%)
            xp_threshold: Maximum xP to flag as over-owned (default: 5.0)
        """
        self.ownership_threshold = ownership_threshold
        self.xp_threshold = xp_threshold

    def find_over_owned_players(
        self, players: List[Dict], gameweek: int = 1
    ) -> List[Dict]:
        """
        Find players with high ownership but low expected points.

        Args:
            players: List of player dictionaries with ownership and xP
            gameweek: Current gameweek

        Returns:
            List of over-owned players with arbitrage opportunities
        """
        over_owned = []

        for player in players:
            ownership = player.get("ownership_percent", 0.0) / 100.0
            xp_key = f"expected_points_gw{gameweek}"
            xp = player.get(xp_key, player.get("expected_points", 0.0))

            # Flag if high ownership but low xP
            if ownership >= self.ownership_threshold and xp <= self.xp_threshold:
                # Calculate value score
                price = player.get("price", 50.0)
                value_score = xp / price if price > 0 else 0.0

                over_owned.append(
                    {
                        "player_id": player.get("id"),
                        "name": player.get("name"),
                        "position": player.get("position"),
                        "team": player.get("team_name"),
                        "ownership_percent": ownership * 100,
                        "expected_points": xp,
                        "price": price,
                        "value_score": value_score,
                        "arbitrage_opportunity": True,
                    }
                )

        # Sort by ownership (highest first) then by xP (lowest first)
        over_owned.sort(key=lambda x: (-x["ownership_percent"], x["expected_points"]))

        return over_owned

    def find_differential_alternatives(
        self,
        over_owned_player: Dict,
        all_players: List[Dict],
        gameweek: int = 1,
        max_price_diff: float = 2.0,
        min_xp_improvement: float = 1.0,
    ) -> List[Dict]:
        """
        Find differential alternatives to an over-owned player.

        Args:
            over_owned_player: The over-owned player to replace
            all_players: All available players
            gameweek: Current gameweek
            max_price_diff: Maximum price difference (default: 2.0M)
            min_xp_improvement: Minimum xP improvement required (default: 1.0)

        Returns:
            List of differential alternatives sorted by value
        """
        alternatives = []

        position = over_owned_player.get("position")
        price = over_owned_player.get("price", 50.0)
        current_xp = over_owned_player.get("expected_points", 0.0)

        xp_key = f"expected_points_gw{gameweek}"

        for player in all_players:
            # Skip same player
            if player.get("id") == over_owned_player.get("player_id"):
                continue

            # Must be same position
            if player.get("position") != position:
                continue

            # Check price constraint
            player_price = player.get("price", 50.0)
            if abs(player_price - price) > max_price_diff:
                continue

            # Check ownership (should be lower)
            player_ownership = player.get("ownership_percent", 0.0) / 100.0
            if player_ownership >= self.ownership_threshold:
                continue  # Not a differential

            # Check xP improvement
            player_xp = player.get(xp_key, player.get("expected_points", 0.0))
            if player_xp < current_xp + min_xp_improvement:
                continue

            # Calculate metrics
            value_score = player_xp / player_price if player_price > 0 else 0.0
            xp_improvement = player_xp - current_xp
            ownership_diff = (
                over_owned_player.get("ownership_percent", 0.0) / 100.0
            ) - player_ownership

            alternatives.append(
                {
                    "player_id": player.get("id"),
                    "name": player.get("name"),
                    "position": player.get("position"),
                    "team": player.get("team_name"),
                    "ownership_percent": player_ownership * 100,
                    "expected_points": player_xp,
                    "price": player_price,
                    "value_score": value_score,
                    "xp_improvement": xp_improvement,
                    "ownership_differential": ownership_diff * 100,
                    "price_diff": player_price - price,
                }
            )

        # Sort by xP improvement, then by value score
        alternatives.sort(key=lambda x: (-x["xp_improvement"], -x["value_score"]))

        return alternatives[:5]  # Return top 5 alternatives

    def analyze_arbitrage_opportunities(
        self, players: List[Dict], gameweek: int = 1
    ) -> Dict:
        """
        Comprehensive arbitrage analysis.

        Returns:
            Dictionary with over-owned players and their alternatives
        """
        over_owned = self.find_over_owned_players(players, gameweek)

        opportunities = []
        for player in over_owned[:10]:  # Top 10 over-owned
            alternatives = self.find_differential_alternatives(
                player, players, gameweek
            )

            if alternatives:
                opportunities.append(
                    {
                        "over_owned_player": player,
                        "alternatives": alternatives,
                        "potential_gain": alternatives[0]["xp_improvement"]
                        if alternatives
                        else 0.0,
                    }
                )

        # Calculate total arbitrage potential
        total_potential_gain = sum(opp["potential_gain"] for opp in opportunities)

        return {
            "gameweek": gameweek,
            "over_owned_count": len(over_owned),
            "opportunities": opportunities,
            "total_potential_gain": total_potential_gain,
            "recommendations": self._generate_recommendations(opportunities),
        }

    def _generate_recommendations(self, opportunities: List[Dict]) -> List[str]:
        """Generate human-readable recommendations"""
        recommendations = []

        for opp in opportunities[:5]:  # Top 5 opportunities
            over_owned = opp["over_owned_player"]
            best_alt = opp["alternatives"][0] if opp["alternatives"] else None

            if best_alt:
                rec = (
                    f"Consider replacing {over_owned['name']} ({over_owned['ownership_percent']:.1f}% owned, "
                    f"{over_owned['expected_points']:.1f}xP) with {best_alt['name']} "
                    f"({best_alt['ownership_percent']:.1f}% owned, {best_alt['expected_points']:.1f}xP). "
                    f"Potential gain: +{best_alt['xp_improvement']:.1f} points"
                )
                recommendations.append(rec)

        return recommendations


class CaptainViceCaptainParadox:
    """
    Advanced captain selection algorithm.
    Considers the probability of captain not playing and weights
    vice-captain's expected value accordingly.
    """

    def __init__(self, default_no_play_prob: float = 0.05):
        """
        Initialize C/VC selector.

        Args:
            default_no_play_prob: Default probability of captain not playing (default: 5%)
        """
        self.default_no_play_prob = default_no_play_prob

    def calculate_no_play_probability(
        self, player: Dict, fixture_data: Optional[Dict] = None
    ) -> float:
        """
        Calculate probability that player won't play.

        Args:
            player: Player data
            fixture_data: Fixture information

        Returns:
            Probability of not playing (0-1)
        """
        # Base probability from injury status
        status = player.get("status", "a").lower()
        status_prob = {
            "a": 0.0,  # Available
            "d": 0.15,  # Doubtful
            "i": 0.95,  # Injured
            "n": 0.0,  # Not available (suspended)
            "s": 0.95,  # Suspended
        }
        base_prob = status_prob.get(status, self.default_no_play_prob)

        # Adjust based on expected minutes
        expected_minutes = player.get("expected_minutes", 90.0)
        if expected_minutes < 60:
            base_prob += 0.10
        elif expected_minutes < 90:
            base_prob += 0.05

        # Adjust based on rotation risk
        rotation_risk = player.get("rotation_risk", 0.0)
        base_prob += rotation_risk * 0.20

        # Recent minutes pattern (if consistently low, higher risk)
        recent_minutes = player.get("recent_minutes", [])
        if recent_minutes and len(recent_minutes) >= 3:
            avg_recent = np.mean(recent_minutes[:3])
            if avg_recent < 60:
                base_prob += 0.10

        return min(1.0, max(0.0, base_prob))

    def calculate_weighted_expected_value(
        self, captain: Dict, vice_captain: Dict, gameweek: int = 1
    ) -> Dict[str, float]:
        """
        Calculate weighted expected value considering C/VC paradox.

        Formula:
        E[Total] = P(C plays) × (2×C_xP) + P(C doesn't play) × (2×VC_xP)

        Args:
            captain: Captain player data
            vice_captain: Vice-captain player data
            gameweek: Current gameweek

        Returns:
            Dictionary with weighted expected values
        """
        # Get expected points
        c_xp_key = f"expected_points_gw{gameweek}"
        vc_xp_key = f"expected_points_gw{gameweek}"

        c_xp = captain.get(c_xp_key, captain.get("expected_points", 0.0))
        vc_xp = vice_captain.get(vc_xp_key, vice_captain.get("expected_points", 0.0))

        # Calculate no-play probabilities
        c_no_play_prob = self.calculate_no_play_probability(captain)
        vc_no_play_prob = self.calculate_no_play_probability(vice_captain)

        # Weighted expected value
        # If captain plays: 2×C_xP
        # If captain doesn't play: 2×VC_xP
        weighted_value = (1 - c_no_play_prob) * (2 * c_xp) + c_no_play_prob * (
            2 * vc_xp
        )

        # Risk-adjusted value (penalize high no-play probability)
        risk_penalty = c_no_play_prob * 2.0  # Penalty for uncertainty
        risk_adjusted_value = weighted_value - risk_penalty

        # Calculate expected value if we swap C/VC
        swapped_weighted_value = (1 - vc_no_play_prob) * (
            2 * vc_xp
        ) + vc_no_play_prob * (2 * c_xp)
        swapped_risk_penalty = vc_no_play_prob * 2.0
        swapped_risk_adjusted = swapped_weighted_value - swapped_risk_penalty

        return {
            "captain_xp": c_xp,
            "vice_captain_xp": vc_xp,
            "captain_no_play_prob": c_no_play_prob,
            "vice_captain_no_play_prob": vc_no_play_prob,
            "weighted_expected_value": weighted_value,
            "risk_adjusted_value": risk_adjusted_value,
            "swapped_weighted_value": swapped_weighted_value,
            "swapped_risk_adjusted_value": swapped_risk_adjusted,
            "should_swap": swapped_risk_adjusted > risk_adjusted_value,
            "expected_gain_if_swap": swapped_risk_adjusted - risk_adjusted_value,
        }

    def select_optimal_captain_pair(
        self, candidates: List[Dict], gameweek: int = 1
    ) -> Dict:
        """
        Select optimal C/VC pair from candidates.

        Args:
            candidates: List of candidate players
            gameweek: Current gameweek

        Returns:
            Optimal C/VC selection with analysis
        """
        if len(candidates) < 2:
            return {"error": "Need at least 2 candidates"}

        # Sort by expected points
        xp_key = f"expected_points_gw{gameweek}"
        sorted_candidates = sorted(
            candidates,
            key=lambda p: p.get(xp_key, p.get("expected_points", 0.0)),
            reverse=True,
        )

        best_pair = None
        best_value = -float("inf")

        # Try top 5 candidates as captain
        for i, captain in enumerate(sorted_candidates[:5]):
            # Try other candidates as vice-captain
            for j, vc in enumerate(sorted_candidates):
                if i == j:
                    continue

                analysis = self.calculate_weighted_expected_value(captain, vc, gameweek)

                if analysis["risk_adjusted_value"] > best_value:
                    best_value = analysis["risk_adjusted_value"]
                    best_pair = {
                        "captain": captain,
                        "vice_captain": vc,
                        "analysis": analysis,
                    }

        return best_pair if best_pair else {"error": "No valid pair found"}


class ChipTiming:
    """
    Determines optimal timing for FPL chips:
    - Wildcard: Major team restructure
    - Bench Boost: Maximize points from bench
    - Free Hit: Single gameweek boost
    """

    def __init__(
        self,
        wildcard_threshold: float = 20.0,
        bench_boost_threshold: float = 15.0,
        free_hit_threshold: float = 10.0,
        horizon_weeks: int = 5,
    ):
        """
        Initialize chip timing calculator.

        Args:
            wildcard_threshold: Minimum xP gain for Wildcard (default: 20 points over 5 weeks)
            bench_boost_threshold: Minimum xP gain for Bench Boost (default: 15 points)
            free_hit_threshold: Minimum xP gain for Free Hit (default: 10 points)
            horizon_weeks: Weeks to consider for chip value (default: 5)
        """
        self.wildcard_threshold = wildcard_threshold
        self.bench_boost_threshold = bench_boost_threshold
        self.free_hit_threshold = free_hit_threshold
        self.horizon_weeks = horizon_weeks

    def calculate_wildcard_value(
        self, current_squad: List[Dict], optimized_squad: List[Dict], gameweek: int = 1
    ) -> Dict:
        """
        Calculate value of playing Wildcard.

        Args:
            current_squad: Current squad players
            optimized_squad: Optimized squad after Wildcard
            gameweek: Starting gameweek

        Returns:
            Wildcard value analysis
        """
        # Calculate expected points for current vs optimized squad
        current_xp = 0.0
        optimized_xp = 0.0

        for week in range(gameweek, min(gameweek + self.horizon_weeks, 39)):
            xp_key = f"expected_points_gw{week}"

            # Current squad (best 11)
            current_week_points = sorted(
                [p.get(xp_key, p.get("expected_points", 0.0)) for p in current_squad],
                reverse=True,
            )[:11]
            current_xp += sum(current_week_points)

            # Optimized squad (best 11)
            optimized_week_points = sorted(
                [p.get(xp_key, p.get("expected_points", 0.0)) for p in optimized_squad],
                reverse=True,
            )[:11]
            optimized_xp += sum(optimized_week_points)

        gain = optimized_xp - current_xp
        should_play = gain >= self.wildcard_threshold

        return {
            "chip": "Wildcard",
            "current_squad_xp": current_xp,
            "optimized_squad_xp": optimized_xp,
            "expected_gain": gain,
            "threshold": self.wildcard_threshold,
            "should_play": should_play,
            "weeks_analyzed": min(self.horizon_weeks, 39 - gameweek + 1),
            "recommendation": (
                "Play Wildcard"
                if should_play
                else f"Wait (gain: {gain:.1f} < threshold: {self.wildcard_threshold})"
            ),
        }

    def calculate_bench_boost_value(self, squad: List[Dict], gameweek: int = 1) -> Dict:
        """
        Calculate value of playing Bench Boost.

        Args:
            squad: Current squad (15 players)
            gameweek: Target gameweek

        Returns:
            Bench Boost value analysis
        """
        xp_key = f"expected_points_gw{gameweek}"

        # Get all player expected points
        all_points = [p.get(xp_key, p.get("expected_points", 0.0)) for p in squad]

        # Sort and get best 11 (starting XI)
        sorted_points = sorted(all_points, reverse=True)
        starting_xi_points = sum(sorted_points[:11])

        # Bench points (players 12-15)
        bench_points = sum(sorted_points[11:15]) if len(sorted_points) >= 15 else 0.0

        # Bench Boost adds bench points
        gain = bench_points
        should_play = gain >= self.bench_boost_threshold

        # Check if bench has good fixtures
        bench_players = sorted(
            squad,
            key=lambda p: p.get(xp_key, p.get("expected_points", 0.0)),
            reverse=True,
        )[11:15]

        avg_bench_xp = np.mean(
            [p.get(xp_key, p.get("expected_points", 0.0)) for p in bench_players]
        )

        return {
            "chip": "Bench Boost",
            "starting_xi_xp": starting_xi_points,
            "bench_xp": bench_points,
            "expected_gain": gain,
            "avg_bench_xp": avg_bench_xp,
            "threshold": self.bench_boost_threshold,
            "should_play": should_play,
            "recommendation": (
                f"Play Bench Boost (gain: {gain:.1f} points)"
                if should_play
                else f"Wait for better bench fixtures (current: {gain:.1f} < threshold: {self.bench_boost_threshold})"
            ),
        }

    def calculate_free_hit_value(
        self, current_squad: List[Dict], free_hit_squad: List[Dict], gameweek: int = 1
    ) -> Dict:
        """
        Calculate value of playing Free Hit.

        Args:
            current_squad: Current squad
            free_hit_squad: Optimized squad for this gameweek only
            gameweek: Target gameweek

        Returns:
            Free Hit value analysis
        """
        xp_key = f"expected_points_gw{gameweek}"

        # Current squad best 11
        current_points = sorted(
            [p.get(xp_key, p.get("expected_points", 0.0)) for p in current_squad],
            reverse=True,
        )[:11]
        current_xp = sum(current_points)

        # Free Hit squad best 11
        free_hit_points = sorted(
            [p.get(xp_key, p.get("expected_points", 0.0)) for p in free_hit_squad],
            reverse=True,
        )[:11]
        free_hit_xp = sum(free_hit_points)

        gain = free_hit_xp - current_xp
        should_play = gain >= self.free_hit_threshold

        return {
            "chip": "Free Hit",
            "current_squad_xp": current_xp,
            "free_hit_squad_xp": free_hit_xp,
            "expected_gain": gain,
            "threshold": self.free_hit_threshold,
            "should_play": should_play,
            "recommendation": (
                f"Play Free Hit (gain: {gain:.1f} points)"
                if should_play
                else f"Save Free Hit (gain: {gain:.1f} < threshold: {self.free_hit_threshold})"
            ),
        }

    def analyze_all_chips(
        self,
        current_squad: List[Dict],
        optimized_squad: Optional[List[Dict]] = None,
        free_hit_squad: Optional[List[Dict]] = None,
        gameweek: int = 1,
    ) -> Dict:
        """
        Comprehensive chip analysis.

        Returns:
            Analysis for all available chips
        """
        analysis = {"gameweek": gameweek, "chips": {}}

        # Wildcard analysis
        if optimized_squad:
            analysis["chips"]["wildcard"] = self.calculate_wildcard_value(
                current_squad, optimized_squad, gameweek
            )

        # Bench Boost analysis
        analysis["chips"]["bench_boost"] = self.calculate_bench_boost_value(
            current_squad, gameweek
        )

        # Free Hit analysis
        if free_hit_squad:
            analysis["chips"]["free_hit"] = self.calculate_free_hit_value(
                current_squad, free_hit_squad, gameweek
            )

        # Overall recommendation
        playable_chips = [
            (name, chip_data)
            for name, chip_data in analysis["chips"].items()
            if chip_data.get("should_play", False)
        ]

        if playable_chips:
            # Sort by expected gain
            playable_chips.sort(
                key=lambda x: x[1].get("expected_gain", 0.0), reverse=True
            )
            best_chip = playable_chips[0]
            analysis[
                "recommendation"
            ] = f"Play {best_chip[0].replace('_', ' ').title()}: {best_chip[1]['recommendation']}"
        else:
            analysis[
                "recommendation"
            ] = "No chips recommended at this time. Save for better opportunities."

        return analysis


class RiskManagementService:
    """
    Main service orchestrating all risk management strategies.
    """

    def __init__(self):
        self.arbitrage = OwnershipArbitrage()
        self.cvc_selector = CaptainViceCaptainParadox()
        self.chip_timing = ChipTiming()

    def get_comprehensive_analysis(
        self,
        players: List[Dict],
        current_squad: List[Dict],
        gameweek: int = 1,
        optimized_squad: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Get comprehensive risk management analysis.

        Returns:
            Complete analysis with all strategies
        """
        return {
            "gameweek": gameweek,
            "ownership_arbitrage": self.arbitrage.analyze_arbitrage_opportunities(
                players, gameweek
            ),
            "chip_analysis": self.chip_timing.analyze_all_chips(
                current_squad, optimized_squad, None, gameweek
            ),
        }
