"""
FPL Portfolio Optimization and Decision Engine (Solver)
Implements Multi-Period Integer Linear Programming (ILP) using PuLP.
Features:
- Multi-Period Optimization (3-5 week horizon)
- Smart Money objective function with discount factor (γ = 0.9)
- Transfer cost penalty (-4 points per extra transfer)
- FPL constraints (budget, team limits, squad structure)
- C/VC selection based on expected points and P_start
- Integration with PLEngine for xP predictions
"""
import pulp
from typing import Dict, List, Optional, Tuple
import logging
import gc
from datetime import datetime

logger = logging.getLogger(__name__)


class FPLSolver:
    """
    FPL Portfolio Optimization Solver using Multi-Period ILP.
    Optimizes team selection across 3-5 week horizon with Smart Money objective.
    """

    def __init__(
        self,
        budget: float = 100.0,
        horizon_weeks: int = 3,
        free_transfers: int = 1,
        discount_factor: float = 0.9,
        transfer_penalty: int = 4,
    ):
        """
        Initialize FPLSolver.

        Args:
            budget: Total budget in millions (default: 100M)
            horizon_weeks: Number of weeks to optimize (3-5)
            free_transfers: Free transfers per week (default: 1)
            discount_factor: Discount factor γ for future weeks (default: 0.9)
            transfer_penalty: Points penalty per extra transfer (default: -4)
        """
        self.budget = budget
        self.horizon_weeks = min(max(horizon_weeks, 3), 5)  # Clamp between 3-5
        self.free_transfers = free_transfers
        self.discount_factor = discount_factor  # γ = 0.9 for Smart Money
        self.transfer_penalty = transfer_penalty  # -4 points per extra transfer

        # FPL Squad structure constraints (2-5-5-3)
        self.squad_structure = {
            "GK": 2,  # 2 Goalkeepers
            "DEF": 5,  # 5 Defenders
            "MID": 5,  # 5 Midfielders
            "FWD": 3,  # 3 Forwards
        }
        self.max_players_per_team = 3  # Max 3 players from same team

    def optimize_team(
        self,
        players_data: List[Dict],
        current_squad: Optional[List[int]] = None,
        locked_players: Optional[List[int]] = None,
        excluded_players: Optional[List[int]] = None,
    ) -> Dict:
        """
        Optimize FPL team using Multi-Period ILP.

        Args:
            players_data: List of player dictionaries with xP predictions from PLEngine.
                        Each player dict should have:
                        - 'id': Player ID
                        - 'name': Player name
                        - 'position': 'GK', 'DEF', 'MID', 'FWD'
                        - 'price': Player price in millions
                        - 'team_id': Team ID
                        - 'expected_points': List of expected points for each week [gw1, gw2, ...]
                        - 'p_start': List of starting probabilities for each week [gw1, gw2, ...]
            current_squad: List of current squad player IDs
            locked_players: List of player IDs to keep in squad
            excluded_players: List of player IDs to exclude

        Returns:
            Dictionary with optimized team, transfers, and starting XI
        """
        try:
            # Create optimization model
            prob, variables = self._create_ilp_model(
                players_data, current_squad, locked_players, excluded_players
            )

            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))  # Silent mode

            # Check solution status
            if pulp.LpStatus[prob.status] != "Optimal":
                logger.warning(f"Solver status: {pulp.LpStatus[prob.status]}")
                # Try to get solution anyway if feasible
                if pulp.LpStatus[prob.status] != "Feasible":
                    raise ValueError(
                        f"Optimization failed: {pulp.LpStatus[prob.status]}"
                    )

            # Extract solution
            solution = self._extract_solution(
                prob, variables, players_data, current_squad
            )

            # Select C/VC
            captain_vc = self._select_captain_vice_captain(
                solution["starting_xi_week1"], players_data
            )
            solution["captain"] = captain_vc["captain"]
            solution["vice_captain"] = captain_vc["vice_captain"]

            # Memory management
            gc.collect()

            return solution

        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            raise

    def _create_ilp_model(
        self,
        players_data: List[Dict],
        current_squad: Optional[List[int]] = None,
        locked_players: Optional[List[int]] = None,
        excluded_players: Optional[List[int]] = None,
    ) -> Tuple[pulp.LpProblem, Dict]:
        """
        Create Multi-Period ILP optimization model.

        Returns:
            (problem, variables_dict) tuple
        """
        # Create problem: Maximize expected points
        prob = pulp.LpProblem("FPL_Multi_Period_Optimization", pulp.LpMaximize)

        # Player indices
        player_ids = [p["id"] for p in players_data]
        _n_players = len(players_data)
        weeks = list(range(1, self.horizon_weeks + 1))

        # Create player lookup
        player_dict = {p["id"]: p for p in players_data}

        # Decision variables
        # x[i][w] = 1 if player i is in squad in week w, 0 otherwise
        x = pulp.LpVariable.dicts(
            "player_in_squad", [(i, w) for i in player_ids for w in weeks], cat="Binary"
        )

        # y[i][w] = 1 if player i is in starting XI in week w, 0 otherwise
        y = pulp.LpVariable.dicts(
            "player_in_xi", [(i, w) for i in player_ids for w in weeks], cat="Binary"
        )

        # t[i][w] = 1 if player i is transferred IN in week w, 0 otherwise
        t = pulp.LpVariable.dicts(
            "transfer_in", [(i, w) for i in player_ids for w in weeks], cat="Binary"
        )

        # u[i][w] = 1 if player i is transferred OUT in week w, 0 otherwise
        u = pulp.LpVariable.dicts(
            "transfer_out", [(i, w) for i in player_ids for w in weeks], cat="Binary"
        )

        # Objective function: Smart Money with discount factor
        # Maximize: Σ_w (γ^(w-1) * Σ_i (xP[i][w] * y[i][w])) - 4 * Σ_w (extra_transfers[w])

        # Points component with discount factor
        points_component = pulp.lpSum(
            [
                (self.discount_factor ** (w - 1))  # Discount factor: γ^(w-1)
                * pulp.lpSum(
                    [
                        player_dict[i].get(
                            "expected_points", [0.0] * self.horizon_weeks
                        )[w - 1]
                        * y[(i, w)]
                        for i in player_ids
                    ]
                )
                for w in weeks
            ]
        )

        # Transfer penalty variables (extra transfers beyond free transfers)
        extra_transfers_vars = {}
        transfer_penalty_component = 0

        for w in weeks:
            # Create variable for extra transfers in week w
            extra_transfers_vars[w] = pulp.LpVariable(
                f"extra_transfers_w{w}", lowBound=0, cat="Continuous"
            )

            if w == 1:
                # Week 1: transfers from current squad
                if current_squad:
                    transfers_in = pulp.lpSum(
                        [t[(i, w)] for i in player_ids if i not in current_squad]
                    )
                    transfers_out = pulp.lpSum([u[(i, w)] for i in current_squad])
                    total_transfers = transfers_in + transfers_out
                    # Extra transfers = total - free transfers (if positive)
                    prob += (
                        extra_transfers_vars[w] >= total_transfers - self.free_transfers
                    )
                else:
                    prob += extra_transfers_vars[w] == 0
            else:
                # Week w > 1: transfers from previous week
                transfers_in = pulp.lpSum([t[(i, w)] for i in player_ids])
                transfers_out = pulp.lpSum([u[(i, w)] for i in player_ids])
                total_transfers = transfers_in + transfers_out
                # Extra transfers = total - free transfers (if positive)
                prob += extra_transfers_vars[w] >= total_transfers - self.free_transfers

            # Add penalty to objective
            transfer_penalty_component += (
                self.transfer_penalty * extra_transfers_vars[w]
            )

        # Final objective: maximize points - transfer penalties
        prob += points_component - transfer_penalty_component, "Smart_Money_Objective"

        # ==================== CONSTRAINTS ====================

        # 1. Budget constraint (100M)
        for w in weeks:
            prob += (
                pulp.lpSum([player_dict[i]["price"] * x[(i, w)] for i in player_ids])
                <= self.budget
            )

        # 2. Squad structure constraints (2-5-5-3)
        _position_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        for w in weeks:
            for pos_name, count in self.squad_structure.items():
                prob += (
                    pulp.lpSum(
                        [
                            x[(i, w)]
                            for i in player_ids
                            if player_dict[i].get("position") == pos_name
                        ]
                    )
                    == count
                )

        # 3. Starting XI constraints (exactly 11 players)
        for w in weeks:
            prob += pulp.lpSum([y[(i, w)] for i in player_ids]) == 11

        # 4. Starting XI must be in squad
        for i in player_ids:
            for w in weeks:
                prob += y[(i, w)] <= x[(i, w)]

        # 5. Starting XI position constraints (at least 1 GK, 3 DEF, 3 MID, 1 FWD)
        for w in weeks:
            # At least 1 goalkeeper
            prob += (
                pulp.lpSum(
                    [
                        y[(i, w)]
                        for i in player_ids
                        if player_dict[i].get("position") == "GK"
                    ]
                )
                >= 1
            )

            # At least 3 defenders
            prob += (
                pulp.lpSum(
                    [
                        y[(i, w)]
                        for i in player_ids
                        if player_dict[i].get("position") == "DEF"
                    ]
                )
                >= 3
            )

            # At least 3 midfielders
            prob += (
                pulp.lpSum(
                    [
                        y[(i, w)]
                        for i in player_ids
                        if player_dict[i].get("position") == "MID"
                    ]
                )
                >= 3
            )

            # At least 1 forward
            prob += (
                pulp.lpSum(
                    [
                        y[(i, w)]
                        for i in player_ids
                        if player_dict[i].get("position") == "FWD"
                    ]
                )
                >= 1
            )

        # 6. Team limit constraint (max 3 players per team)
        team_ids = set(p.get("team_id") for p in players_data if p.get("team_id"))
        for team_id in team_ids:
            for w in weeks:
                prob += (
                    pulp.lpSum(
                        [
                            x[(i, w)]
                            for i in player_ids
                            if player_dict[i].get("team_id") == team_id
                        ]
                    )
                    <= self.max_players_per_team
                )

        # 7. Transfer logic constraints
        for w in weeks:
            if w == 1:
                # Week 1: transfers from current squad
                if current_squad:
                    for i in player_ids:
                        if i in current_squad:
                            # If player in current squad and in week 1 squad, no transfer
                            # If player in current squad but not in week 1 squad, transfer out
                            prob += u[(i, w)] >= (1 - x[(i, w)])
                            prob += (
                                t[(i, w)] == 0
                            )  # Can't transfer in from current squad
                        else:
                            # If player not in current squad but in week 1 squad, transfer in
                            prob += t[(i, w)] >= x[(i, w)]
                            prob += (
                                u[(i, w)] == 0
                            )  # Can't transfer out if not in current squad
                else:
                    # No current squad: all are transfers in
                    for i in player_ids:
                        prob += t[(i, w)] == x[(i, w)]
                        prob += u[(i, w)] == 0
            else:
                # Week w > 1: transfers from previous week
                for i in player_ids:
                    # If player in squad this week but not last week, must transfer in
                    prob += t[(i, w)] >= x[(i, w)] - x[(i, w - 1)]

                    # If player not in squad this week but was last week, must transfer out
                    prob += u[(i, w)] >= x[(i, w - 1)] - x[(i, w)]

                    # Transfer in and out are mutually exclusive
                    prob += t[(i, w)] + u[(i, w)] <= 1

                    # If no change, no transfers
                    prob += t[(i, w)] + u[(i, w)] <= 1

        # 8. Locked players constraint
        if locked_players:
            for i in locked_players:
                if i in player_ids:
                    for w in weeks:
                        prob += x[(i, w)] == 1

        # 9. Excluded players constraint
        if excluded_players:
            for i in excluded_players:
                if i in player_ids:
                    for w in weeks:
                        prob += x[(i, w)] == 0

        variables = {"x": x, "y": y, "t": t, "u": u}

        return prob, variables

    def _extract_solution(
        self,
        prob: pulp.LpProblem,
        variables: Dict,
        players_data: List[Dict],
        current_squad: Optional[List[int]] = None,
    ) -> Dict:
        """
        Extract solution from solved ILP model.

        Returns:
            Dictionary with squad, starting XI, transfers, and expected points
        """
        x = variables["x"]
        y = variables["y"]
        t = variables["t"]
        u = variables["u"]

        player_dict = {p["id"]: p for p in players_data}
        weeks = list(range(1, self.horizon_weeks + 1))

        solution = {
            "squad_by_week": {},
            "starting_xi_by_week": {},
            "transfers_by_week": {},
            "expected_points_by_week": {},
            "total_expected_points": 0.0,
            "total_transfer_cost": 0,
            "total_cost": 0.0,
        }

        # Extract solution for each week
        for w in weeks:
            squad = []
            starting_xi = []
            transfers_in = []
            transfers_out = []
            week_points = 0.0

            for player_id in player_dict.keys():
                # Squad
                if pulp.value(x[(player_id, w)]) == 1:
                    player_info = {
                        "id": player_id,
                        "name": player_dict[player_id].get("name", ""),
                        "position": player_dict[player_id].get("position", ""),
                        "price": player_dict[player_id].get("price", 0.0),
                        "team_id": player_dict[player_id].get("team_id", 0),
                        "expected_points": player_dict[player_id].get(
                            "expected_points", [0.0] * self.horizon_weeks
                        )[w - 1],
                    }
                    squad.append(player_info)

                    # Starting XI
                    if pulp.value(y[(player_id, w)]) == 1:
                        starting_xi.append(player_info)
                        week_points += player_info["expected_points"]

                # Transfers
                if w > 1:
                    if pulp.value(t[(player_id, w)]) == 1:
                        transfers_in.append(
                            {
                                "id": player_id,
                                "name": player_dict[player_id].get("name", ""),
                                "price": player_dict[player_id].get("price", 0.0),
                            }
                        )

                    if pulp.value(u[(player_id, w)]) == 1:
                        transfers_out.append(
                            {
                                "id": player_id,
                                "name": player_dict[player_id].get("name", ""),
                                "price": player_dict[player_id].get("price", 0.0),
                            }
                        )

            solution["squad_by_week"][f"week_{w}"] = squad
            solution["starting_xi_by_week"][f"week_{w}"] = starting_xi
            solution["expected_points_by_week"][f"week_{w}"] = week_points

            if w > 1:
                total_transfers = len(transfers_in) + len(transfers_out)
                extra_transfers = max(0, total_transfers - self.free_transfers)
                solution["transfers_by_week"][f"week_{w}"] = {
                    "transfers_in": transfers_in,
                    "transfers_out": transfers_out,
                    "transfer_count": total_transfers,
                    "extra_transfers": extra_transfers,
                    "transfer_cost": extra_transfers * self.transfer_penalty,
                }
                solution["total_transfer_cost"] += solution["transfers_by_week"][
                    f"week_{w}"
                ]["transfer_cost"]

            # Week 1 transfers (if any)
            if w == 1 and current_squad:
                transfers_in_w1 = []
                transfers_out_w1 = []
                for player_id in player_dict.keys():
                    if pulp.value(t[(player_id, w)]) == 1:
                        transfers_in_w1.append(
                            {
                                "id": player_id,
                                "name": player_dict[player_id].get("name", ""),
                                "price": player_dict[player_id].get("price", 0.0),
                            }
                        )
                    if pulp.value(u[(player_id, w)]) == 1:
                        transfers_out_w1.append(
                            {
                                "id": player_id,
                                "name": player_dict[player_id].get("name", ""),
                                "price": player_dict[player_id].get("price", 0.0),
                            }
                        )

                if transfers_in_w1 or transfers_out_w1:
                    total_transfers_w1 = len(transfers_in_w1) + len(transfers_out_w1)
                    extra_transfers_w1 = max(
                        0, total_transfers_w1 - self.free_transfers
                    )
                    solution["transfers_by_week"]["week_1"] = {
                        "transfers_in": transfers_in_w1,
                        "transfers_out": transfers_out_w1,
                        "transfer_count": total_transfers_w1,
                        "extra_transfers": extra_transfers_w1,
                        "transfer_cost": extra_transfers_w1 * self.transfer_penalty,
                    }
                    solution["total_transfer_cost"] += solution["transfers_by_week"][
                        "week_1"
                    ]["transfer_cost"]

            solution["total_expected_points"] += (
                self.discount_factor ** (w - 1)
            ) * week_points

        # Week 1 squad and starting XI (for easy access)
        solution["squad_week1"] = solution["squad_by_week"]["week_1"]
        solution["starting_xi_week1"] = solution["starting_xi_by_week"]["week_1"]

        # Total cost
        solution["total_cost"] = sum(p["price"] for p in solution["squad_week1"])

        return solution

    def _select_captain_vice_captain(
        self, starting_xi: List[Dict], players_data: List[Dict]
    ) -> Dict:
        """
        Select Captain and Vice-Captain based on expected points and P_start.
        Score = expected_points * P_start

        Args:
            starting_xi: List of players in starting XI
            players_data: Full players data with P_start probabilities

        Returns:
            Dictionary with captain and vice_captain info
        """
        player_dict = {p["id"]: p for p in players_data}

        # Calculate scores for each player
        scores = []
        for player in starting_xi:
            player_id = player["id"]
            player_info = player_dict.get(player_id, {})

            # Get expected points for week 1
            expected_points = player_info.get(
                "expected_points", [0.0] * self.horizon_weeks
            )[0]

            # Get P_start for week 1
            p_start = player_info.get("p_start", [0.7] * self.horizon_weeks)[0]

            # Score = expected_points * P_start
            score = expected_points * p_start

            scores.append(
                {
                    "id": player_id,
                    "name": player.get("name", ""),
                    "position": player.get("position", ""),
                    "expected_points": expected_points,
                    "p_start": p_start,
                    "score": score,
                }
            )

        # Sort by score (descending)
        scores.sort(key=lambda x: x["score"], reverse=True)

        # Select captain (highest score)
        captain = scores[0] if scores else None

        # Select vice-captain (second highest, different position if possible)
        vice_captain = None
        if len(scores) > 1:
            # Try to select from different position
            captain_pos = captain["position"] if captain else None
            for candidate in scores[1:]:
                if candidate["position"] != captain_pos:
                    vice_captain = candidate
                    break

            # If no different position found, take second best
            if not vice_captain:
                vice_captain = scores[1]

        return {"captain": captain, "vice_captain": vice_captain}

    def optimize_from_plengine(
        self,
        plengine,
        players_data: List[Dict],
        fixture_data_by_week: Dict[int, List[Dict]],
        current_squad: Optional[List[int]] = None,
        locked_players: Optional[List[int]] = None,
        excluded_players: Optional[List[int]] = None,
        include_strategy_notes: bool = False,
        strategy_service=None,
        all_players_for_strategy: Optional[List[Dict]] = None,
        current_squad_data: Optional[List[Dict]] = None,
        gameweek: int = 1,
    ) -> Dict:
        """
        Optimize team using PLEngine predictions.

        Args:
            plengine: PLEngine instance for xP predictions
            players_data: List of player base data (id, name, position, price, team_id)
            fixture_data_by_week: Dictionary mapping week -> list of fixture data for each player
            current_squad: Current squad player IDs
            locked_players: Locked player IDs
            excluded_players: Excluded player IDs
            include_strategy_notes: Whether to include strategy notes (default: False)
            strategy_service: StrategyService instance for strategy analysis
            all_players_for_strategy: All players data for ownership analysis
            current_squad_data: Current squad player data for chip analysis
            gameweek: Current gameweek

        Returns:
            Optimized team solution as JSON-serializable dictionary
        """
        # Get xP predictions from PLEngine for each player and week
        enriched_players = []

        for player in players_data:
            player_id = player["id"]
            expected_points = []
            p_start_list = []

            for week in range(1, self.horizon_weeks + 1):
                # Get fixture data for this week
                week_fixtures = fixture_data_by_week.get(week, [])
                player_fixture = next(
                    (f for f in week_fixtures if f.get("player_id") == player_id), None
                )

                # Get prediction from PLEngine
                try:
                    prediction = plengine.predict(
                        player_data=player, fixture_data=player_fixture
                    )

                    expected_points.append(prediction["expected_points"])
                    p_start_list.append(prediction.get("p_start", 0.7))
                except Exception as e:
                    logger.warning(
                        f"Error predicting for player {player_id} week {week}: {str(e)}"
                    )
                    expected_points.append(0.0)
                    p_start_list.append(0.7)

            # Enrich player data with predictions
            enriched_player = player.copy()
            enriched_player["expected_points"] = expected_points
            enriched_player["p_start"] = p_start_list
            enriched_players.append(enriched_player)

        # Optimize
        solution = self.optimize_team(
            enriched_players, current_squad, locked_players, excluded_players
        )

        # Convert to JSON-serializable format
        json_solution = self._to_json_format(solution)

        # Add strategy notes if requested
        if include_strategy_notes and strategy_service and all_players_for_strategy:
            try:
                strategy_notes = strategy_service.generate_strategy_notes(
                    json_solution,
                    all_players_for_strategy,
                    plengine,
                    current_squad_data,
                    gameweek,
                )
                json_solution = strategy_service.add_strategy_notes_to_solution(
                    json_solution, strategy_notes
                )
            except Exception as e:
                logger.error(f"Error adding strategy notes: {str(e)}")
                json_solution["strategy_notes_error"] = str(e)

        return json_solution

    def _to_json_format(self, solution: Dict) -> Dict:
        """
        Convert solution to JSON-serializable format.

        Returns:
            JSON-serializable dictionary
        """
        json_solution = {
            "squad_week1": solution.get("squad_week1", []),
            "starting_xi_week1": solution.get("starting_xi_week1", []),
            "captain": solution.get("captain"),
            "vice_captain": solution.get("vice_captain"),
            "transfers_by_week": solution.get("transfers_by_week", {}),
            "expected_points_by_week": solution.get("expected_points_by_week", {}),
            "total_expected_points": float(solution.get("total_expected_points", 0.0)),
            "total_transfer_cost": int(solution.get("total_transfer_cost", 0)),
            "total_cost": float(solution.get("total_cost", 0.0)),
            "horizon_weeks": self.horizon_weeks,
            "discount_factor": self.discount_factor,
            "optimization_timestamp": datetime.now().isoformat(),
        }

        return json_solution
