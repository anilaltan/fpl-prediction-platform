"""
Team Optimization Solver for FPL
Implements Multi-Period Integer Linear Programming (ILP) for team selection.
Uses PuLP library for optimization with Smart Money objective function.
"""
import pulp
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TeamSolver:
    """
    Multi-Period ILP Solver for FPL team optimization.
    Optimizes team selection across multiple gameweeks (3-5 week horizon).
    """
    
    def __init__(
        self,
        budget: float = 100.0,
        horizon_weeks: int = 3,
        free_transfers: int = 1
    ):
        """
        Initialize the solver.
        
        Args:
            budget: Total budget in millions (default: 100M)
            horizon_weeks: Number of weeks to optimize (3-5)
            free_transfers: Free transfers per week (default: 1)
        """
        self.budget = budget
        self.horizon_weeks = min(max(horizon_weeks, 3), 5)  # Clamp between 3-5
        self.free_transfers = free_transfers
        self.transfer_penalty = 4  # Points penalty per extra transfer
        
        # Squad structure constraints
        self.squad_structure = {
            'GK': 2,  # 2 Goalkeepers
            'DEF': 5,  # 5 Defenders
            'MID': 5,  # 5 Midfielders
            'FWD': 3   # 3 Forwards
        }
        self.max_players_per_team = 3
    
    def create_optimization_model(
        self,
        players: List[Dict],
        current_squad: Optional[List[int]] = None,
        locked_players: Optional[List[int]] = None,
        excluded_players: Optional[List[int]] = None
    ) -> Tuple[pulp.LpProblem, Dict]:
        """
        Create multi-period ILP optimization model.
        
        Args:
            players: List of player dictionaries with predictions for each week
            current_squad: List of current squad player IDs
            locked_players: List of player IDs to keep in squad
            excluded_players: List of player IDs to exclude
        
        Returns:
            (model, variables_dict) tuple
        """
        # Create problem: Maximize expected points
        prob = pulp.LpProblem("FPL_Team_Optimization", pulp.LpMaximize)
        
        # Player indices
        player_ids = [p['id'] for p in players]
        n_players = len(players)
        weeks = list(range(1, self.horizon_weeks + 1))
        
        # Decision variables
        # x[i][w] = 1 if player i is in squad in week w, 0 otherwise
        x = pulp.LpVariable.dicts(
            "player_in_squad",
            [(i, w) for i in player_ids for w in weeks],
            cat='Binary'
        )
        
        # y[i][w] = 1 if player i is in starting XI in week w, 0 otherwise
        y = pulp.LpVariable.dicts(
            "player_in_xi",
            [(i, w) for i in player_ids for w in weeks],
            cat='Binary'
        )
        
        # t[i][w] = 1 if player i is transferred in week w, 0 otherwise
        t = pulp.LpVariable.dicts(
            "transfer_in",
            [(i, w) for i in player_ids for w in weeks],
            cat='Binary'
        )
        
        # u[i][w] = 1 if player i is transferred out in week w, 0 otherwise
        u = pulp.LpVariable.dicts(
            "transfer_out",
            [(i, w) for i in player_ids for w in weeks],
            cat='Binary'
        )
        
        # Objective function: Maximize expected points - transfer penalties
        # Σ(Σ(expected_points[i][w] * y[i][w])) - 4 * Σ(Σ(t[i][w] + u[i][w]) - free_transfers)
        objective = pulp.lpSum([
            players[player_ids.index(i)].get(f'expected_points_gw{w}', 0.0) * y[(i, w)]
            for i in player_ids
            for w in weeks
        ])
        
        # Transfer penalty: -4 points per extra transfer beyond free transfers
        transfer_cost = pulp.lpSum([
            self.transfer_penalty * (t[(i, w)] + u[(i, w)])
            for i in player_ids
            for w in weeks
        ]) - (self.free_transfers * self.transfer_penalty * self.horizon_weeks)
        
        # Final objective
        prob += objective - transfer_cost, "Total_Expected_Points"
        
        # Constraints
        
        # 1. Budget constraint (for each week)
        for w in weeks:
            prob += pulp.lpSum([
                players[player_ids.index(i)]['price'] * x[(i, w)]
                for i in player_ids
            ]) <= self.budget, f"Budget_Week_{w}"
        
        # 2. Squad size constraint: Exactly 15 players
        for w in weeks:
            prob += pulp.lpSum([x[(i, w)] for i in player_ids]) == 15, f"Squad_Size_Week_{w}"
        
        # 3. Starting XI size: Exactly 11 players
        for w in weeks:
            prob += pulp.lpSum([y[(i, w)] for i in player_ids]) == 11, f"Starting_XI_Size_Week_{w}"
        
        # 4. Starting XI must be subset of squad
        for i in player_ids:
            for w in weeks:
                prob += y[(i, w)] <= x[(i, w)], f"XI_Subset_{i}_{w}"
        
        # 5. Position constraints
        position_map = {'GK': 'GK', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD'}
        for pos, count in self.squad_structure.items():
            for w in weeks:
                prob += pulp.lpSum([
                    x[(i, w)]
                    for i in player_ids
                    if players[player_ids.index(i)].get('position') == pos
                ]) == count, f"Position_{pos}_Week_{w}"
        
        # 6. Starting XI position constraints (at least 1 GK, etc.)
        for w in weeks:
            # At least 1 goalkeeper in starting XI
            prob += pulp.lpSum([
                y[(i, w)]
                for i in player_ids
                if players[player_ids.index(i)].get('position') == 'GK'
            ]) >= 1, f"XI_GK_Week_{w}"
            
            # At least 3 defenders
            prob += pulp.lpSum([
                y[(i, w)]
                for i in player_ids
                if players[player_ids.index(i)].get('position') == 'DEF'
            ]) >= 3, f"XI_DEF_Week_{w}"
            
            # At least 1 forward
            prob += pulp.lpSum([
                y[(i, w)]
                for i in player_ids
                if players[player_ids.index(i)].get('position') == 'FWD'
            ]) >= 1, f"XI_FWD_Week_{w}"
        
        # 7. Max 3 players per team constraint
        teams = set(p.get('team_id') for p in players if 'team_id' in p)
        for team_id in teams:
            for w in weeks:
                prob += pulp.lpSum([
                    x[(i, w)]
                    for i in player_ids
                    if players[player_ids.index(i)].get('team_id') == team_id
                ]) <= self.max_players_per_team, f"Team_Limit_{team_id}_Week_{w}"
        
        # 8. Transfer constraints
        if current_squad:
            # Week 1: Initial transfers
            for i in player_ids:
                is_current = i in current_squad
                # If not in current squad but in week 1 squad, must transfer in
                prob += t[(i, 1)] >= x[(i, 1)] - (1 if is_current else 0), f"Transfer_In_1_{i}"
                # If in current squad but not in week 1 squad, must transfer out
                prob += u[(i, 1)] >= (1 if is_current else 0) - x[(i, 1)], f"Transfer_Out_1_{i}"
        
        # Subsequent weeks: Transfers based on previous week
        for w in weeks[1:]:
            for i in player_ids:
                # Transfer in: not in previous week, but in current week
                prob += t[(i, w)] >= x[(i, w)] - x[(i, w-1)], f"Transfer_In_{w}_{i}"
                # Transfer out: in previous week, but not in current week
                prob += u[(i, w)] >= x[(i, w-1)] - x[(i, w)], f"Transfer_Out_{w}_{i}"
        
        # 9. Locked players (must be in squad)
        if locked_players:
            for player_id in locked_players:
                if player_id in player_ids:
                    for w in weeks:
                        prob += x[(player_id, w)] == 1, f"Locked_Player_{player_id}_Week_{w}"
        
        # 10. Excluded players (cannot be in squad)
        if excluded_players:
            for player_id in excluded_players:
                if player_id in player_ids:
                    for w in weeks:
                        prob += x[(player_id, w)] == 0, f"Excluded_Player_{player_id}_Week_{w}"
        
        variables = {
            'x': x,  # Squad selection
            'y': y,  # Starting XI
            't': t,  # Transfers in
            'u': u   # Transfers out
        }
        
        return prob, variables
    
    def solve(
        self,
        players: List[Dict],
        current_squad: Optional[List[int]] = None,
        locked_players: Optional[List[int]] = None,
        excluded_players: Optional[List[int]] = None,
        solver: Optional[str] = None
    ) -> Dict:
        """
        Solve the optimization problem.
        
        Args:
            players: List of players with predictions
            current_squad: Current squad player IDs
            locked_players: Players to keep
            excluded_players: Players to exclude
            solver: Solver to use (default: COIN_CMD or PULP_CBC_CMD)
        
        Returns:
            Dictionary with solution details
        """
        try:
            # Create model
            prob, variables = self.create_optimization_model(
                players, current_squad, locked_players, excluded_players
            )
            
            # Solve
            if solver is None:
                # Try to use COIN_CMD (CBC), fallback to default
                try:
                    solver = pulp.COIN_CMD(msg=0)
                except:
                    solver = pulp.PULP_CBC_CMD(msg=0)
            
            prob.solve(solver)
            
            # Check solution status
            if prob.status != pulp.LpStatusOptimal:
                logger.warning(f"Optimization status: {pulp.LpStatus[prob.status]}")
                return {
                    'status': pulp.LpStatus[prob.status],
                    'optimal': False,
                    'squads': {},
                    'starting_xis': {},
                    'transfers': {},
                    'total_points': 0.0,
                    'total_transfers': 0
                }
            
            # Extract solution
            solution = self._extract_solution(prob, variables, players)
            
            logger.info(f"Optimization completed. Total expected points: {solution['total_points']:.2f}")
            
            return solution
            
        except Exception as e:
            logger.error(f"Error solving optimization: {str(e)}")
            raise
    
    def _extract_solution(
        self,
        prob: pulp.LpProblem,
        variables: Dict,
        players: List[Dict]
    ) -> Dict:
        """Extract solution from solved model"""
        player_ids = [p['id'] for p in players]
        weeks = list(range(1, self.horizon_weeks + 1))
        
        squads = {}
        starting_xis = {}
        transfers = {}
        
        total_points = pulp.value(prob.objective)
        total_transfers = 0
        
        for w in weeks:
            # Squad for week w
            squad = [
                i for i in player_ids
                if pulp.value(variables['x'][(i, w)]) == 1
            ]
            squads[w] = squad
            
            # Starting XI for week w
            xi = [
                i for i in player_ids
                if pulp.value(variables['y'][(i, w)]) == 1
            ]
            starting_xis[w] = xi
            
            # Transfers for week w
            transfers_in = [
                i for i in player_ids
                if pulp.value(variables['t'][(i, w)]) == 1
            ]
            transfers_out = [
                i for i in player_ids
                if pulp.value(variables['u'][(i, w)]) == 1
            ]
            
            transfers[w] = {
                'in': transfers_in,
                'out': transfers_out,
                'count': len(transfers_in) + len(transfers_out),
                'cost': max(0, (len(transfers_in) + len(transfers_out) - self.free_transfers) * self.transfer_penalty)
            }
            
            total_transfers += transfers[w]['count']
        
        # Calculate detailed points breakdown
        points_breakdown = {}
        for w in weeks:
            week_points = sum(
                players[player_ids.index(i)].get(f'expected_points_gw{w}', 0.0)
                for i in starting_xis[w]
            )
            transfer_cost = transfers[w]['cost']
            net_points = week_points - transfer_cost
            points_breakdown[w] = {
                'expected_points': week_points,
                'transfer_cost': transfer_cost,
                'net_points': net_points
            }
        
        return {
            'status': 'Optimal',
            'optimal': True,
            'squads': squads,
            'starting_xis': starting_xis,
            'transfers': transfers,
            'points_breakdown': points_breakdown,
            'total_points': float(total_points) if total_points else 0.0,
            'total_transfers': total_transfers,
            'budget_used': {
                w: sum(players[player_ids.index(i)]['price'] for i in squads[w])
                for w in weeks
            }
        }
    
    def calculate_expected_points(
        self,
        player: Dict,
        week: int,
        predictions: Optional[Dict] = None
    ) -> float:
        """
        Calculate expected points for a player in a given week.
        Combines predictions from ML engine.
        
        Args:
            player: Player data
            week: Gameweek number
            predictions: ML predictions (xg, xa, xcs, expected_minutes, etc.)
        
        Returns:
            Expected points
        """
        if predictions is None:
            predictions = {}
        
        position = player.get('position', 'MID')
        expected_minutes = predictions.get('expected_minutes', 90.0)
        minutes_factor = expected_minutes / 90.0
        
        # Base points calculation
        expected_points = 0.0
        
        # Attack points
        xg = predictions.get('xg', 0.0)
        xa = predictions.get('xa', 0.0)
        
        if position in ['MID', 'FWD']:
            # Goals: 4 points for MID, 5 for FWD
            goal_points = 5 if position == 'FWD' else 4
            expected_points += xg * goal_points * minutes_factor
            
            # Assists: 3 points
            expected_points += xa * 3 * minutes_factor
        
        # Defense points
        xcs = predictions.get('xcs', 0.0)
        if position in ['GK', 'DEF']:
            # Clean sheet: 4 points for GK/DEF
            expected_points += xcs * 4 * minutes_factor
        
        # Appearance points (2 points for playing 60+ minutes)
        if expected_minutes >= 60:
            expected_points += 2 * (expected_minutes / 90.0)
        elif expected_minutes > 0:
            expected_points += 1 * (expected_minutes / 90.0)
        
        # Bonus points (simplified: based on xG+xA)
        if position in ['MID', 'FWD']:
            bonus_factor = min(1.0, (xg + xa) / 2.0)
            expected_points += bonus_factor * 1.0  # Average bonus points
        
        # DefCon floor points (for 2025/26 rules)
        floor_points = predictions.get('floor_points', 0.0)
        expected_points = max(expected_points, floor_points)
        
        return float(expected_points)