// TypeScript types for API responses matching FastAPI schemas

export interface PlayerOptimizationData {
  id: number
  name: string
  position: string
  price: number
  team_id: number
  team_name: string
  expected_points_gw1: number
  expected_points_gw2?: number
  expected_points_gw3?: number
  expected_points_gw4?: number
  expected_points_gw5?: number
}

export interface TeamOptimizationRequest {
  players: PlayerOptimizationData[]
  current_squad?: number[]
  locked_players?: number[]
  excluded_players?: number[]
  budget?: number
  horizon_weeks?: number
  free_transfers?: number
}

export interface TransferInfo {
  transfers_in: number[]
  transfers_out: number[]
  count: number
  cost: number
}

export interface WeekPoints {
  expected_points: number
  transfer_cost: number
  net_points: number
}

export interface TeamOptimizationResponse {
  status: string
  optimal: boolean
  squads: Record<number, number[]>
  starting_xis: Record<number, number[]>
  transfers: Record<number, TransferInfo>
  points_breakdown: Record<number, WeekPoints>
  total_points: number
  total_transfers: number
  budget_used: Record<number, number>
}

export interface TeamPlanRequest {
  players: PlayerOptimizationData[]
  current_squad?: number[]
  locked_players?: number[]
  excluded_players?: number[]
  budget?: number
  horizon_weeks?: number
  free_transfers?: number
}

export interface TransferStrategy {
  gameweek: number
  transfers_in: number[]
  transfers_out: number[]
  transfer_count: number
  transfer_cost: number
  expected_points_gain: number
}

export interface TeamPlanResponse {
  status: string
  optimal: boolean
  horizon_weeks: number
  squads: Record<number, number[]>
  starting_xis: Record<number, number[]>
  transfer_strategy: TransferStrategy[]
  total_expected_points: number
  total_transfer_cost: number
  net_expected_points: number
  budget_used: Record<number, number>
}

export interface MarketIntelligencePlayer {
  player_id: number
  name: string
  position: string
  team: string
  price: number
  xp: number
  ownership: number
  xp_rank: number
  ownership_rank: number
  arbitrage_score: number
  category: 'Differential' | 'Overvalued' | 'Neutral'
}

export interface MarketIntelligenceResponse {
  gameweek: number
  season: string
  players: MarketIntelligencePlayer[]
  total_players: number
  differentials_count: number
  overvalued_count: number
  neutral_count: number
}
