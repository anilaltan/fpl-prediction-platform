'use client'

import { useMemo } from 'react'
import type { TeamPlanResponse, PlayerOptimizationData } from '@/lib/types/api'

interface PlanningHeatmapProps {
  plan: TeamPlanResponse
  players: PlayerOptimizationData[] // Map of player ID to player data
  colorBy?: 'expected_points' | 'fixture_difficulty' // What to color by (default: expected_points)
}

/**
 * PlanningHeatmap Component
 * Visualizes multi-period transfer strategy as a heatmap
 * Shows players vs gameweeks with color coding for expected points
 */
export function PlanningHeatmap({
  plan,
  players,
  colorBy = 'expected_points',
}: PlanningHeatmapProps) {
  // Create player map for quick lookup
  const playerMap = useMemo(() => {
    const map = new Map<number, PlayerOptimizationData>()
    players.forEach((p) => map.set(p.id, p))
    return map
  }, [players])

  // Get all unique players across all gameweeks
  const allPlayerIds = useMemo(() => {
    const ids = new Set<number>()
    for (let gw = 1; gw <= plan.horizon_weeks; gw++) {
      const squad = plan.squads[gw] || []
      squad.forEach((id) => ids.add(id))
    }
    return Array.from(ids)
  }, [plan])

  // Organize players by position for better grouping
  const playersByPosition = useMemo(() => {
    const byPos: Record<string, number[]> = { GK: [], DEF: [], MID: [], FWD: [] }
    allPlayerIds.forEach((id) => {
      const player = playerMap.get(id)
      if (player) {
        byPos[player.position]?.push(id)
      }
    })
    return byPos
  }, [allPlayerIds, playerMap])

  // Get expected points for a player in a gameweek
  const getExpectedPoints = (playerId: number, gameweek: number): number => {
    const player = playerMap.get(playerId)
    if (!player) {return 0}

    if (gameweek === 1) {return player.expected_points_gw1}
    if (gameweek === 2) {return player.expected_points_gw2 ?? player.expected_points_gw1}
    if (gameweek === 3) {return player.expected_points_gw3 ?? player.expected_points_gw1}
    if (gameweek === 4) {return player.expected_points_gw4 ?? player.expected_points_gw1}
    if (gameweek === 5) {return player.expected_points_gw5 ?? player.expected_points_gw1}
    return player.expected_points_gw1
  }

  // Get color intensity based on expected points
  const getColorIntensity = (points: number): string => {
    // Normalize points to 0-1 scale (assuming max ~15 points per player)
    const normalized = Math.min(points / 15, 1)
    
    if (normalized >= 0.8) {return 'bg-green-600'} // High points
    if (normalized >= 0.6) {return 'bg-green-500'}
    if (normalized >= 0.4) {return 'bg-yellow-500'}
    if (normalized >= 0.2) {return 'bg-orange-500'}
    return 'bg-red-500' // Low points
  }

  // Get transfer info for a player in a gameweek
  const getTransferInfo = (playerId: number, gameweek: number) => {
    const strategy = plan.transfer_strategy.find((s) => s.gameweek === gameweek)
    if (!strategy) {return null}

    const isTransferIn = strategy.transfers_in.includes(playerId)
    const isTransferOut = strategy.transfers_out.includes(playerId)

    return { isTransferIn, isTransferOut }
  }

  // Check if player is in starting XI for a gameweek
  const isInStartingXi = (playerId: number, gameweek: number): boolean => {
    const startingXi = plan.starting_xis[gameweek] || []
    return startingXi.includes(playerId)
  }

  // Check if player is in squad for a gameweek
  const isInSquad = (playerId: number, gameweek: number): boolean => {
    const squad = plan.squads[gameweek] || []
    return squad.includes(playerId)
  }

  // Get position color
  const getPositionColor = (position: string): string => {
    switch (position) {
      case 'GK':
        return 'border-blue-500'
      case 'DEF':
        return 'border-green-500'
      case 'MID':
        return 'border-yellow-500'
      case 'FWD':
        return 'border-red-500'
      default:
        return 'border-gray-500'
    }
  }

  // Flatten players by position order
  const orderedPlayers = [
    ...playersByPosition.GK,
    ...playersByPosition.DEF,
    ...playersByPosition.MID,
    ...playersByPosition.FWD,
  ]

  return (
    <div className="w-full space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Multi-Week Planning Heatmap</h2>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-600 rounded" />
            <span className="text-gray-300">High xP</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-500 rounded" />
            <span className="text-gray-300">Medium xP</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded" />
            <span className="text-gray-300">Low xP</span>
          </div>
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4 overflow-x-auto">
        <div className="min-w-full">
          {/* Gameweek Headers */}
          <div className="grid grid-cols-[200px_repeat(var(--gw-count),minmax(120px,1fr))] gap-2 mb-2">
            <div className="font-semibold text-gray-300 text-sm">Player</div>
            {Array.from({ length: plan.horizon_weeks }, (_, i) => i + 1).map((gw) => {
              const strategy = plan.transfer_strategy.find((s) => s.gameweek === gw)
              return (
                <div
                  key={gw}
                  className="text-center font-semibold text-white text-sm border-b border-fpl-dark-800 pb-2"
                >
                  <div>GW {gw}</div>
                  {strategy && strategy.transfer_count > 0 && (
                    <div className="text-xs text-fpl-green-400 mt-1">
                      {strategy.transfer_count} transfer{strategy.transfer_count !== 1 ? 's' : ''}
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Player Rows */}
          <div className="space-y-1">
            {orderedPlayers.map((playerId) => {
              const player = playerMap.get(playerId)
              if (!player) {return null}

              return (
                <div
                  key={playerId}
                  className="grid grid-cols-[200px_repeat(var(--gw-count),minmax(120px,1fr))] gap-2 items-center hover:bg-fpl-dark-800/50 transition-colors rounded px-2 py-1"
                  style={{ '--gw-count': plan.horizon_weeks } as React.CSSProperties}
                >
                  {/* Player Info */}
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full ${getPositionColor(player.position).replace('border-', 'bg-')}`}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-semibold text-white truncate" title={player.name}>
                        {player.name}
                      </div>
                      <div className="text-xs text-gray-400">{player.position}</div>
                    </div>
                  </div>

                  {/* Gameweek Cells */}
                  {Array.from({ length: plan.horizon_weeks }, (_, i) => i + 1).map((gw) => {
                    const inSquad = isInSquad(playerId, gw)
                    const inStartingXi = isInStartingXi(playerId, gw)
                    const transferInfo = getTransferInfo(playerId, gw)
                    const expectedPoints = getExpectedPoints(playerId, gw)

                    if (!inSquad) {
                      return (
                        <div
                          key={gw}
                          className="h-16 bg-fpl-dark-950 border border-fpl-dark-800 rounded flex items-center justify-center"
                        >
                          <span className="text-xs text-gray-600">-</span>
                        </div>
                      )
                    }

                    return (
                      <div
                        key={gw}
                        className={`h-16 ${getColorIntensity(expectedPoints)} border-2 rounded flex flex-col items-center justify-center relative ${
                          inStartingXi ? 'border-white' : 'border-transparent'
                        }`}
                        title={`${player.name} - GW${gw}: ${expectedPoints.toFixed(1)} xP`}
                      >
                        {/* Transfer Indicators */}
                        {transferInfo && (
                          <div className="absolute top-1 right-1 flex gap-1">
                            {transferInfo.isTransferIn && (
                              <div className="w-3 h-3 bg-blue-500 rounded-full border border-white" title="Transfer In" />
                            )}
                            {transferInfo.isTransferOut && (
                              <div className="w-3 h-3 bg-red-500 rounded-full border border-white" title="Transfer Out" />
                            )}
                          </div>
                        )}

                        {/* Starting XI Indicator */}
                        {inStartingXi && (
                          <div className="absolute top-1 left-1">
                            <div className="w-2 h-2 bg-white rounded-full" title="Starting XI" />
                          </div>
                        )}

                        {/* Expected Points */}
                        <div className="text-xs font-bold text-white">
                          {expectedPoints.toFixed(1)}
                        </div>

                        {/* Position Badge */}
                        <div className="text-xs text-white/80 mt-0.5">{player.position}</div>
                      </div>
                    )
                  })}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-white mb-3">Legend</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-600 rounded border-2 border-white" />
            <span className="text-gray-300">Starting XI (High xP)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-500 rounded" />
            <span className="text-gray-300">Bench (Medium xP)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full border border-white" />
            <span className="text-gray-300">Transfer In</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full border border-white" />
            <span className="text-gray-300">Transfer Out</span>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
          <div className="text-sm text-gray-400">Total Expected Points</div>
          <div className="text-2xl font-bold text-fpl-green-400">
            {plan.total_expected_points.toFixed(1)}
          </div>
        </div>
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
          <div className="text-sm text-gray-400">Transfer Cost</div>
          <div className="text-2xl font-bold text-yellow-400">
            -{plan.total_transfer_cost.toFixed(1)}
          </div>
        </div>
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
          <div className="text-sm text-gray-400">Net Expected Points</div>
          <div className="text-2xl font-bold text-white">
            {plan.net_expected_points.toFixed(1)}
          </div>
        </div>
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
          <div className="text-sm text-gray-400">Planning Horizon</div>
          <div className="text-2xl font-bold text-white">
            {plan.horizon_weeks} weeks
          </div>
        </div>
      </div>
    </div>
  )
}
