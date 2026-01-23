'use client'

import { useMemo } from 'react'
import type { TeamOptimizationResponse, PlayerOptimizationData } from '@/lib/types/api'

interface PitchViewProps {
  optimization: TeamOptimizationResponse
  players: PlayerOptimizationData[] // Map of player ID to player data
  gameweek?: number // Which gameweek to display (default: 1)
  showBench?: boolean // Whether to show bench players
}

/**
 * PitchView Component
 * Visualizes a 15-player FPL squad on a football pitch layout
 * Shows starting XI on the pitch and bench players below
 */
// eslint-disable-next-line max-lines-per-function
export function PitchView({
  optimization,
  players,
  gameweek = 1,
  showBench = true,
}: PitchViewProps) {
  // Get squad and starting XI for the specified gameweek
  const squadIds = useMemo(
    () => optimization.squads[gameweek] || [],
    [optimization.squads, gameweek],
  )
  const startingXiIds = useMemo(
    () => optimization.starting_xis[gameweek] || [],
    [optimization.starting_xis, gameweek],
  )

  // Create player map for quick lookup
  const playerMap = useMemo(() => {
    const map = new Map<number, PlayerOptimizationData>()
    players.forEach((p) => map.set(p.id, p))
    return map
  }, [players])

  // Organize players by position
  const organized = useMemo(() => {
    const byPosition: Record<string, Array<{ id: number; player: PlayerOptimizationData }>> = {
      GK: [],
      DEF: [],
      MID: [],
      FWD: [],
    }

    // Add starting XI players
    startingXiIds.forEach((id) => {
      const player = playerMap.get(id)
      if (player) {
        byPosition[player.position]?.push({ id, player })
      }
    })

    // Add bench players
    const benchIds = squadIds.filter((id) => !startingXiIds.includes(id))
    benchIds.forEach((id) => {
      const player = playerMap.get(id)
      if (player) {
        byPosition[player.position]?.push({ id, player })
      }
    })

    return byPosition
  }, [squadIds, startingXiIds, playerMap])

  // Get position color
  const getPositionColor = (position: string) => {
    switch (position) {
      case 'GK':
        return 'bg-blue-600/30 border-blue-500/50'
      case 'DEF':
        return 'bg-green-600/30 border-green-500/50'
      case 'MID':
        return 'bg-yellow-600/30 border-yellow-500/50'
      case 'FWD':
        return 'bg-red-600/30 border-red-500/50'
      default:
        return 'bg-gray-600/30 border-gray-500/50'
    }
  }

  // Get starting XI organized by position
  const startingXi = {
    GK: organized.GK.filter((p) => startingXiIds.includes(p.id)),
    DEF: organized.DEF.filter((p) => startingXiIds.includes(p.id)),
    MID: organized.MID.filter((p) => startingXiIds.includes(p.id)),
    FWD: organized.FWD.filter((p) => startingXiIds.includes(p.id)),
  }

  // Get bench players
  const benchPlayers = squadIds
    .filter((id) => !startingXiIds.includes(id))
    .map((id) => {
      const player = playerMap.get(id)
      return player ? { id, player } : null
    })
    .filter((p): p is { id: number; player: PlayerOptimizationData } => p !== null)

  // Determine formation (e.g., "4-4-2", "3-5-2")
  const formation = `${startingXi.DEF.length}-${startingXi.MID.length}-${startingXi.FWD.length}`

  return (
    <div className="w-full space-y-6">
      {/* Pitch Visualization */}
      <div className="bg-gradient-to-b from-green-900/20 to-green-800/10 border-2 border-green-700/30 rounded-lg p-4 md:p-8 overflow-x-auto">
        {/* Formation Display */}
        <div className="text-center mb-4">
          <span className="text-sm font-semibold text-gray-300 bg-fpl-dark-900 px-3 py-1 rounded">
            Formation: {formation}
          </span>
          <span className="ml-4 text-sm text-gray-400">
            Total xP: <span className="text-fpl-green-400 font-bold">{optimization.total_points.toFixed(2)}</span>
          </span>
        </div>

        {/* Pitch Layout */}
        <div className="relative min-h-[500px] md:min-h-[600px] flex flex-col justify-between h-full py-4">
          {/* Forwards Row */}
          <div className="flex justify-center items-center gap-2 flex-wrap">
            {startingXi.FWD.map(({ id, player }) => (
              <PlayerCard
                key={id}
                player={player}
                gameweek={gameweek}
                getPositionColor={getPositionColor}
              />
            ))}
          </div>

          {/* Midfielders Row */}
          <div className="flex justify-center items-center gap-2 flex-wrap">
            {startingXi.MID.map(({ id, player }) => (
              <PlayerCard
                key={id}
                player={player}
                gameweek={gameweek}
                getPositionColor={getPositionColor}
              />
            ))}
          </div>

          {/* Defenders Row */}
          <div className="flex justify-center items-center gap-2 flex-wrap">
            {startingXi.DEF.map(({ id, player }) => (
              <PlayerCard
                key={id}
                player={player}
                gameweek={gameweek}
                getPositionColor={getPositionColor}
              />
            ))}
          </div>

          {/* Goalkeeper Row */}
          <div className="flex justify-center items-center gap-2">
            {startingXi.GK[0] && (
              <PlayerCard
                player={startingXi.GK[0].player}
                gameweek={gameweek}
                getPositionColor={getPositionColor}
              />
            )}
          </div>
        </div>
      </div>

      {/* Bench */}
      {showBench && benchPlayers.length > 0 && (
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-6">
          <h2 className="text-xl font-bold text-white mb-4">Bench ({benchPlayers.length})</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {benchPlayers.map(({ id, player }) => (
              <div
                key={id}
                className={`p-4 rounded-lg border ${getPositionColor(player.position)}`}
              >
                <div className="text-sm font-semibold text-white">{player.name}</div>
                <div className="text-xs text-gray-300 mt-1">{player.position}</div>
                <div className="text-xs text-fpl-green-400 mt-1 font-bold">
                  xP: {getExpectedPoints(player, gameweek).toFixed(2)}
                </div>
                <div className="text-xs text-gray-400 mt-1">£{player.price.toFixed(1)}M</div>
                <div className="text-xs text-gray-500 mt-1">{player.team_name}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Squad Summary */}
      <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="text-gray-400">Squad Size</div>
            <div className="text-white font-bold">{squadIds.length}/15</div>
          </div>
          <div>
            <div className="text-gray-400">Starting XI</div>
            <div className="text-white font-bold">{startingXiIds.length}/11</div>
          </div>
          <div>
            <div className="text-gray-400">Budget Used</div>
            <div className="text-white font-bold">
              £{optimization.budget_used[gameweek]?.toFixed(1) || '0.0'}M
            </div>
          </div>
          <div>
            <div className="text-gray-400">Status</div>
            <div className={`font-bold ${optimization.optimal ? 'text-fpl-green-400' : 'text-yellow-400'}`}>
              {optimization.optimal ? 'Optimal' : optimization.status}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Player Card Component
function PlayerCard({
  player,
  gameweek,
  getPositionColor,
}: {
  player: PlayerOptimizationData
  gameweek: number
  getPositionColor: (pos: string) => string
}) {
  // Get short name
  const getShortName = (name: string) => {
    const parts = name.split(' ')
    if (parts.length > 1) {
      return `${parts[0]} ${parts[parts.length - 1][0]}.`
    }
    return name.length > 12 ? `${name.substring(0, 10)  }..` : name
  }

  // Get team icon/initial
  const getTeamIcon = (team: string) => {
    return team.substring(0, 3).toUpperCase()
  }

  // Get expected points for the gameweek
  const expectedPoints = getExpectedPoints(player, gameweek)

  return (
    <div
      className={`p-2 rounded-lg border-2 w-20 md:w-24 h-28 md:h-32 flex-shrink-0 flex flex-col justify-between ${getPositionColor(player.position)}`}
    >
      <div className="flex flex-col">
        {/* Team Icon/Initial */}
        <div className="text-xs font-bold text-white mb-0.5 bg-black/20 px-1 rounded text-center">
          {getTeamIcon(player.team_name)}
        </div>

        {/* Player Name (Short) */}
        <div className="text-xs font-semibold text-white truncate" title={player.name}>
          {getShortName(player.name)}
        </div>
      </div>

      <div className="flex flex-col mt-auto">
        {/* xP (Bold and Prominent) */}
        <div className="text-xs font-bold text-fpl-green-400">
          {expectedPoints.toFixed(1)}
        </div>

        {/* Price */}
        <div className="text-xs text-gray-400 mt-0.5">
          £{player.price.toFixed(1)}M
        </div>
      </div>
    </div>
  )
}

// Helper function to get expected points for a gameweek
function getExpectedPoints(player: PlayerOptimizationData, gameweek: number): number {
  if (gameweek === 1) {return player.expected_points_gw1}
  if (gameweek === 2) {return player.expected_points_gw2 ?? player.expected_points_gw1}
  if (gameweek === 3) {return player.expected_points_gw3 ?? player.expected_points_gw1}
  if (gameweek === 4) {return player.expected_points_gw4 ?? player.expected_points_gw1}
  if (gameweek === 5) {return player.expected_points_gw5 ?? player.expected_points_gw1}
  return player.expected_points_gw1
}
