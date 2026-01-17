'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'

interface DreamTeamPlayer {
  player_id: number
  name: string
  position: string
  team: string
  expected_points: number
  price: number
}

interface DreamTeam {
  gameweek: number
  squad: DreamTeamPlayer[]
  starting_xi: DreamTeamPlayer[]
  total_expected_points: number
  total_cost: number
  formation: string
}

export default function DreamTeamPage() {
  const [dreamTeam, setDreamTeam] = useState<DreamTeam | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDreamTeam()
  }, [])

  const fetchDreamTeam = async () => {
    try {
      setLoading(true)
      // Backend automatically uses current gameweek from FPL API
      const response = await axios.get('/api/dream-team')
      setDreamTeam(response.data)
    } catch (error) {
      console.error('Error fetching dream team:', error)
    } finally {
      setLoading(false)
    }
  }

  const getPositionColor = (position: string) => {
    switch (position) {
      case 'GK': return 'bg-blue-500/30 border-blue-500'
      case 'DEF': return 'bg-green-500/30 border-green-500'
      case 'MID': return 'bg-yellow-500/30 border-yellow-500'
      case 'FWD': return 'bg-purple-500/30 border-purple-500'
      default: return 'bg-gray-500/30 border-gray-500'
    }
  }

  const organizePlayersByPosition = (players: DreamTeamPlayer[]) => {
    const organized: { [key: string]: DreamTeamPlayer[] } = {
      GK: [],
      DEF: [],
      MID: [],
      FWD: []
    }

    players.forEach(player => {
      if (organized[player.position]) {
        organized[player.position].push(player)
      }
    })

    return organized
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-fpl-green-500 mx-auto"></div>
          <p className="mt-4 text-gray-400">Loading dream team...</p>
        </div>
      </div>
    )
  }

  if (!dreamTeam) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-400">
          <p>Failed to load dream team</p>
        </div>
      </div>
    )
  }

  const organized = organizePlayersByPosition(dreamTeam.starting_xi)
  const formation = dreamTeam.formation.split('-')
  const defCount = parseInt(formation[0]) || 4
  const midCount = parseInt(formation[1]) || 4
  const fwdCount = parseInt(formation[2]) || 2

  // Select captain (highest xP) and vice-captain (second highest)
  const sortedByXP = [...dreamTeam.starting_xi].sort((a, b) => b.expected_points - a.expected_points)
  const captain = sortedByXP[0]
  const viceCaptain = sortedByXP[1]

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Dream Team</h1>
        <p className="text-gray-400">Optimal team selection powered by ILP Solver</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
          <div className="text-sm text-gray-400">Total xP</div>
          <div className="text-2xl font-bold text-fpl-green-500 mt-1">
            {dreamTeam.total_expected_points.toFixed(2)}
          </div>
        </div>
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
          <div className="text-sm text-gray-400">Total Cost</div>
          <div className="text-2xl font-bold text-fpl-purple-500 mt-1">
            £{dreamTeam.total_cost.toFixed(1)}M
          </div>
        </div>
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
          <div className="text-sm text-gray-400">Formation</div>
          <div className="text-2xl font-bold text-white mt-1">
            {dreamTeam.formation}
          </div>
        </div>
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
          <div className="text-sm text-gray-400">Gameweek</div>
          <div className="text-2xl font-bold text-white mt-1">
            GW {dreamTeam.gameweek}
          </div>
        </div>
      </div>

      {/* Pitch Visualization */}
      <div className="bg-gradient-to-b from-green-900/20 to-green-800/10 border-2 border-green-700/30 rounded-lg p-4 md:p-8 mb-6 overflow-x-auto">
        <div className="relative min-h-[500px] md:min-h-[600px] flex flex-col justify-between h-full py-4">
          {/* Forwards Row */}
          <div className="flex justify-center items-center gap-2 flex-wrap">
            {organized.FWD.slice(0, fwdCount).map((player) => (
              <PlayerCard
                key={player.player_id}
                player={player}
                isCaptain={player.player_id === captain?.player_id}
                isViceCaptain={player.player_id === viceCaptain?.player_id}
                getPositionColor={getPositionColor}
              />
            ))}
          </div>

          {/* Midfielders Row */}
          <div className="flex justify-center items-center gap-2 flex-wrap">
            {organized.MID.slice(0, midCount).map((player) => (
              <PlayerCard
                key={player.player_id}
                player={player}
                isCaptain={player.player_id === captain?.player_id}
                isViceCaptain={player.player_id === viceCaptain?.player_id}
                getPositionColor={getPositionColor}
              />
            ))}
          </div>

          {/* Defenders Row */}
          <div className="flex justify-center items-center gap-2 flex-wrap">
            {organized.DEF.slice(0, defCount).map((player) => (
              <PlayerCard
                key={player.player_id}
                player={player}
                isCaptain={player.player_id === captain?.player_id}
                isViceCaptain={player.player_id === viceCaptain?.player_id}
                getPositionColor={getPositionColor}
              />
            ))}
          </div>

          {/* Goalkeeper Row */}
          <div className="flex justify-center items-center gap-2">
            {organized.GK[0] && (
              <PlayerCard
                player={organized.GK[0]}
                isCaptain={organized.GK[0].player_id === captain?.player_id}
                isViceCaptain={organized.GK[0].player_id === viceCaptain?.player_id}
                getPositionColor={getPositionColor}
              />
            )}
          </div>
        </div>
      </div>

      {/* Bench */}
      <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-6">
        <h2 className="text-xl font-bold text-white mb-4">Bench</h2>
        <div className="grid grid-cols-5 gap-4">
          {dreamTeam.squad
            .filter(p => !dreamTeam.starting_xi.some(xi => xi.player_id === p.player_id))
            .map((player) => (
              <div
                key={player.player_id}
                className={`p-4 rounded-lg border ${getPositionColor(player.position)}`}
              >
                <div className="text-sm font-semibold text-white">{player.name}</div>
                <div className="text-xs text-gray-300 mt-1">{player.position}</div>
                <div className="text-xs text-fpl-green-400 mt-1 font-bold">
                  xP: {player.expected_points.toFixed(2)}
                </div>
                <div className="text-xs text-gray-400 mt-1">£{player.price.toFixed(1)}M</div>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
}

function PlayerCard({
  player,
  isCaptain,
  isViceCaptain,
  getPositionColor,
}: {
  player: DreamTeamPlayer
  isCaptain: boolean
  isViceCaptain: boolean
  getPositionColor: (pos: string) => string
}) {
  // Get short name (first name + last name initial, or truncate if too long)
  const getShortName = (name: string) => {
    const parts = name.split(' ')
    if (parts.length > 1) {
      return `${parts[0]} ${parts[parts.length - 1][0]}.`
    }
    return name.length > 12 ? name.substring(0, 10) + '..' : name
  }

  // Get team icon/initial
  const getTeamIcon = (team: string) => {
    return team.substring(0, 3).toUpperCase()
  }

  return (
    <div
      className={`p-2 rounded-lg border-2 w-20 md:w-24 h-28 md:h-32 flex-shrink-0 flex flex-col justify-between ${getPositionColor(player.position)}`}
    >
      <div className="flex flex-col">
        {/* Captain/Vice Captain Badge - Always reserve space */}
        <div className="h-5 mb-0.5 flex items-start">
          {(isCaptain || isViceCaptain) && (
            <div className={`text-xs font-bold px-1.5 py-0.5 rounded ${
              isCaptain 
                ? 'bg-yellow-500 text-black' 
                : 'bg-blue-500 text-white'
            }`}>
              {isCaptain ? '(C)' : '(V)'}
            </div>
          )}
        </div>
        
        {/* Team Icon/Initial */}
        <div className="text-xs font-bold text-white mb-0.5 bg-black/20 px-1 rounded text-center">
          {getTeamIcon(player.team)}
        </div>
        
        {/* Player Name (Short) */}
        <div className="text-xs font-semibold text-white truncate" title={player.name}>
          {getShortName(player.name)}
        </div>
      </div>
      
      <div className="flex flex-col mt-auto">
        {/* xP (Bold and Prominent) */}
        <div className="text-xs font-bold text-fpl-green-400">
          {player.expected_points.toFixed(1)}
        </div>
        
        {/* Price */}
        <div className="text-xs text-gray-400 mt-0.5">
          £{player.price.toFixed(1)}M
        </div>
      </div>
    </div>
  )
}
