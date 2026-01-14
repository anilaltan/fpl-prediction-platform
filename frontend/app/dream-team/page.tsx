"use client"

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

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function DreamTeamPage() {
  const [dreamTeam, setDreamTeam] = useState<DreamTeam | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [gameweek, setGameweek] = useState(1)

  useEffect(() => {
    fetchDreamTeam()
  }, [gameweek])

  const fetchDreamTeam = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`${API_URL}/api/dream-team?gameweek=${gameweek}`)
      setDreamTeam(response.data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch dream team')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const positionColors = {
    GK: 'bg-blue-500/20 border-blue-500/50',
    DEF: 'bg-green-500/20 border-green-500/50',
    MID: 'bg-yellow-500/20 border-yellow-500/50',
    FWD: 'bg-red-500/20 border-red-500/50'
  }

  const groupByPosition = (players: DreamTeamPlayer[]) => {
    return {
      GK: players.filter(p => p.position === 'GK'),
      DEF: players.filter(p => p.position === 'DEF'),
      MID: players.filter(p => p.position === 'MID'),
      FWD: players.filter(p => p.position === 'FWD')
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading dream team...</div>
      </div>
    )
  }

  if (!dreamTeam) {
    return null
  }

  const startingByPos = groupByPosition(dreamTeam.starting_xi)
  const bench = dreamTeam.squad.filter(p => !dreamTeam.starting_xi.some(xi => xi.player_id === p.player_id))

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="container mx-auto max-w-6xl">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-4xl font-bold text-white">Dream Team</h1>
          <div className="flex items-center gap-4">
            <label className="text-white">Gameweek:</label>
            <input
              type="number"
              value={gameweek}
              onChange={(e) => setGameweek(Number(e.target.value))}
              min="1"
              max="38"
              className="bg-white/10 text-white border border-white/20 rounded px-4 py-2 w-20"
            />
          </div>
        </div>

        {error && (
          <div className="bg-red-500/20 border border-red-500/50 text-red-300 p-4 rounded mb-6">
            {error}
          </div>
        )}

        {/* Summary */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 mb-6 border border-white/20">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-gray-300 text-sm">Total Expected Points</p>
              <p className="text-3xl font-bold text-white">{dreamTeam.total_expected_points.toFixed(1)}</p>
            </div>
            <div>
              <p className="text-gray-300 text-sm">Total Cost</p>
              <p className="text-3xl font-bold text-white">£{dreamTeam.total_cost.toFixed(1)}M</p>
            </div>
            <div>
              <p className="text-gray-300 text-sm">Formation</p>
              <p className="text-3xl font-bold text-white">{dreamTeam.formation}</p>
            </div>
          </div>
        </div>

        {/* Starting XI */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 mb-6 border border-white/20">
          <h2 className="text-2xl font-semibold text-white mb-4">Starting XI</h2>
          
          {/* Formation Layout */}
          <div className="space-y-4">
            {/* Goalkeeper */}
            {startingByPos.GK.length > 0 && (
              <div className="flex justify-center">
                {startingByPos.GK.map(player => (
                  <div
                    key={player.player_id}
                    className={`${positionColors.GK} rounded-lg p-4 border min-w-[200px] text-center`}
                  >
                    <p className="text-white font-semibold">{player.name}</p>
                    <p className="text-gray-300 text-sm">{player.team}</p>
                    <p className="text-yellow-400 font-bold">{player.expected_points.toFixed(1)} xP</p>
                  </div>
                ))}
              </div>
            )}

            {/* Defenders */}
            {startingByPos.DEF.length > 0 && (
              <div className="flex justify-center gap-4">
                {startingByPos.DEF.map(player => (
                  <div
                    key={player.player_id}
                    className={`${positionColors.DEF} rounded-lg p-4 border min-w-[150px] text-center`}
                  >
                    <p className="text-white font-semibold">{player.name}</p>
                    <p className="text-gray-300 text-sm">{player.team}</p>
                    <p className="text-yellow-400 font-bold">{player.expected_points.toFixed(1)} xP</p>
                  </div>
                ))}
              </div>
            )}

            {/* Midfielders */}
            {startingByPos.MID.length > 0 && (
              <div className="flex justify-center gap-4">
                {startingByPos.MID.map(player => (
                  <div
                    key={player.player_id}
                    className={`${positionColors.MID} rounded-lg p-4 border min-w-[150px] text-center`}
                  >
                    <p className="text-white font-semibold">{player.name}</p>
                    <p className="text-gray-300 text-sm">{player.team}</p>
                    <p className="text-yellow-400 font-bold">{player.expected_points.toFixed(1)} xP</p>
                  </div>
                ))}
              </div>
            )}

            {/* Forwards */}
            {startingByPos.FWD.length > 0 && (
              <div className="flex justify-center gap-4">
                {startingByPos.FWD.map(player => (
                  <div
                    key={player.player_id}
                    className={`${positionColors.FWD} rounded-lg p-4 border min-w-[150px] text-center`}
                  >
                    <p className="text-white font-semibold">{player.name}</p>
                    <p className="text-gray-300 text-sm">{player.team}</p>
                    <p className="text-yellow-400 font-bold">{player.expected_points.toFixed(1)} xP</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Bench */}
        {bench.length > 0 && (
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
            <h2 className="text-2xl font-semibold text-white mb-4">Bench</h2>
            <div className="grid grid-cols-4 gap-4">
              {bench.map(player => (
                <div
                  key={player.player_id}
                  className={`${positionColors[player.position as keyof typeof positionColors]} rounded-lg p-4 border`}
                >
                  <p className="text-white font-semibold text-sm">{player.name}</p>
                  <p className="text-gray-300 text-xs">{player.position} • {player.team}</p>
                  <p className="text-yellow-400 font-bold text-sm">{player.expected_points.toFixed(1)} xP</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  )
}