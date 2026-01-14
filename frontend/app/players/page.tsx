"use client"

import { useState, useEffect } from 'react'
import axios from 'axios'

interface Player {
  id: number
  fpl_id: number
  name: string
  position: string
  team: string
  price: number
  expected_points: number
  ownership_percent: number
  form: number
  xg?: number
  xa?: number
  xcs?: number
  defcon_score?: number
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function AllPlayersPage() {
  const [players, setPlayers] = useState<Player[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState({ position: 'ALL', sortBy: 'expected_points' })

  useEffect(() => {
    fetchPlayers()
  }, [])

  const fetchPlayers = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`${API_URL}/api/players/all`)
      setPlayers(response.data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch players')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const filteredPlayers = players
    .filter(p => filter.position === 'ALL' || p.position === filter.position)
    .sort((a, b) => {
      if (filter.sortBy === 'expected_points') {
        return b.expected_points - a.expected_points
      } else if (filter.sortBy === 'price') {
        return b.price - a.price
      } else if (filter.sortBy === 'ownership') {
        return b.ownership_percent - a.ownership_percent
      }
      return 0
    })

  const positionColors = {
    GK: 'bg-blue-500/20 border-blue-500/50',
    DEF: 'bg-green-500/20 border-green-500/50',
    MID: 'bg-yellow-500/20 border-yellow-500/50',
    FWD: 'bg-red-500/20 border-red-500/50'
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading players...</div>
      </div>
    )
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="container mx-auto">
        <h1 className="text-4xl font-bold text-white mb-8">All Players</h1>

        {/* Filters */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-4 mb-6 border border-white/20 flex gap-4">
          <select
            value={filter.position}
            onChange={(e) => setFilter({ ...filter, position: e.target.value })}
            className="bg-white/10 text-white border border-white/20 rounded px-4 py-2"
          >
            <option value="ALL">All Positions</option>
            <option value="GK">Goalkeepers</option>
            <option value="DEF">Defenders</option>
            <option value="MID">Midfielders</option>
            <option value="FWD">Forwards</option>
          </select>

          <select
            value={filter.sortBy}
            onChange={(e) => setFilter({ ...filter, sortBy: e.target.value })}
            className="bg-white/10 text-white border border-white/20 rounded px-4 py-2"
          >
            <option value="expected_points">Expected Points</option>
            <option value="price">Price</option>
            <option value="ownership">Ownership %</option>
          </select>
        </div>

        {error && (
          <div className="bg-red-500/20 border border-red-500/50 text-red-300 p-4 rounded mb-6">
            {error}
          </div>
        )}

        {/* Players Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredPlayers.map((player) => (
            <div
              key={player.id}
              className={`bg-white/10 backdrop-blur-lg rounded-lg p-6 border ${positionColors[player.position as keyof typeof positionColors]}`}
            >
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-xl font-semibold text-white">{player.name}</h3>
                  <p className="text-gray-300 text-sm">{player.position} • {player.team}</p>
                </div>
                <span className="text-lg font-bold text-white">£{player.price.toFixed(1)}M</span>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-300">Expected Points:</span>
                  <span className="text-white font-semibold">{player.expected_points.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Ownership:</span>
                  <span className="text-white">{player.ownership_percent.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Form:</span>
                  <span className="text-white">{player.form.toFixed(1)}</span>
                </div>
                {player.xg !== undefined && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">xG:</span>
                    <span className="text-gray-300">{player.xg.toFixed(2)}</span>
                  </div>
                )}
                {player.xa !== undefined && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">xA:</span>
                    <span className="text-gray-300">{player.xa.toFixed(2)}</span>
                  </div>
                )}
                {player.defcon_score !== undefined && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">DefCon:</span>
                    <span className="text-gray-300">{player.defcon_score.toFixed(1)}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  )
}