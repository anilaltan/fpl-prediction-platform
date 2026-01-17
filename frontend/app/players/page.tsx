'use client'

import { useState, useEffect, useMemo } from 'react'
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
  xmins?: number
  xcs?: number
  defcon_score?: number
}

type SortField = 'expected_points' | 'price' | 'form' | 'name' | 'ownership_percent' | null
type SortDirection = 'asc' | 'desc'

export default function PlayersPage() {
  const [players, setPlayers] = useState<Player[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [positionFilter, setPositionFilter] = useState<string>('ALL')
  const [teamFilter, setTeamFilter] = useState<string>('ALL')
  const [maxPriceFilter, setMaxPriceFilter] = useState<number>(16.0)
  const [sortField, setSortField] = useState<SortField>('expected_points')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchPlayers()
  }, [])

  // Get unique teams for filter dropdown
  const uniqueTeams = useMemo(() => {
    const teams = Array.from(new Set(players.map((p: Player) => p.team))).sort()
    return teams
  }, [players])

  // Filter and sort players
  const filteredAndSortedPlayers = useMemo(() => {
    let filtered = [...players]

    // Search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase()
      filtered = filtered.filter(p =>
        p.name.toLowerCase().includes(term) ||
        p.team.toLowerCase().includes(term)
      )
    }

    // Position filter
    if (positionFilter !== 'ALL') {
      filtered = filtered.filter(p => p.position === positionFilter)
    }

    // Team filter
    if (teamFilter !== 'ALL') {
      filtered = filtered.filter(p => p.team === teamFilter)
    }

    // Max price filter
    filtered = filtered.filter(p => p.price <= maxPriceFilter)

    // Sorting
    if (sortField) {
      filtered.sort((a, b) => {
        let aValue: number | string = a[sortField as keyof Player] as number | string
        let bValue: number | string = b[sortField as keyof Player] as number | string

        // Handle optional fields
        if (aValue === undefined || aValue === null) aValue = 0
        if (bValue === undefined || bValue === null) bValue = 0

        if (typeof aValue === 'string' && typeof bValue === 'string') {
          return sortDirection === 'asc'
            ? aValue.localeCompare(bValue)
            : bValue.localeCompare(aValue)
        }

        const numA = Number(aValue)
        const numB = Number(bValue)

        return sortDirection === 'asc' ? numA - numB : numB - numA
      })
    }

    return filtered
  }, [players, searchTerm, positionFilter, teamFilter, maxPriceFilter, sortField, sortDirection])

  const fetchPlayers = async () => {
    try {
      setLoading(true)
      // Backend automatically uses current gameweek from FPL API
      // Cache busting: Add timestamp to prevent browser caching
      const timestamp = new Date().getTime()
      const response = await axios.get(`/api/players/all?t=${timestamp}`)
      setPlayers(response.data)
    } catch (error) {
      console.error('Error fetching players:', error)
      // Show error state
      setPlayers([])
    } finally {
      setLoading(false)
    }
  }

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      // Toggle direction if same field
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      // New field, default to descending
      setSortField(field)
      setSortDirection('desc')
    }
  }

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return (
        <span className="text-gray-500 ml-1">↕</span>
      )
    }
    return sortDirection === 'asc' ? (
      <span className="text-fpl-green-500 ml-1">↑</span>
    ) : (
      <span className="text-fpl-green-500 ml-1">↓</span>
    )
  }

  const getXPColor = (xp: number) => {
    if (xp >= 6) return 'text-fpl-green-500 font-bold'
    if (xp >= 4) return 'text-fpl-green-400 font-semibold'
    if (xp >= 2) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getPositionColor = (position: string) => {
    switch (position) {
      case 'GK': return 'bg-blue-500/20 text-blue-400'
      case 'DEF': return 'bg-green-500/20 text-green-400'
      case 'MID': return 'bg-yellow-500/20 text-yellow-400'
      case 'FWD': return 'bg-purple-500/20 text-purple-400'
      default: return 'bg-gray-500/20 text-gray-400'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-16 w-16 border-4 border-fpl-dark-800 border-t-fpl-green-500 mx-auto"></div>
            <div className="absolute top-0 left-1/2 transform -translate-x-1/2 animate-spin rounded-full h-16 w-16 border-4 border-transparent border-r-fpl-purple-500" style={{ animationDuration: '1.5s' }}></div>
          </div>
          <p className="mt-6 text-gray-400 text-lg">Loading players...</p>
          <p className="mt-2 text-gray-500 text-sm">Fetching latest predictions...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Players</h1>
        <p className="text-gray-400">ML-powered expected points predictions</p>
      </div>

      {/* Advanced Filters */}
      <div className="mb-6 space-y-4">
        {/* Search Bar */}
        <div>
          <input
            type="text"
            placeholder="Search by name or team..."
            value={searchTerm}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-fpl-green-500"
          />
        </div>

        {/* Filter Row */}
        <div className="flex flex-wrap gap-4 items-center">
          {/* Position Filter - Button Group */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400 mr-2">Position:</span>
            <div className="flex gap-2">
              {['ALL', 'GK', 'DEF', 'MID', 'FWD'].map((pos) => (
                <button
                  key={pos}
                  onClick={() => setPositionFilter(pos)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                    positionFilter === pos
                      ? 'bg-fpl-green-500 text-white'
                      : 'bg-fpl-dark-800 text-gray-300 hover:bg-fpl-dark-700'
                  }`}
                >
                  {pos === 'ALL' ? 'All' : pos}
                </button>
              ))}
            </div>
          </div>

          {/* Team Filter - Dropdown */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Team:</span>
            <select
              value={teamFilter}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setTeamFilter(e.target.value)}
              className="px-3 py-1.5 bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg text-white text-sm focus:outline-none focus:border-fpl-green-500"
            >
              <option value="ALL">All Teams</option>
              {uniqueTeams.map((team) => (
                <option key={team} value={team}>
                  {team}
                </option>
              ))}
            </select>
          </div>

          {/* Max Price Filter */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Max Price:</span>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="4.0"
                max="16.0"
                step="0.5"
                value={maxPriceFilter}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setMaxPriceFilter(parseFloat(e.target.value))}
                className="w-32"
              />
              <span className="text-sm text-white font-medium min-w-[3rem]">
                £{maxPriceFilter.toFixed(1)}M
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Professional Data Table */}
      <div className="bg-fpl-dark-900 rounded-lg border border-fpl-dark-800 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-fpl-dark-800">
              <tr>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('name')}
                >
                  <div className="flex items-center">
                    Player {getSortIcon('name')}
                  </div>
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Position
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('price')}
                >
                  <div className="flex items-center">
                    Price {getSortIcon('price')}
                  </div>
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('expected_points')}
                >
                  <div className="flex items-center">
                    xP {getSortIcon('expected_points')}
                  </div>
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  xG
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  xA
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('form')}
                >
                  <div className="flex items-center">
                    Form {getSortIcon('form')}
                  </div>
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('ownership_percent')}
                >
                  <div className="flex items-center">
                    Ownership % {getSortIcon('ownership_percent')}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-fpl-dark-800">
              {filteredAndSortedPlayers.map((player) => (
                <tr key={player.id} className="hover:bg-fpl-dark-800 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div>
                        <div className="text-sm font-medium text-white">{player.name}</div>
                        <div className="text-xs text-gray-400 mt-0.5">{player.team}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs font-semibold rounded ${getPositionColor(player.position)}`}>
                      {player.position}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300 font-medium">
                    £{player.price.toFixed(1)}M
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-lg font-bold ${getXPColor(player.expected_points)}`}>
                      {player.expected_points.toFixed(2)}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                    {player.xg?.toFixed(2) || '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                    {player.xa?.toFixed(2) || '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    {player.form.toFixed(1)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    {player.ownership_percent.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {filteredAndSortedPlayers.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            No players found matching your criteria.
          </div>
        )}
      </div>

      <div className="mt-4 text-sm text-gray-400">
        Showing {filteredAndSortedPlayers.length} of {players.length} players
      </div>
    </div>
  )
}
