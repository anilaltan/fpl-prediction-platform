'use client'

import { useState, useMemo } from 'react'
import type { MarketIntelligencePlayer } from '@/lib/types/api'

interface MarketIntelligenceTableProps {
  players: MarketIntelligencePlayer[]
  gameweek?: number
  season?: string
}

type SortField =
  | 'arbitrage_score'
  | 'xp_rank'
  | 'ownership_rank'
  | 'xp'
  | 'ownership'
  | 'price'
  | 'name'
  | 'category'
  | null

type SortDirection = 'asc' | 'desc'

/**
 * MarketIntelligenceTable Component
 * Displays ownership arbitrage analysis in a sortable table
 * Highlights differentials and overvalued players
 */
// eslint-disable-next-line max-lines-per-function
export function MarketIntelligenceTable({
  players,
  gameweek,
  season: _season,
}: MarketIntelligenceTableProps) {
  const [sortField, setSortField] = useState<SortField>('arbitrage_score')
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc')
  const [searchTerm, setSearchTerm] = useState('')
  const [categoryFilter, setCategoryFilter] = useState<string>('ALL')
  const [positionFilter, setPositionFilter] = useState<string>('ALL')

  // Get unique positions and categories for filters
  const uniquePositions = useMemo(() => {
    return Array.from(new Set(players.map((p) => p.position))).sort()
  }, [players])

  // Filter and sort players
  const filteredAndSortedPlayers = useMemo(() => {
    let filtered = [...players]

    // Search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase()
      filtered = filtered.filter(
        (p) =>
          p.name.toLowerCase().includes(term) ||
          p.team.toLowerCase().includes(term),
      )
    }

    // Category filter
    if (categoryFilter !== 'ALL') {
      filtered = filtered.filter((p) => p.category === categoryFilter)
    }

    // Position filter
    if (positionFilter !== 'ALL') {
      filtered = filtered.filter((p) => p.position === positionFilter)
    }

    // Sorting
    if (sortField) {
      filtered.sort((a, b) => {
        let aValue: number | string = a[sortField as keyof MarketIntelligencePlayer] as
          | number
          | string
        let bValue: number | string = b[sortField as keyof MarketIntelligencePlayer] as
          | number
          | string

        if (aValue === undefined || aValue === null) {
          aValue = 0
        }
        if (bValue === undefined || bValue === null) {
          bValue = 0
        }

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
  }, [players, searchTerm, categoryFilter, positionFilter, sortField, sortDirection])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('desc')
    }
  }

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return (
        <span className="text-gray-500 ml-1">
          <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
          </svg>
        </span>
      )
    }
    return (
      <span className="text-fpl-green-400 ml-1">
        {sortDirection === 'asc' ? '↑' : '↓'}
      </span>
    )
  }

  // Get category badge color
  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Differential':
        return 'bg-green-600/30 border-green-500/50 text-green-300'
      case 'Overvalued':
        return 'bg-red-600/30 border-red-500/50 text-red-300'
      case 'Neutral':
        return 'bg-gray-600/30 border-gray-500/50 text-gray-300'
      default:
        return 'bg-gray-600/30 border-gray-500/50 text-gray-300'
    }
  }

  // Get position color
  const getPositionColor = (position: string) => {
    switch (position) {
      case 'GK':
        return 'bg-blue-600/30 border-blue-500/50 text-blue-300'
      case 'DEF':
        return 'bg-green-600/30 border-green-500/50 text-green-300'
      case 'MID':
        return 'bg-yellow-600/30 border-yellow-500/50 text-yellow-300'
      case 'FWD':
        return 'bg-red-600/30 border-red-500/50 text-red-300'
      default:
        return 'bg-gray-600/30 border-gray-500/50 text-gray-300'
    }
  }

  // Get arbitrage score color
  const getArbitrageScoreColor = (score: number) => {
    if (score < -20) {
      return 'text-green-400 font-bold'
    } // Strong differential
    if (score < -10) {
      return 'text-green-300'
    } // Moderate differential
    if (score > 20) {
      return 'text-red-400 font-bold'
    } // Strong overvalued
    if (score > 10) {
      return 'text-red-300'
    } // Moderate overvalued
    return 'text-gray-300' // Neutral
  }

  // Count by category
  const categoryCounts = useMemo(() => {
    return {
      Differential: players.filter((p) => p.category === 'Differential').length,
      Overvalued: players.filter((p) => p.category === 'Overvalued').length,
      Neutral: players.filter((p) => p.category === 'Neutral').length,
    }
  }, [players])

  return (
    <div className="w-full space-y-4">
      {/* Header with filters */}
      <div className="flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Market Intelligence</h2>
          {gameweek && (
            <p className="text-sm text-gray-400 mt-1">Gameweek {gameweek}</p>
          )}
        </div>

        {/* Category Stats */}
        <div className="flex gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-600 rounded" />
            <span className="text-gray-300">
              Differentials: <span className="text-green-400 font-semibold">{categoryCounts.Differential}</span>
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-600 rounded" />
            <span className="text-gray-300">
              Overvalued: <span className="text-red-400 font-semibold">{categoryCounts.Overvalued}</span>
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-gray-600 rounded" />
            <span className="text-gray-300">
              Neutral: <span className="text-gray-400 font-semibold">{categoryCounts.Neutral}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Search */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Search</label>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Player or team..."
              className="w-full bg-fpl-dark-800 border border-fpl-dark-700 text-white rounded px-3 py-2 text-sm focus:outline-none focus:border-fpl-green-500"
            />
          </div>

          {/* Category Filter */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Category</label>
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="w-full bg-fpl-dark-800 border border-fpl-dark-700 text-white rounded px-3 py-2 text-sm focus:outline-none focus:border-fpl-green-500"
            >
              <option value="ALL">All Categories</option>
              <option value="Differential">Differentials</option>
              <option value="Overvalued">Overvalued</option>
              <option value="Neutral">Neutral</option>
            </select>
          </div>

          {/* Position Filter */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Position</label>
            <select
              value={positionFilter}
              onChange={(e) => setPositionFilter(e.target.value)}
              className="w-full bg-fpl-dark-800 border border-fpl-dark-700 text-white rounded px-3 py-2 text-sm focus:outline-none focus:border-fpl-green-500"
            >
              <option value="ALL">All Positions</option>
              {uniquePositions.map((pos) => (
                <option key={pos} value={pos}>
                  {pos}
                </option>
              ))}
            </select>
          </div>

          {/* Results Count */}
          <div className="flex items-end">
            <div className="text-sm text-gray-400">
              Showing <span className="text-white font-semibold">{filteredAndSortedPlayers.length}</span> of{' '}
              <span className="text-white font-semibold">{players.length}</span> players
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-fpl-dark-800">
              <tr>
                <th
                  className="px-6 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('name')}
                >
                  <div className="flex items-center">
                    Player {getSortIcon('name')}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('position')}
                >
                  Pos
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('price')}
                >
                  <div className="flex items-center">
                    Price {getSortIcon('price')}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('xp')}
                >
                  <div className="flex items-center">
                    xP {getSortIcon('xp')}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('xp_rank')}
                >
                  <div className="flex items-center">
                    xP Rank {getSortIcon('xp_rank')}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('ownership')}
                >
                  <div className="flex items-center">
                    Ownership % {getSortIcon('ownership')}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('ownership_rank')}
                >
                  <div className="flex items-center">
                    Own Rank {getSortIcon('ownership_rank')}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('arbitrage_score')}
                >
                  <div className="flex items-center">
                    Arbitrage Score {getSortIcon('arbitrage_score')}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-fpl-dark-700 transition-colors"
                  onClick={() => handleSort('category')}
                >
                  <div className="flex items-center">
                    Category {getSortIcon('category')}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-fpl-dark-800">
              {filteredAndSortedPlayers.map((player) => (
                <tr
                  key={player.player_id}
                  className="hover:bg-fpl-dark-800/50 transition-colors"
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div>
                        <div className="text-sm font-medium text-white">{player.name}</div>
                        <div className="text-xs text-gray-400 mt-0.5">{player.team}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 text-xs font-semibold rounded border ${getPositionColor(player.position)}`}
                    >
                      {player.position}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300 font-medium">
                    £{player.price.toFixed(1)}M
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-sm font-bold text-fpl-green-400">
                      {player.xp.toFixed(2)}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    #{player.xp_rank}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    {player.ownership.toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    #{player.ownership_rank}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`text-sm font-semibold ${getArbitrageScoreColor(player.arbitrage_score)}`}
                    >
                      {player.arbitrage_score > 0 ? '+' : ''}
                      {player.arbitrage_score.toFixed(1)}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 text-xs font-semibold rounded border ${getCategoryColor(player.category)}`}
                    >
                      {player.category}
                    </span>
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

      {/* Info Box */}
      <div className="bg-blue-900/20 border border-blue-500/50 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-blue-300 mb-2">About Arbitrage Score</h3>
        <p className="text-xs text-gray-300">
          <strong>Negative score (Differential):</strong> High xP rank but low ownership rank -
          undervalued by the market.
          <br />
          <strong>Positive score (Overvalued):</strong> Low xP rank but high ownership rank -
          overvalued by the market.
          <br />
          <strong>Formula:</strong> Arbitrage Score = xP Rank - Ownership Rank
        </p>
      </div>
    </div>
  )
}
