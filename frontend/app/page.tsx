'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import Link from 'next/link'

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
}

interface Fixture {
  id: number
  gameweek: number
  home_team_name: string
  away_team_name: string
  home_difficulty: number
  away_difficulty: number
  kickoff_time: string | null
  finished: boolean
}

interface GameweekInfo {
  id: number
  name: string
  deadline_time: string
  is_next: boolean
}

export default function DashboardPage() {
  const [players, setPlayers] = useState<Player[]>([])
  const [fixtures, setFixtures] = useState<Fixture[]>([])
  const [nextGameweek, setNextGameweek] = useState<GameweekInfo | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      setLoading(true)
      
      // Fetch players for next gameweek
      const playersResponse = await axios.get('/api/players/all?use_next_gameweek=true')
      setPlayers(playersResponse.data)
      
      // Fetch bootstrap data to get next gameweek info
      try {
        const bootstrapResponse = await axios.get('/api/fpl/bootstrap')
        const events = bootstrapResponse.data?.events || []
        
        // Find next gameweek (is_next === true)
        const nextGW = events.find((e: any) => e.is_next === true)
        if (nextGW) {
          setNextGameweek({
            id: nextGW.id,
            name: nextGW.name,
            deadline_time: nextGW.deadline_time,
            is_next: true,
          })
          
          // Fetch fixtures for next gameweek
          try {
            const fixturesResponse = await axios.get(`/api/fpl/fixtures?gameweek=${nextGW.id}&future_only=true`)
            if (fixturesResponse.data?.fixtures) {
              setFixtures(fixturesResponse.data.fixtures)
            }
          } catch (err) {
            console.error('Error fetching fixtures:', err)
          }
        }
      } catch (err) {
        console.error('Error fetching bootstrap data:', err)
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  // Calculate metrics
  const avgXP = players.length > 0
    ? players.reduce((sum, p) => sum + p.expected_points, 0) / players.length
    : 0

  // Best Captain: Highest xP player (sorted by xP descending)
  const bestCaptain = players.length > 0
    ? [...players].sort((a, b) => b.expected_points - a.expected_points)[0]
    : null

  // Surprise player: High xP but low ownership (relaxed filter)
  const surprisePlayer = players.length > 0
    ? [...players]
      .filter(p => p.ownership_percent < 15 && p.expected_points > 3)
      .sort((a, b) => {
        // Sort by xP/ownership ratio (higher is better)
        const ratioA = a.expected_points / Math.max(a.ownership_percent, 0.1)
        const ratioB = b.expected_points / Math.max(b.ownership_percent, 0.1)
        return ratioB - ratioA
      })[0] || null
    : null

  const getDifficultyColor = (difficulty: number) => {
    if (difficulty <= 2) {return 'text-fpl-green-500 bg-fpl-green-500/20'}
    if (difficulty === 3) {return 'text-yellow-500 bg-yellow-500/20'}
    return 'text-red-500 bg-red-500/20'
  }

  const formatDeadline = (deadlineTime: string) => {
    try {
      const date = new Date(deadlineTime)
      return date.toLocaleString('en-GB', {
        day: 'numeric',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
      })
    } catch {
      return deadlineTime
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-fpl-green-500 mx-auto" />
          <p className="mt-4 text-gray-400">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Dashboard</h1>
            <p className="text-gray-400">FPL Analytics Overview</p>
          </div>
          {nextGameweek && (
            <div className="text-right">
              <div className="text-2xl font-bold text-fpl-green-500">
                GW {nextGameweek.id}
              </div>
              <div className="text-sm text-gray-400 mt-1">
                Deadline: {formatDeadline(nextGameweek.deadline_time)}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        {/* Average Prediction Points */}
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-400">Average xP</h3>
            <svg className="w-6 h-6 text-fpl-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div className="text-3xl font-bold text-fpl-green-500 mb-2">
            {avgXP.toFixed(2)}
          </div>
          <p className="text-xs text-gray-400">
            Across all {players.length} players
          </p>
        </div>

        {/* Best Captain Suggestion */}
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-400">Best Captain</h3>
            <svg className="w-6 h-6 text-fpl-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
            </svg>
          </div>
          {bestCaptain ? (
            <>
              <div className="text-xl font-bold text-white mb-1">
                {bestCaptain.name}
              </div>
              <div className="text-2xl font-bold text-fpl-purple-500 mb-2">
                {bestCaptain.expected_points.toFixed(2)} <span className="text-sm text-gray-400">xP</span>
              </div>
              <p className="text-xs text-gray-400">
                {bestCaptain.position} • {bestCaptain.team}
              </p>
            </>
          ) : (
            <p className="text-gray-400">No data</p>
          )}
        </div>

        {/* Surprise Player */}
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-400">Surprise Player</h3>
            <svg className="w-6 h-6 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          {surprisePlayer ? (
            <>
              <div className="text-xl font-bold text-white mb-1">
                {surprisePlayer.name}
              </div>
              <div className="text-2xl font-bold text-yellow-500 mb-2">
                {surprisePlayer.expected_points.toFixed(2)} <span className="text-sm text-gray-400">xP</span>
              </div>
              <p className="text-xs text-gray-400">
                {surprisePlayer.ownership_percent.toFixed(1)}% owned • {surprisePlayer.team}
              </p>
            </>
          ) : (
            <p className="text-gray-400">No surprise players found</p>
          )}
        </div>
      </div>

      {/* Upcoming Fixtures Difficulty */}
      <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-6 mb-8">
        <h2 className="text-xl font-bold text-white mb-4">
          {nextGameweek ? `GW${nextGameweek.id} Fixtures` : 'Upcoming Fixture Difficulty (FDR)'}
        </h2>
        {fixtures.length > 0 ? (
          <div className="space-y-3">
            {fixtures.map((fixture) => (
              <div
                key={fixture.id}
                className="flex items-center justify-between p-4 bg-fpl-dark-800 rounded-lg"
              >
                <div className="flex items-center gap-4 flex-1">
                  <div className="flex items-center gap-2">
                    <div className="text-white font-medium">{fixture.home_team_name}</div>
                    <div className={`px-2 py-1 rounded text-xs font-semibold ${getDifficultyColor(fixture.home_difficulty)}`}>
                      FDR {fixture.home_difficulty}
                    </div>
                  </div>
                  <div className="text-gray-400">vs</div>
                  <div className="flex items-center gap-2">
                    <div className="text-gray-300 font-medium">{fixture.away_team_name}</div>
                    <div className={`px-2 py-1 rounded text-xs font-semibold ${getDifficultyColor(fixture.away_difficulty)}`}>
                      FDR {fixture.away_difficulty}
                    </div>
                  </div>
                  {fixture.kickoff_time && (
                    <div className="text-xs text-gray-500 ml-auto">
                      {new Date(fixture.kickoff_time).toLocaleDateString('en-GB', {
                        day: 'numeric',
                        month: 'short',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">
            {loading ? 'Loading fixtures...' : 'No upcoming fixtures found'}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-3 gap-6">
        <Link
          href="/players"
          className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-6 hover:border-fpl-green-500 transition-colors group"
        >
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-white group-hover:text-fpl-green-500 transition-colors">
              View All Players
            </h3>
            <svg className="w-5 h-5 text-gray-400 group-hover:text-fpl-green-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
          <p className="text-sm text-gray-400">
            Explore {players.length} players with ML predictions
          </p>
        </Link>

        <Link
          href="/dream-team"
          className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-6 hover:border-fpl-purple-500 transition-colors group"
        >
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-white group-hover:text-fpl-purple-500 transition-colors">
              Dream Team
            </h3>
            <svg className="w-5 h-5 text-gray-400 group-hover:text-fpl-purple-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
          <p className="text-sm text-gray-400">
            View optimal team selection
          </p>
        </Link>

        <Link
          href="/solver"
          className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-6 hover:border-fpl-purple-500 transition-colors group"
        >
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-white group-hover:text-fpl-purple-500 transition-colors">
              Team Solver
            </h3>
            <svg className="w-5 h-5 text-gray-400 group-hover:text-fpl-purple-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
          <p className="text-sm text-gray-400">
            Optimize your team with ILP solver
          </p>
        </Link>
      </div>
    </div>
  )
}
