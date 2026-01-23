'use client'

import { useState } from 'react'
import { useTeamOptimize } from '@/lib/hooks'
import { PitchView } from '@/components'
import type { PlayerOptimizationData } from '@/lib/types/api'

export default function TeamOptimizerPage() {
  const [players, setPlayers] = useState<PlayerOptimizationData[]>([])
  const [selectedPlayers, setSelectedPlayers] = useState<PlayerOptimizationData[]>([])
  const { optimize, data, isLoading, error } = useTeamOptimize({
    onSuccess: (result) => {
      console.log('Optimization complete:', result)
    },
    onError: (err) => {
      console.error('Optimization failed:', err)
    },
  })

  const handleOptimize = async () => {
    if (selectedPlayers.length < 15) {
      alert('Please select at least 15 players')
      return
    }

    await optimize({
      players: selectedPlayers,
      budget: 100.0,
      horizon_weeks: 1,
      free_transfers: 1,
    })
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Team Optimizer</h1>
        <button
          onClick={handleOptimize}
          disabled={isLoading || selectedPlayers.length < 15}
          className="px-6 py-3 bg-fpl-green-500 hover:bg-fpl-green-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-colors"
        >
          {isLoading ? 'Optimizing...' : 'Optimize Team'}
        </button>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4">
          <p className="text-red-400">Error: {error.message}</p>
        </div>
      )}

      {data && (
        <div className="space-y-6">
          <PitchView
            optimization={data}
            players={selectedPlayers}
            gameweek={1}
            showBench
          />
        </div>
      )}

      {!data && !isLoading && (
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-8 text-center">
          <p className="text-gray-400">
            Select players and click "Optimize Team" to see the optimized squad visualization.
          </p>
        </div>
      )}
    </div>
  )
}
