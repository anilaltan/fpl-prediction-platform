'use client'

import { useState } from 'react'
import { useTeamPlan } from '@/lib/hooks'
import { PlanningHeatmap } from '@/components'
import type { PlayerOptimizationData } from '@/lib/types/api'

export default function TeamPlannerPage() {
  const [selectedPlayers] = useState<PlayerOptimizationData[]>([])
  const [horizonWeeks, setHorizonWeeks] = useState(3)
  const { plan, data, isLoading, error } = useTeamPlan({
    onSuccess: (_result) => {
      // Planning complete
    },
    onError: (_err) => {
      // Planning failed
    },
  })

  const handlePlan = async () => {
    if (selectedPlayers.length < 15) {
      // eslint-disable-next-line no-alert
      alert('Please select at least 15 players')
      return
    }

    await plan({
      players: selectedPlayers,
      budget: 100.0,
      horizon_weeks: horizonWeeks,
      free_transfers: 1,
    })
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Multi-Week Team Planner</h1>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-300">Horizon:</label>
            <select
              value={horizonWeeks}
              onChange={(e) => setHorizonWeeks(Number(e.target.value))}
              className="bg-fpl-dark-800 border border-fpl-dark-700 text-white rounded px-3 py-2 text-sm"
            >
              <option value={3}>3 weeks</option>
              <option value={4}>4 weeks</option>
              <option value={5}>5 weeks</option>
            </select>
          </div>
          <button
            onClick={handlePlan}
            disabled={isLoading || selectedPlayers.length < 15}
            className="px-6 py-3 bg-fpl-purple-500 hover:bg-fpl-purple-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-colors"
          >
            {isLoading ? 'Planning...' : 'Generate Plan'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4">
          <p className="text-red-400">Error: {error.message}</p>
        </div>
      )}

      {data && (
        <PlanningHeatmap plan={data} players={selectedPlayers} />
      )}

      {!data && !isLoading && (
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-8 text-center">
          <p className="text-gray-400">
            Select players and click &quot;Generate Plan&quot; to see the multi-week planning
            heatmap.
          </p>
        </div>
      )}
    </div>
  )
}
