'use client'

import { useState } from 'react'
import { useMarketIntelligence } from '@/lib/hooks'
import { MarketIntelligenceTable } from '@/components'

export default function MarketIntelligencePage() {
  const [gameweek, setGameweek] = useState<number | undefined>(undefined)
  const { data, isLoading, error } = useMarketIntelligence({
    gameweek,
    season: '2025-26',
  })

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Market Intelligence</h1>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-300">Gameweek:</label>
            <input
              type="number"
              value={gameweek || ''}
              onChange={(e) =>
                setGameweek(e.target.value ? Number(e.target.value) : undefined)
              }
              placeholder="Current"
              min={1}
              max={38}
              className="w-24 bg-fpl-dark-800 border border-fpl-dark-700 text-white rounded px-3 py-2 text-sm focus:outline-none focus:border-fpl-green-500"
            />
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4">
          <p className="text-red-400">Error: {error.message}</p>
        </div>
      )}

      {isLoading && (
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-8 text-center">
          <p className="text-gray-400">Loading market intelligence data...</p>
        </div>
      )}

      {data && !isLoading && (
        <MarketIntelligenceTable
          players={data.players}
          gameweek={data.gameweek}
          season={data.season}
        />
      )}

      {!data && !isLoading && !error && (
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-8 text-center">
          <p className="text-gray-400">No market intelligence data available.</p>
        </div>
      )}
    </div>
  )
}
