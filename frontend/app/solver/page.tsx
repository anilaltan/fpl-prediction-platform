'use client'

import { useState } from 'react'
import axios from 'axios'

export default function SolverSandboxPage() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [config, setConfig] = useState({
    budget: 100.0,
    horizon_weeks: 3,
    free_transfers: 1,
  })

  const handleOptimize = async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Use relative path - Next.js will forward to backend via rewrite rule
      // This is a simplified example - in production, you'd fetch actual player data
      const response = await axios.post('/api/solver/optimize-team', {
        players: [], // Would be populated with actual player data
        current_squad: [],
        budget: config.budget,
        horizon_weeks: config.horizon_weeks,
        free_transfers: config.free_transfers,
      })
      
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Optimization failed')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="container mx-auto max-w-6xl">
        <h1 className="text-4xl font-bold text-white mb-8">Solver Sandbox</h1>

        {/* Configuration */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 mb-6 border border-white/20">
          <h2 className="text-2xl font-semibold text-white mb-4">Configuration</h2>
          
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-gray-300 mb-2">Budget (£M)</label>
              <input
                type="number"
                value={config.budget}
                onChange={(e) => setConfig({ ...config, budget: Number(e.target.value) })}
                className="w-full bg-white/10 text-white border border-white/20 rounded px-4 py-2"
                min="0"
                max="200"
                step="0.1"
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Horizon (Weeks)</label>
              <input
                type="number"
                value={config.horizon_weeks}
                onChange={(e) => setConfig({ ...config, horizon_weeks: Number(e.target.value) })}
                className="w-full bg-white/10 text-white border border-white/20 rounded px-4 py-2"
                min="3"
                max="5"
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Free Transfers</label>
              <input
                type="number"
                value={config.free_transfers}
                onChange={(e) => setConfig({ ...config, free_transfers: Number(e.target.value) })}
                className="w-full bg-white/10 text-white border border-white/20 rounded px-4 py-2"
                min="0"
                max="2"
              />
            </div>
          </div>

          <button
            onClick={handleOptimize}
            disabled={loading}
            className="mt-6 bg-primary-600 hover:bg-primary-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors disabled:opacity-50"
          >
            {loading ? 'Optimizing...' : 'Optimize Team'}
          </button>
        </div>

        {error && (
          <div className="bg-red-500/20 border border-red-500/50 text-red-300 p-4 rounded mb-6">
            {error}
          </div>
        )}

        {result && (
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
            <h2 className="text-2xl font-semibold text-white mb-4">Optimization Results</h2>
            
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div>
                <p className="text-gray-300">Status:</p>
                <p className="text-white font-semibold">{result.status}</p>
              </div>
              <div>
                <p className="text-gray-300">Total Expected Points:</p>
                <p className="text-white font-semibold">{result.total_points?.toFixed(1)}</p>
              </div>
            </div>

            {/* Weekly Squads */}
            {result.squads && Object.entries(result.squads).map(([week, squad]: [string, any]) => (
              <div key={week} className="mb-4">
                <h3 className="text-xl font-semibold text-white mb-2">Gameweek {week}</h3>
                <div className="bg-white/5 rounded p-4">
                  <p className="text-gray-300 text-sm">Squad Size: {squad.length}</p>
                  <p className="text-gray-300 text-sm">Budget Used: £{result.budget_used?.[week]?.toFixed(1)}M</p>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="mt-8 bg-blue-500/20 border border-blue-500/50 rounded-lg p-6">
          <p className="text-blue-300 text-sm">
            <strong>Note:</strong> This is a sandbox environment. In production, this would include:
            - Player selection interface
            - Lock/exclude player functionality
            - Real-time optimization
            - Transfer recommendations
          </p>
        </div>
      </div>
    </main>
  )
}