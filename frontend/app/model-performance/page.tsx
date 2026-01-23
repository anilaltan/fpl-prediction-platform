'use client'

import { useModelPerformance } from '@/lib/hooks'

// Helper function to safely format numbers
const formatNumber = (value: number | null | undefined, decimals: number = 3): string => {
  if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
    return '-'
  }
  return value.toFixed(decimals)
}

export default function ModelPerformancePage() {
  const { data, isLoading, isError, error } = useModelPerformance({ season: '2025-26' })

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-8 text-center">
          <p className="text-gray-400">Loading model performance data...</p>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="p-6">
        <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4">
          <p className="text-red-400">Error: {error?.message || 'Failed to load model performance'}</p>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="p-6">
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-8 text-center">
          <p className="text-gray-400">No model performance data available</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Model Performance</h1>
          <p className="text-gray-400 mt-1">Season {data.season}</p>
        </div>
      </div>

      {/* Backtest Summaries */}
      {data.summaries.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-white">Backtest Summaries</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.summaries.map((summary, idx) => (
              <div
                key={idx}
                className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">{summary.model_version}</h3>
                  <span className="text-xs text-gray-400 bg-fpl-dark-800 px-2 py-1 rounded">
                    {summary.methodology}
                  </span>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">RMSE</span>
                    <span className="text-sm font-medium text-white">{formatNumber(summary.overall_rmse, 3)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">MAE</span>
                    <span className="text-sm font-medium text-white">{formatNumber(summary.overall_mae, 3)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Spearman Corr</span>
                    <span className="text-sm font-medium text-white">{formatNumber(summary.overall_spearman_corr, 3)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">R²</span>
                    <span className="text-sm font-medium text-white">{formatNumber(summary.r_squared, 3)}</span>
                  </div>
                  <div className="flex justify-between items-center pt-2 border-t border-fpl-dark-800">
                    <span className="text-sm text-gray-400">Weeks Tested</span>
                    <span className="text-sm font-medium text-white">{summary.total_weeks_tested}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Predictions</span>
                    <span className="text-sm font-medium text-white">{summary.total_predictions.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-Gameweek Results */}
      {data.results.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-white">Per-Gameweek Performance</h2>
          <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-fpl-dark-800">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      Gameweek
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      Model
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      RMSE
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      MAE
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      Spearman
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      Predictions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-fpl-dark-800">
                  {data.results.map((result, idx) => (
                    <tr key={idx} className="hover:bg-fpl-dark-800/50 transition-colors">
                      <td className="px-4 py-3 text-sm text-white">{result.gameweek}</td>
                      <td className="px-4 py-3 text-sm text-gray-300">{result.model_version}</td>
                      <td className="px-4 py-3 text-sm text-white">{formatNumber(result.rmse, 3)}</td>
                      <td className="px-4 py-3 text-sm text-white">{formatNumber(result.mae, 3)}</td>
                      <td className="px-4 py-3 text-sm text-white">{formatNumber(result.spearman_corr, 3)}</td>
                      <td className="px-4 py-3 text-sm text-gray-300">{result.n_predictions}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Model Performance Records */}
      {data.model_performance.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-white">Model Performance Records</h2>
          <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-fpl-dark-800">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      Gameweek
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      Model Version
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      RMSE
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      MAE
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      Accuracy
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-fpl-dark-800">
                  {data.model_performance.map((record, idx) => (
                    <tr key={idx} className="hover:bg-fpl-dark-800/50 transition-colors">
                      <td className="px-4 py-3 text-sm text-white">{record.gameweek}</td>
                      <td className="px-4 py-3 text-sm text-gray-300">{record.model_version}</td>
                      <td className="px-4 py-3 text-sm text-white">{formatNumber(record.rmse, 3)}</td>
                      <td className="px-4 py-3 text-sm text-white">{formatNumber(record.mae, 3)}</td>
                      <td className="px-4 py-3 text-sm text-white">
                        {record.accuracy !== null && record.accuracy !== undefined 
                          ? `${formatNumber(record.accuracy * 100, 1)}%`
                          : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {data.summaries.length === 0 && data.results.length === 0 && data.model_performance.length === 0 && (
        <div className="bg-fpl-dark-900 border border-fpl-dark-800 rounded-lg p-8 text-center">
          <p className="text-gray-400">No model performance data available yet.</p>
          <p className="text-sm text-gray-500 mt-2">Run backtests to generate performance metrics.</p>
        </div>
      )}
    </div>
  )
}
