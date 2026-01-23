export interface ModelPerformanceSummary {
  model_version: string
  methodology: string
  season: string
  total_weeks_tested: number
  overall_rmse: number | null
  overall_mae: number | null
  overall_spearman_corr: number | null
  r_squared: number | null
  total_predictions: number
  created_at: string | null
  updated_at: string | null
}

export interface ModelPerformanceResult {
  model_version: string
  methodology: string
  gameweek: number
  rmse: number | null
  mae: number | null
  spearman_corr: number | null
  n_predictions: number
  created_at: string | null
}

export interface ModelPerformanceRecord {
  model_version: string
  gameweek: number
  mae: number | null
  rmse: number | null
  accuracy: number | null
  created_at: string | null
}

export interface ModelPerformanceResponse {
  season: string
  summaries: ModelPerformanceSummary[]
  results: ModelPerformanceResult[]
  model_performance: ModelPerformanceRecord[]
}
