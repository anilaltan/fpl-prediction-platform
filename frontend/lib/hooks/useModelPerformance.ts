import useSWR from 'swr'
import { fetcher } from '@/lib/api/fetcher'
import { ModelPerformanceResponse } from '@/lib/types/model-performance'

interface UseModelPerformanceOptions {
  season?: string
}

export function useModelPerformance(options: UseModelPerformanceOptions = {}) {
  const { season = '2025-26' } = options
  
  const { data, error, isLoading, mutate } = useSWR<ModelPerformanceResponse>(
    `/api/models/performance?season=${season}`,
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: true,
      refreshInterval: 0, // Don't auto-refresh
    },
  )

  return {
    data,
    isLoading,
    isError: error,
    error,
    mutate,
  }
}
