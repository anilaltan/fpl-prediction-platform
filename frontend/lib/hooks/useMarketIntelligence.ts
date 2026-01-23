'use client'

import useSWR from 'swr'
import { fetcher } from '../api/fetcher'
import type { MarketIntelligenceResponse } from '../types/api'

interface UseMarketIntelligenceOptions {
  gameweek?: number
  season?: string
  enabled?: boolean
}

/**
 * SWR hook for market intelligence endpoint
 * Fetches ownership arbitrage analysis and player rankings
 * 
 * @param options - Configuration options
 * @returns SWR response with market intelligence data
 */
export function useMarketIntelligence(options: UseMarketIntelligenceOptions = {}) {
  const { gameweek, season = '2025-26', enabled = true } = options

  // Build query string
  const params = new URLSearchParams()
  if (gameweek) {
    params.append('gameweek', gameweek.toString())
  }
  if (season) {
    params.append('season', season)
  }

  const url = params.toString()
    ? `/api/market/intelligence?${params.toString()}`
    : '/api/market/intelligence'

  const { data, error, isLoading, mutate } = useSWR<MarketIntelligenceResponse>(
    enabled ? url : null,
    fetcher,
    {
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
      refreshInterval: 60000, // Refresh every minute (market data changes frequently)
      dedupingInterval: 30000, // 30 seconds
      errorRetryCount: 3,
      errorRetryInterval: 2000,
    },
  )

  return {
    data,
    error,
    isLoading,
    isError: !!error,
    mutate,
  }
}
