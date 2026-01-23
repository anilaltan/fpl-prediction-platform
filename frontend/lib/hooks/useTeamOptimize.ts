'use client'

import { useCallback } from 'react'
import useSWRMutation from 'swr/mutation'
import { postFetcher } from '../api/fetcher'
import type { TeamOptimizationRequest, TeamOptimizationResponse } from '../types/api'

interface UseTeamOptimizeOptions {
  onSuccess?: (data: TeamOptimizationResponse) => void
  onError?: (error: Error) => void
}

/**
 * SWR hook for team optimization endpoint
 * Optimizes FPL team selection for a single gameweek
 * 
 * Uses SWR mutation for POST requests since optimization is a mutation operation
 * 
 * @param options - Configuration options
 * @returns Mutation hook with optimize function
 */
export function useTeamOptimize(options: UseTeamOptimizeOptions = {}) {
  const { onSuccess, onError } = options

  const { trigger, data, error, isMutating, reset } = useSWRMutation<TeamOptimizationResponse>(
    '/api/team/optimize',
    (url: string, { arg }: { arg: TeamOptimizationRequest }) =>
      postFetcher<TeamOptimizationResponse>(url, arg),
    {
      onSuccess,
      onError,
    },
  )

  const optimize = useCallback(
    (request: TeamOptimizationRequest) => {
      return trigger(request)
    },
    [trigger],
  )

  return {
    data,
    error,
    isLoading: isMutating,
    isError: !!error,
    optimize,
    reset,
  }
}
