'use client'

import { useCallback } from 'react'
import useSWRMutation from 'swr/mutation'
import { postFetcher } from '../api/fetcher'
import type { TeamPlanRequest, TeamPlanResponse } from '../types/api'

interface UseTeamPlanOptions {
  onSuccess?: (data: TeamPlanResponse) => void
  onError?: (error: Error) => void
}

/**
 * SWR hook for multi-period team planning endpoint
 * Generates transfer strategy across 3-5 gameweeks
 * 
 * Uses SWR mutation for POST requests since planning is a mutation operation
 * 
 * @param options - Configuration options
 * @returns Mutation hook with plan function
 */
export function useTeamPlan(options: UseTeamPlanOptions = {}) {
  const { onSuccess, onError } = options

  const { trigger, data, error, isMutating, reset } = useSWRMutation<TeamPlanResponse>(
    '/api/team/plan',
    (url: string, { arg }: { arg: TeamPlanRequest }) =>
      postFetcher<TeamPlanResponse>(url, arg),
    {
      onSuccess,
      onError,
    },
  )

  const plan = useCallback(
    (request: TeamPlanRequest) => {
      return trigger(request)
    },
    [trigger],
  )

  return {
    data,
    error,
    isLoading: isMutating,
    isError: !!error,
    plan,
    reset,
  }
}
