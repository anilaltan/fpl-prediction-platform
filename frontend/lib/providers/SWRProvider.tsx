'use client'

import { SWRConfig } from 'swr'
import { fetcher } from '../api/fetcher'

interface SWRProviderProps {
  children: React.ReactNode
}

/**
 * SWR Provider component for global SWR configuration
 * Provides default fetcher and configuration for all SWR hooks
 */
export function SWRProvider({ children }: SWRProviderProps) {
  return (
    <SWRConfig
      value={{
        fetcher,
        revalidateOnFocus: true,
        revalidateOnReconnect: true,
        dedupingInterval: 2000,
        errorRetryCount: 3,
        errorRetryInterval: 5000,
        onError: (error, key) => {
          // Global error handler
          console.error('SWR Error:', error, 'for key:', key)
        },
      }}
    >
      {children}
    </SWRConfig>
  )
}
