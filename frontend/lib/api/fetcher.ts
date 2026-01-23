// SWR fetcher function for API calls
// Handles errors and provides consistent error handling

interface FetchError extends Error {
  status?: number
  info?: unknown
}

export const fetcher = async <T = unknown>(url: string): Promise<T> => {
  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    },
  })

  if (!res.ok) {
    const error = new Error('An error occurred while fetching the data.') as FetchError
    // Attach extra info to the error object
    error.status = res.status
    error.info = await res.json().catch(() => ({}))
    throw error
  }

  return res.json()
}

// POST fetcher for mutations
export const postFetcher = async <T = unknown>(
  url: string,
  data: unknown,
): Promise<T> => {
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!res.ok) {
    const error = new Error('An error occurred while posting the data.') as FetchError
    error.status = res.status
    error.info = await res.json().catch(() => ({}))
    throw error
  }

  return res.json()
}
