// SWR fetcher function for API calls
// Handles errors and provides consistent error handling

export const fetcher = async <T = any>(url: string): Promise<T> => {
  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    },
  })

  if (!res.ok) {
    const error = new Error('An error occurred while fetching the data.')
    // Attach extra info to the error object
    ;(error as any).status = res.status
    ;(error as any).info = await res.json().catch(() => ({}))
    throw error
  }

  return res.json()
}

// POST fetcher for mutations
export const postFetcher = async <T = any>(
  url: string,
  data: any,
): Promise<T> => {
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!res.ok) {
    const error = new Error('An error occurred while posting the data.')
    ;(error as any).status = res.status
    ;(error as any).info = await res.json().catch(() => ({}))
    throw error
  }

  return res.json()
}
