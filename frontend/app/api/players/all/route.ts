import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    // If gameweek is provided, use it; otherwise let backend use current gameweek
    const gameweek = searchParams.get('gameweek')
    const useNextGameweek = searchParams.get('use_next_gameweek') === 'true'
    
    // Build URL with parameters
    const params = new URLSearchParams()
    if (gameweek) {
      params.append('gameweek', gameweek)
    }
    if (useNextGameweek) {
      params.append('use_next_gameweek', 'true')
    }
    
    const url = params.toString()
      ? `${BACKEND_URL}/api/players/all?${params.toString()}`
      : `${BACKEND_URL}/api/players/all`
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      // Increase timeout for ML predictions
      // Reduced to 30 seconds - backend should respond faster with optimizations
      signal: AbortSignal.timeout(30000), // 30 seconds
    })
    
    if (!response.ok) {
      // Try to get error details from backend
      let errorDetail = 'Backend request failed'
      try {
        const errorData = await response.json().catch(() => null)
        if (errorData && errorData.detail) {
          errorDetail = errorData.detail
        }
      } catch {
        // Ignore JSON parse errors
      }
      
      // eslint-disable-next-line no-console
      console.error(`Backend API error (${response.status}): ${errorDetail}`)
      return NextResponse.json(
        { error: errorDetail, status: response.status },
        { status: response.status },
      )
    }
    
    const data = await response.json()
    
    // Validate response is an array
    if (!Array.isArray(data)) {
      // eslint-disable-next-line no-console
      console.error('Backend returned non-array response:', typeof data)
      return NextResponse.json(
        { error: 'Invalid response format from backend', data },
        { status: 500 },
      )
    }
    
    return NextResponse.json(data)
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error('API route error:', error)
    
    // Handle timeout errors
    if (error instanceof Error && (error.name === 'TimeoutError' || error.name === 'AbortError')) {
      return NextResponse.json(
        { error: 'Request timeout - backend took too long to respond' },
        { status: 504 },
      )
    }
    
    const errorMessage = error instanceof Error ? error.message : 'Internal server error'
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 },
    )
  }
}
