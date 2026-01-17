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
      signal: AbortSignal.timeout(60000), // 60 seconds
    })
    
    if (!response.ok) {
      return NextResponse.json(
        { error: 'Backend request failed', status: response.status },
        { status: response.status }
      )
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error: any) {
    console.error('API route error:', error)
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    )
  }
}
