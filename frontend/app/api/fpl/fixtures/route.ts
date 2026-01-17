import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const gameweek = searchParams.get('gameweek')
    const future_only = searchParams.get('future_only') === 'true'
    
    // Build URL with parameters
    const params = new URLSearchParams()
    if (gameweek) {
      params.append('gameweek', gameweek)
    }
    if (future_only) {
      params.append('future_only', 'true')
    }
    
    const url = params.toString()
      ? `${BACKEND_URL}/api/fpl/fixtures?${params.toString()}`
      : `${BACKEND_URL}/api/fpl/fixtures`
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(30000), // 30 seconds
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
