import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const gameweek = searchParams.get('gameweek')
    const season = searchParams.get('season') || '2025-26'
    
    // Build URL with parameters
    const params = new URLSearchParams()
    if (gameweek) {
      params.append('gameweek', gameweek)
    }
    if (season) {
      params.append('season', season)
    }
    
    const url = params.toString()
      ? `${BACKEND_URL}/market/intelligence?${params.toString()}`
      : `${BACKEND_URL}/market/intelligence`
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(30000), // 30 seconds
    })
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      return NextResponse.json(
        { 
          error: errorData.detail || 'Backend request failed', 
          status: response.status, 
        },
        { status: response.status },
      )
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error: any) {
    console.error('API route error:', error)
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 },
    )
  }
}
