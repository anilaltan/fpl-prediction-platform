import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    // If gameweek is provided, use it; otherwise let backend use current gameweek
    const gameweek = searchParams.get('gameweek')
    const url = gameweek 
      ? `${BACKEND_URL}/api/dream-team?gameweek=${gameweek}`
      : `${BACKEND_URL}/api/dream-team`
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(120000), // 120 seconds for solver
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
