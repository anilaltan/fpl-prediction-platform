import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000'

export async function GET(_request: Request) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/fpl/bootstrap`, {
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(30000), // 30 seconds
    })
    
    if (!response.ok) {
      return NextResponse.json(
        { error: 'Backend request failed', status: response.status },
        { status: response.status },
      )
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Internal server error'
    // eslint-disable-next-line no-console
    console.error('API route error:', error)
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 },
    )
  }
}
