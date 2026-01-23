import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Link from 'next/link'
import { SWRProvider } from '@/lib/providers/SWRProvider'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'FPL Analytics Dashboard',
  description: 'Machine Learning powered Fantasy Premier League point predictions',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <div className="flex h-screen bg-fpl-dark-950">
          {/* Sidebar */}
          <aside className="w-64 bg-fpl-dark-900 border-r border-fpl-dark-800 flex flex-col">
            <div className="p-6 border-b border-fpl-dark-800">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-fpl-green-500 to-fpl-purple-500 bg-clip-text text-transparent">
                FPL Analytics
              </h1>
              <p className="text-xs text-gray-400 mt-1">ML-Powered Predictions</p>
            </div>
            
            <nav className="flex-1 p-4 space-y-6 overflow-y-auto">
              {/* Overview Section */}
              <div className="space-y-2">
                <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-4 mb-2">
                  Overview
                </h2>
                <Link
                  href="/"
                  className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-fpl-dark-800 hover:text-white transition-colors group"
                >
                  <svg className="w-5 h-5 group-hover:text-fpl-green-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                  </svg>
                  <span className="font-medium">Dashboard</span>
                </Link>
                
                <Link
                  href="/players"
                  className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-fpl-dark-800 hover:text-white transition-colors group"
                >
                  <svg className="w-5 h-5 group-hover:text-fpl-green-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                  <span className="font-medium">Players</span>
                </Link>
                
                <Link
                  href="/dream-team"
                  className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-fpl-dark-800 hover:text-white transition-colors group"
                >
                  <svg className="w-5 h-5 group-hover:text-fpl-purple-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                  </svg>
                  <span className="font-medium">Dream Team</span>
                </Link>
              </div>

              {/* Optimization Section */}
              <div className="space-y-2">
                <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-4 mb-2">
                  Optimization
                </h2>
                <Link
                  href="/team-optimizer"
                  className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-fpl-dark-800 hover:text-white transition-colors group"
                >
                  <svg className="w-5 h-5 group-hover:text-fpl-green-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                  <span className="font-medium">Team Optimizer</span>
                </Link>
                
                <Link
                  href="/team-planner"
                  className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-fpl-dark-800 hover:text-white transition-colors group"
                >
                  <svg className="w-5 h-5 group-hover:text-fpl-purple-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                  <span className="font-medium">Team Planner</span>
                </Link>
                
                <Link
                  href="/solver"
                  className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-fpl-dark-800 hover:text-white transition-colors group"
                >
                  <svg className="w-5 h-5 group-hover:text-fpl-purple-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  <span className="font-medium">Solver</span>
                </Link>
              </div>

              {/* Intelligence Section */}
              <div className="space-y-2">
                <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-4 mb-2">
                  Intelligence
                </h2>
                <Link
                  href="/market-intelligence"
                  className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-fpl-dark-800 hover:text-white transition-colors group"
                >
                  <svg className="w-5 h-5 group-hover:text-fpl-green-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  <span className="font-medium">Market Intelligence</span>
                </Link>
                
                <Link
                  href="/model-performance"
                  className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-fpl-dark-800 hover:text-white transition-colors group"
                >
                  <svg className="w-5 h-5 group-hover:text-fpl-purple-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  <span className="font-medium">Model Performance</span>
                </Link>
              </div>
            </nav>
            
            <div className="p-4 border-t border-fpl-dark-800">
              <p className="text-xs text-gray-500 text-center">
                Powered by ML Engine
              </p>
            </div>
          </aside>
          
          {/* Main Content */}
          <main className="flex-1 overflow-y-auto">
            <SWRProvider>
              {children}
            </SWRProvider>
          </main>
        </div>
      </body>
    </html>
  )
}
