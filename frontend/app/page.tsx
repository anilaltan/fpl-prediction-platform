export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center">
          {/* Hero Section */}
          <div className="mb-16">
            <h1 className="text-6xl font-bold text-white mb-6">
              FPL Point Prediction
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              Machine Learning powered predictions for Fantasy Premier League
            </p>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto">
              Leveraging "Moneyball" principles and advanced statistical analysis 
              to predict player performance and optimize your FPL team selection.
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 mb-16">
            <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
              <div className="text-4xl mb-4">📊</div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Data-Driven Predictions
              </h3>
              <p className="text-gray-300">
                Advanced ML models analyze player statistics, form, and fixtures 
                to predict point returns.
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
              <div className="text-4xl mb-4">🎯</div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Moneyball Principles
              </h3>
              <p className="text-gray-300">
                Statistical analysis identifies undervalued players and optimal 
                team configurations.
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
              <div className="text-4xl mb-4">⚡</div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Real-Time Updates
              </h3>
              <p className="text-gray-300">
                Stay ahead with live predictions updated throughout the gameweek 
                based on latest data.
              </p>
            </div>
          </div>

          {/* CTA Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-8 border border-white/20">
            <h2 className="text-2xl font-semibold text-white mb-4">
              Ready to Optimize Your Team?
            </h2>
            <p className="text-gray-300 mb-6">
              Get started with AI-powered FPL predictions today.
            </p>
            <div className="flex gap-4 justify-center">
              <button className="bg-primary-600 hover:bg-primary-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors">
                Get Started
              </button>
              <button className="bg-white/10 hover:bg-white/20 text-white font-semibold py-3 px-8 rounded-lg border border-white/20 transition-colors">
                Learn More
              </button>
            </div>
          </div>

          {/* Status Indicator */}
          <div className="mt-12">
            <div className="inline-flex items-center gap-2 bg-green-500/20 text-green-300 px-4 py-2 rounded-full border border-green-500/30">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm">API Status: Operational</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
