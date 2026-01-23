/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  
  // API rewrites for Docker environment
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://backend:8000/api/:path*', // Docker içindeki backend servisi
      },
    ]
  },
  
  // TypeScript ve ESLint hatalarını build sırasında yoksay
  typescript: {
    // Build sırasında TypeScript hatalarını yoksay
    ignoreBuildErrors: true,
  },
  eslint: {
    // Build sırasında ESLint hatalarını yoksay
    ignoreDuringBuilds: true,
  },
}

module.exports = nextConfig
