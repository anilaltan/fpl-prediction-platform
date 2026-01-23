/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        // FPL Analytics Theme - Green & Purple accents
        fpl: {
          green: {
            400: '#4ade80',
            500: '#22c55e',
            600: '#16a34a',
          },
          purple: {
            400: '#a78bfa',
            500: '#8b5cf6',
            600: '#7c3aed',
          },
          dark: {
            800: '#1e293b',
            900: '#0f172a',
            950: '#020617',
          },
        },
      },
    },
  },
  plugins: [],
}
