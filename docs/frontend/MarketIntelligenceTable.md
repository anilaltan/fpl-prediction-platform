# MarketIntelligenceTable Component

A React component that displays ownership arbitrage analysis in a sortable, filterable table.

## Features

- **Sortable Columns**: Click any column header to sort by that field
- **Category Highlighting**: Visual distinction between Differentials, Overvalued, and Neutral players
- **Arbitrage Score**: Color-coded scores to identify market opportunities
- **Filtering**: Search by name/team, filter by category and position
- **Real-time Stats**: Category counts and player statistics
- **Responsive Design**: Works on mobile and desktop devices

## Usage

```tsx
import { MarketIntelligenceTable } from '@/components'
import { useMarketIntelligence } from '@/lib/hooks'

function MyComponent() {
  const { data, isLoading } = useMarketIntelligence({
    gameweek: 1,
    season: '2025-26',
  })

  if (data) {
    return (
      <MarketIntelligenceTable
        players={data.players}
        gameweek={data.gameweek}
        season={data.season}
      />
    )
  }
}
```

## Props

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `players` | `MarketIntelligencePlayer[]` | Yes | Array of market intelligence player data |
| `gameweek` | `number` | No | Gameweek number (for display) |
| `season` | `string` | No | Season string (for display) |

## Table Columns

1. **Player**: Player name and team
2. **Pos**: Position (GK, DEF, MID, FWD) with color coding
3. **Price**: Player price in millions
4. **xP**: Expected points (green, bold)
5. **xP Rank**: Rank by expected points (lower = better)
6. **Ownership %**: Ownership percentage
7. **Own Rank**: Rank by ownership (lower = more owned)
8. **Arbitrage Score**: Calculated score (xP Rank - Ownership Rank)
9. **Category**: Differential, Overvalued, or Neutral

## Sorting

Click any column header to sort. Click again to reverse direction. Default sort is by Arbitrage Score (ascending - best differentials first).

## Category Logic

- **Differential** (Green): High xP rank, low ownership rank (negative arbitrage score)
  - Undervalued by the market
  - Good opportunity for differential picks
  
- **Overvalued** (Red): Low xP rank, high ownership rank (positive arbitrage score)
  - Overvalued by the market
  - Consider avoiding or selling

- **Neutral** (Gray): Balanced xP and ownership ranks
  - Market pricing is fair

## Arbitrage Score

**Formula**: `Arbitrage Score = xP Rank - Ownership Rank`

- **Negative values** (green): Differential opportunity
  - Player has high expected points but low ownership
  - Example: xP Rank #5, Ownership Rank #50 → Score: -45 (strong differential)
  
- **Positive values** (red): Overvalued
  - Player has low expected points but high ownership
  - Example: xP Rank #100, Ownership Rank #10 → Score: +90 (strongly overvalued)

- **Near zero** (gray): Neutral
  - Market pricing aligns with expected points

## Color Coding

### Arbitrage Score Colors
- **Green (bold)**: Score < -20 (strong differential)
- **Green**: Score < -10 (moderate differential)
- **Gray**: Score between -10 and +10 (neutral)
- **Red**: Score > +10 (moderate overvalued)
- **Red (bold)**: Score > +20 (strongly overvalued)

### Position Colors
- **GK**: Blue
- **DEF**: Green
- **MID**: Yellow
- **FWD**: Red

### Category Colors
- **Differential**: Green background
- **Overvalued**: Red background
- **Neutral**: Gray background

## Filters

- **Search**: Filter by player name or team name
- **Category**: Filter by Differential, Overvalued, or Neutral
- **Position**: Filter by GK, DEF, MID, or FWD

## Example

See `/app/market-intelligence/page.tsx` for a complete example of using the MarketIntelligenceTable component with the market intelligence hook.

## Styling

The component uses Tailwind CSS with the FPL theme colors:
- Table background: Dark theme with borders
- Hover effects: Subtle background color change
- Category badges: Color-coded with borders
- Responsive: Horizontal scroll on mobile devices
