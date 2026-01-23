# PitchView Component

A React component that visualizes a 15-player FPL squad on a football pitch layout.

## Features

- **Visual Pitch Layout**: Displays starting XI players on a football pitch with proper positioning
- **Position-Based Organization**: Automatically organizes players by position (GK, DEF, MID, FWD)
- **Bench Display**: Shows bench players (4 players not in starting XI)
- **Formation Display**: Automatically calculates and displays formation (e.g., "4-4-2")
- **Squad Summary**: Shows squad size, budget used, and optimization status
- **Responsive Design**: Works on mobile and desktop devices

## Usage

```tsx
import { PitchView } from '@/components'
import { useTeamOptimize } from '@/lib/hooks'
import type { TeamOptimizationResponse, PlayerOptimizationData } from '@/lib/types/api'

function MyComponent() {
  const { optimize, data } = useTeamOptimize()
  const [players, setPlayers] = useState<PlayerOptimizationData[]>([])

  // ... optimize team logic ...

  if (data) {
    return (
      <PitchView
        optimization={data}
        players={players}
        gameweek={1}
        showBench={true}
      />
    )
  }
}
```

## Props

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `optimization` | `TeamOptimizationResponse` | Yes | The optimization result from `/team/optimize` endpoint |
| `players` | `PlayerOptimizationData[]` | Yes | Array of all players used in optimization (for player details) |
| `gameweek` | `number` | No | Gameweek to display (default: 1) |
| `showBench` | `boolean` | No | Whether to show bench players (default: true) |

## Component Structure

### Starting XI
- **Forwards (FWD)**: Top row of the pitch
- **Midfielders (MID)**: Middle row
- **Defenders (DEF)**: Bottom row (above goalkeeper)
- **Goalkeeper (GK)**: Bottom of the pitch

### Bench
- Displays all players in the squad who are not in the starting XI
- Shows player name, position, expected points, price, and team

### Squad Summary
- Squad size (current/15)
- Starting XI count (current/11)
- Budget used
- Optimization status

## Player Card Information

Each player card displays:
- **Team Initial**: First 3 letters of team name
- **Player Name**: Shortened name (first name + last initial)
- **Expected Points**: Bold, green text
- **Price**: Gray text below expected points
- **Position Color**: Color-coded border based on position
  - GK: Blue
  - DEF: Green
  - MID: Yellow
  - FWD: Red

## Styling

The component uses Tailwind CSS with the FPL theme colors:
- Pitch background: Green gradient with border
- Player cards: Position-colored borders with dark background
- Bench: Dark background with border
- Summary: Dark background with grid layout

## Example

See `/app/team-optimizer/page.tsx` for a complete example of using the PitchView component with the team optimization hook.
