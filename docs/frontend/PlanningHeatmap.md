# PlanningHeatmap Component

A React component that visualizes multi-period transfer strategy as a heatmap grid.

## Features

- **Player vs Gameweek Grid**: Shows all players (rows) across all gameweeks (columns)
- **Color-Coded Expected Points**: Color intensity represents expected points (green = high, red = low)
- **Transfer Indicators**: Visual markers for transfers in/out each gameweek
- **Starting XI Indicators**: Shows which players are in starting XI vs bench
- **Position Grouping**: Players organized by position (GK, DEF, MID, FWD)
- **Summary Statistics**: Total expected points, transfer costs, and net points

## Usage

```tsx
import { PlanningHeatmap } from '@/components'
import { useTeamPlan } from '@/lib/hooks'
import type { TeamPlanResponse, PlayerOptimizationData } from '@/lib/types/api'

function MyComponent() {
  const { plan, data } = useTeamPlan()
  const [players, setPlayers] = useState<PlayerOptimizationData[]>([])

  // ... generate plan logic ...

  if (data) {
    return (
      <PlanningHeatmap
        plan={data}
        players={players}
        colorBy="expected_points"
      />
    )
  }
}
```

## Props

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `plan` | `TeamPlanResponse` | Yes | The multi-period plan from `/team/plan` endpoint |
| `players` | `PlayerOptimizationData[]` | Yes | Array of all players used in planning (for player details) |
| `colorBy` | `'expected_points' \| 'fixture_difficulty'` | No | What to color by (default: 'expected_points') |

## Component Structure

### Grid Layout
- **Rows**: Each player in the squad (organized by position)
- **Columns**: Each gameweek in the planning horizon (3-5 weeks)
- **Cells**: Show player status for that gameweek

### Cell States
- **In Squad**: Colored cell with expected points
- **Not in Squad**: Gray cell with "-"
- **Starting XI**: White border around cell
- **Bench**: No border (colored cell only)

### Color Coding
- **Green (High)**: Expected points ≥ 12 (80%+ of max)
- **Green (Medium-High)**: Expected points ≥ 9 (60-80%)
- **Yellow**: Expected points ≥ 6 (40-60%)
- **Orange**: Expected points ≥ 3 (20-40%)
- **Red (Low)**: Expected points < 3 (< 20%)

### Transfer Indicators
- **Blue Dot**: Transfer In (player added to squad)
- **Red Dot**: Transfer Out (player removed from squad)
- Shown in top-right corner of cell

### Starting XI Indicator
- **White Dot**: Player is in starting XI for that gameweek
- Shown in top-left corner of cell

## Summary Statistics

The component displays:
- **Total Expected Points**: Sum of all expected points across all gameweeks
- **Transfer Cost**: Total points deducted for transfers
- **Net Expected Points**: Total points minus transfer costs
- **Planning Horizon**: Number of weeks in the plan

## Example

See `/app/team-planner/page.tsx` for a complete example of using the PlanningHeatmap component with the team planning hook.

## Styling

The component uses Tailwind CSS with the FPL theme colors:
- Grid background: Dark theme with borders
- Color gradients: Green to red based on expected points
- Position indicators: Color-coded dots (blue=GK, green=DEF, yellow=MID, red=FWD)
- Responsive design: Horizontal scroll on mobile devices
