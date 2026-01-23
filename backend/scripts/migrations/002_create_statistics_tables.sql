-- Migration: 002_create_statistics_tables.sql
-- Description: Create statistics tables (player_stats, team_stats) and convert to TimescaleDB hypertables
-- Dependencies: 
--   - TimescaleDB extension (subtask 1.1)
--   - Core reference tables: teams, players (subtask 1.2)
-- Date: 2026-01-18

-- ============================================================================
-- PLAYER_STATS TABLE
-- ============================================================================
-- Time-series table for player performance statistics per gameweek
-- Note: Using SERIAL without PRIMARY KEY to allow TimescaleDB hypertable conversion
-- We'll create a unique index on id after hypertable conversion
CREATE TABLE IF NOT EXISTS player_stats (
    id SERIAL,
    player_id INTEGER NOT NULL REFERENCES players(id),
    gameweek INTEGER NOT NULL,
    season VARCHAR(9) NOT NULL,
    minutes INTEGER DEFAULT 0,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    clean_sheets INTEGER DEFAULT 0,
    points INTEGER DEFAULT 0,
    xg DECIMAL(6,3),
    xa DECIMAL(6,3),
    npxg DECIMAL(6,3),
    xmins DECIMAL(5,2),
    xp DECIMAL(6,2),
    defcon_points INTEGER DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for player_stats queries (created before hypertable conversion)
CREATE INDEX IF NOT EXISTS idx_player_stats_player_id ON player_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_stats_gameweek_season ON player_stats(gameweek, season);
CREATE INDEX IF NOT EXISTS idx_player_stats_timestamp ON player_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_player_stats_player_gameweek_season ON player_stats(player_id, gameweek, season);

-- Note: Unique constraint will be created AFTER hypertable conversion
-- TimescaleDB requires unique constraints to include the partitioning column (timestamp)

COMMENT ON TABLE player_stats IS 'Time-series statistics for FPL players per gameweek';
COMMENT ON COLUMN player_stats.player_id IS 'Foreign key to players.id';
COMMENT ON COLUMN player_stats.gameweek IS 'Gameweek number (1-38)';
COMMENT ON COLUMN player_stats.season IS 'Season identifier (e.g., 2025-26)';
COMMENT ON COLUMN player_stats.xg IS 'Expected Goals';
COMMENT ON COLUMN player_stats.xa IS 'Expected Assists';
COMMENT ON COLUMN player_stats.npxg IS 'Non-Penalty Expected Goals';
COMMENT ON COLUMN player_stats.xmins IS 'Expected Minutes';
COMMENT ON COLUMN player_stats.xp IS 'Expected Points';
COMMENT ON COLUMN player_stats.defcon_points IS 'DefCon points (blocks, tackles, interceptions)';
COMMENT ON COLUMN player_stats.timestamp IS 'Timestamp for time-series partitioning (TimescaleDB hypertable)';

-- ============================================================================
-- TEAM_STATS TABLE
-- ============================================================================
-- Time-series table for team performance statistics per gameweek
-- Note: Using SERIAL without PRIMARY KEY to allow TimescaleDB hypertable conversion
-- We'll create a unique index on id after hypertable conversion
CREATE TABLE IF NOT EXISTS team_stats (
    id SERIAL,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    gameweek INTEGER NOT NULL,
    season VARCHAR(9) NOT NULL,
    xgc DECIMAL(6,3),
    xgs DECIMAL(6,3),
    possession DECIMAL(5,2),
    clean_sheets INTEGER DEFAULT 0,
    goals_conceded INTEGER DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for team_stats queries (created before hypertable conversion)
CREATE INDEX IF NOT EXISTS idx_team_stats_team_id ON team_stats(team_id);
CREATE INDEX IF NOT EXISTS idx_team_stats_gameweek_season ON team_stats(gameweek, season);
CREATE INDEX IF NOT EXISTS idx_team_stats_timestamp ON team_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_team_stats_team_gameweek_season ON team_stats(team_id, gameweek, season);

-- Note: Unique constraint will be created AFTER hypertable conversion
-- TimescaleDB requires unique constraints to include the partitioning column (timestamp)

COMMENT ON TABLE team_stats IS 'Time-series statistics for Premier League teams per gameweek';
COMMENT ON COLUMN team_stats.team_id IS 'Foreign key to teams.id';
COMMENT ON COLUMN team_stats.gameweek IS 'Gameweek number (1-38)';
COMMENT ON COLUMN team_stats.season IS 'Season identifier (e.g., 2025-26)';
COMMENT ON COLUMN team_stats.xgc IS 'Expected Goals Conceded';
COMMENT ON COLUMN team_stats.xgs IS 'Expected Goals Scored';
COMMENT ON COLUMN team_stats.possession IS 'Possession percentage (0-100)';
COMMENT ON COLUMN team_stats.timestamp IS 'Timestamp for time-series partitioning (TimescaleDB hypertable)';

-- ============================================================================
-- CONVERT TO TIMESCALEDB HYPERTABLES
-- ============================================================================
-- Convert player_stats to hypertable for time-series optimization
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'player_stats'
    ) THEN
        PERFORM create_hypertable('player_stats', 'timestamp');
        RAISE NOTICE '✓ Converted player_stats to TimescaleDB hypertable';
        
        -- Create unique constraint AFTER hypertable conversion
        -- TimescaleDB requires unique constraints to include the partitioning column (timestamp)
        CREATE UNIQUE INDEX IF NOT EXISTS uq_player_stats_player_gameweek_season_timestamp 
            ON player_stats(player_id, gameweek, season, timestamp);
        
        -- Create unique index on id for lookups (acts as primary key)
        CREATE UNIQUE INDEX IF NOT EXISTS player_stats_id_idx ON player_stats(id);
    ELSE
        RAISE NOTICE '⏭  player_stats is already a hypertable';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Could not convert player_stats to hypertable: %', SQLERRM;
END $$;

-- Convert team_stats to hypertable for time-series optimization
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'team_stats'
    ) THEN
        PERFORM create_hypertable('team_stats', 'timestamp');
        RAISE NOTICE '✓ Converted team_stats to TimescaleDB hypertable';
        
        -- Create unique constraint AFTER hypertable conversion
        -- TimescaleDB requires unique constraints to include the partitioning column (timestamp)
        CREATE UNIQUE INDEX IF NOT EXISTS uq_team_stats_team_gameweek_season_timestamp 
            ON team_stats(team_id, gameweek, season, timestamp);
        
        -- Create unique index on id for lookups (acts as primary key)
        CREATE UNIQUE INDEX IF NOT EXISTS team_stats_id_idx ON team_stats(id);
    ELSE
        RAISE NOTICE '⏭  team_stats is already a hypertable';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Could not convert team_stats to hypertable: %', SQLERRM;
END $$;

-- ============================================================================
-- VERIFICATION
-- ============================================================================
-- Verify tables and hypertables were created
DO $$
DECLARE
    table_count INTEGER;
    hypertable_count INTEGER;
BEGIN
    -- Check tables exist
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN ('player_stats', 'team_stats');
    
    IF table_count = 2 THEN
        RAISE NOTICE '✓ All 2 statistics tables created successfully';
    ELSE
        RAISE WARNING 'Only % of 2 tables were created', table_count;
    END IF;
    
    -- Check hypertables
    SELECT COUNT(*) INTO hypertable_count
    FROM timescaledb_information.hypertables
    WHERE hypertable_name IN ('player_stats', 'team_stats');
    
    IF hypertable_count = 2 THEN
        RAISE NOTICE '✓ Both tables converted to TimescaleDB hypertables';
    ELSE
        RAISE WARNING 'Only % of 2 tables are hypertables', hypertable_count;
    END IF;
END $$;
