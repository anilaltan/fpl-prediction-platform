-- Migration: 001_create_core_reference_tables.sql
-- Description: Create core reference tables (teams, players, fixtures, entity_mappings)
-- Dependencies: TimescaleDB extension must be installed (subtask 1.1)
-- Date: 2026-01-18

-- ============================================================================
-- TEAMS TABLE
-- ============================================================================
-- Core reference table for Premier League teams
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    short_name VARCHAR(10),
    strength_attack INTEGER,
    strength_defense INTEGER,
    strength_overall INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for team lookups
CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(name);
CREATE INDEX IF NOT EXISTS idx_teams_short_name ON teams(short_name);

COMMENT ON TABLE teams IS 'Core reference table for Premier League teams';
COMMENT ON COLUMN teams.id IS 'FPL team ID (primary key)';
COMMENT ON COLUMN teams.strength_attack IS 'FPL attack strength rating (1-5)';
COMMENT ON COLUMN teams.strength_defense IS 'FPL defense strength rating (1-5)';
COMMENT ON COLUMN teams.strength_overall IS 'FPL overall strength rating (1-5)';

-- ============================================================================
-- PLAYERS TABLE
-- ============================================================================
-- Core reference table for FPL players
-- Note: This updates the existing players table structure to match PRD
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    position VARCHAR(3) NOT NULL CHECK (position IN ('GK', 'DEF', 'MID', 'FWD')),
    team_id INTEGER REFERENCES teams(id),
    price DECIMAL(5,2),
    ownership DECIMAL(5,2),
    canonical_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for player lookups
CREATE INDEX IF NOT EXISTS idx_players_team_id ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_players_position ON players(position);
CREATE INDEX IF NOT EXISTS idx_players_canonical_name ON players(canonical_name);

COMMENT ON TABLE players IS 'Core reference table for FPL players';
COMMENT ON COLUMN players.id IS 'FPL player ID (primary key)';
COMMENT ON COLUMN players.position IS 'Player position: GK (Goalkeeper), DEF (Defender), MID (Midfielder), FWD (Forward)';
COMMENT ON COLUMN players.price IS 'Current player price in FPL (e.g., 8.5 for £8.5m)';
COMMENT ON COLUMN players.ownership IS 'Current ownership percentage';
COMMENT ON COLUMN players.canonical_name IS 'Canonical name for entity resolution across data sources';

-- ============================================================================
-- FIXTURES TABLE
-- ============================================================================
-- Table for storing fixture information (historical and future)
CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY,
    home_team_id INTEGER NOT NULL REFERENCES teams(id),
    away_team_id INTEGER NOT NULL REFERENCES teams(id),
    gameweek INTEGER NOT NULL,
    season VARCHAR(9) NOT NULL,
    kickoff_time TIMESTAMP,
    finished BOOLEAN DEFAULT FALSE,
    home_score INTEGER,
    away_score INTEGER,
    fdr_home DECIMAL(4,2),
    fdr_away DECIMAL(4,2),
    xgc_home DECIMAL(6,3),
    xgc_away DECIMAL(6,3),
    xgs_home DECIMAL(6,3),
    xgs_away DECIMAL(6,3),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for fixture queries
CREATE INDEX IF NOT EXISTS idx_fixtures_gameweek_season ON fixtures(gameweek, season);
CREATE INDEX IF NOT EXISTS idx_fixtures_home_team ON fixtures(home_team_id);
CREATE INDEX IF NOT EXISTS idx_fixtures_away_team ON fixtures(away_team_id);
CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff_time ON fixtures(kickoff_time);

-- Unique constraint: one fixture per team pair per gameweek per season
CREATE UNIQUE INDEX IF NOT EXISTS uq_fixtures_teams_gameweek_season 
    ON fixtures(home_team_id, away_team_id, gameweek, season);

COMMENT ON TABLE fixtures IS 'Fixture information for historical and future matches';
COMMENT ON COLUMN fixtures.fdr_home IS 'Fixture Difficulty Rating for home team';
COMMENT ON COLUMN fixtures.fdr_away IS 'Fixture Difficulty Rating for away team';
COMMENT ON COLUMN fixtures.xgc_home IS 'Expected Goals Conceded for home team';
COMMENT ON COLUMN fixtures.xgc_away IS 'Expected Goals Conceded for away team';
COMMENT ON COLUMN fixtures.xgs_home IS 'Expected Goals Scored for home team';
COMMENT ON COLUMN fixtures.xgs_away IS 'Expected Goals Scored for away team';

-- ============================================================================
-- ENTITY MAPPINGS TABLE
-- ============================================================================
-- Table for entity resolution across different data sources (FPL, Understat, FBref)
CREATE TABLE IF NOT EXISTS entity_mappings (
    id SERIAL PRIMARY KEY,
    fpl_id INTEGER NOT NULL REFERENCES players(id),
    understat_name VARCHAR(255),
    fbref_name VARCHAR(255),
    canonical_name VARCHAR(255) NOT NULL,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    manually_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for entity resolution
CREATE INDEX IF NOT EXISTS idx_entity_mappings_fpl_id ON entity_mappings(fpl_id);
CREATE INDEX IF NOT EXISTS idx_entity_mappings_canonical_name ON entity_mappings(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entity_mappings_understat_name ON entity_mappings(understat_name);
CREATE INDEX IF NOT EXISTS idx_entity_mappings_fbref_name ON entity_mappings(fbref_name);
CREATE INDEX IF NOT EXISTS idx_entity_mappings_manually_verified ON entity_mappings(manually_verified);

-- Unique constraint: one mapping per FPL player
CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_mappings_fpl_id ON entity_mappings(fpl_id);

COMMENT ON TABLE entity_mappings IS 'Entity resolution mappings across FPL, Understat, and FBref data sources';
COMMENT ON COLUMN entity_mappings.confidence_score IS 'Confidence score for automated mapping (0.0-1.0)';
COMMENT ON COLUMN entity_mappings.manually_verified IS 'Whether this mapping has been manually verified';

-- ============================================================================
-- VERIFICATION
-- ============================================================================
-- Verify tables were created
DO $$
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN ('teams', 'players', 'fixtures', 'entity_mappings');
    
    IF table_count = 4 THEN
        RAISE NOTICE '✓ All 4 core reference tables created successfully';
    ELSE
        RAISE WARNING 'Only % of 4 tables were created', table_count;
    END IF;
END $$;
