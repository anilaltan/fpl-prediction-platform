-- Migration: 004_add_unique_indexes.sql
-- Description: Add unique indexes to prevent duplicate entries for the same player/team within a specific gameweek and season
-- Dependencies: 
--   - Statistics tables created and converted to hypertables (subtask 1.3, 1.4)
-- Date: 2026-01-18

-- ============================================================================
-- UNIQUE INDEXES FOR INTEGRITY CONSTRAINTS
-- ============================================================================
-- Note: TimescaleDB hypertables require unique constraints/indexes to include 
-- the partitioning column (timestamp). Since we cannot create a unique index
-- on (player_id, gameweek, season) without timestamp, we use a combination of:
-- 1. A non-unique index for fast lookups on (player_id, gameweek, season)
-- 2. A trigger function to enforce uniqueness at insert/update time
-- 3. The existing unique index with timestamp for exact duplicate prevention

-- Create non-unique indexes for fast lookups (these don't enforce uniqueness but help with queries)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'player_stats' 
        AND indexname = 'idx_player_stats_player_gameweek_season_lookup'
    ) THEN
        CREATE INDEX idx_player_stats_player_gameweek_season_lookup 
            ON player_stats(player_id, gameweek, season);
        RAISE NOTICE '✓ Created lookup index on player_stats(player_id, gameweek, season)';
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'team_stats' 
        AND indexname = 'idx_team_stats_team_gameweek_season_lookup'
    ) THEN
        CREATE INDEX idx_team_stats_team_gameweek_season_lookup 
            ON team_stats(team_id, gameweek, season);
        RAISE NOTICE '✓ Created lookup index on team_stats(team_id, gameweek, season)';
    END IF;
END $$;

-- ============================================================================
-- TRIGGER FUNCTIONS FOR UNIQUENESS ENFORCEMENT
-- ============================================================================
-- Create trigger functions to prevent duplicate entries for the same
-- (player_id, gameweek, season) or (team_id, gameweek, season) combination

-- Function to check for duplicate player_stats entries
CREATE OR REPLACE FUNCTION check_player_stats_uniqueness()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if another record exists with the same player_id, gameweek, season
    -- (excluding the current record being updated)
    IF EXISTS (
        SELECT 1 FROM player_stats 
        WHERE player_id = NEW.player_id 
        AND gameweek = NEW.gameweek 
        AND season = NEW.season
        AND (TG_OP = 'INSERT' OR id != NEW.id)  -- Exclude current record on update
    ) THEN
        RAISE EXCEPTION 'Duplicate entry: Player % already has statistics for gameweek %, season %', 
            NEW.player_id, NEW.gameweek, NEW.season;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to check for duplicate team_stats entries
CREATE OR REPLACE FUNCTION check_team_stats_uniqueness()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if another record exists with the same team_id, gameweek, season
    -- (excluding the current record being updated)
    IF EXISTS (
        SELECT 1 FROM team_stats 
        WHERE team_id = NEW.team_id 
        AND gameweek = NEW.gameweek 
        AND season = NEW.season
        AND (TG_OP = 'INSERT' OR id != NEW.id)  -- Exclude current record on update
    ) THEN
        RAISE EXCEPTION 'Duplicate entry: Team % already has statistics for gameweek %, season %', 
            NEW.team_id, NEW.gameweek, NEW.season;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CREATE TRIGGERS
-- ============================================================================
-- Create triggers to enforce uniqueness before insert or update

DO $$
BEGIN
    -- Drop existing trigger if it exists
    DROP TRIGGER IF EXISTS trigger_check_player_stats_uniqueness ON player_stats;
    
    -- Create trigger for player_stats
    CREATE TRIGGER trigger_check_player_stats_uniqueness
        BEFORE INSERT OR UPDATE ON player_stats
        FOR EACH ROW
        EXECUTE FUNCTION check_player_stats_uniqueness();
    
    RAISE NOTICE '✓ Created trigger to enforce uniqueness on player_stats';
END $$;

DO $$
BEGIN
    -- Drop existing trigger if it exists
    DROP TRIGGER IF EXISTS trigger_check_team_stats_uniqueness ON team_stats;
    
    -- Create trigger for team_stats
    CREATE TRIGGER trigger_check_team_stats_uniqueness
        BEFORE INSERT OR UPDATE ON team_stats
        FOR EACH ROW
        EXECUTE FUNCTION check_team_stats_uniqueness();
    
    RAISE NOTICE '✓ Created trigger to enforce uniqueness on team_stats';
END $$;

-- ============================================================================
-- VERIFICATION
-- ============================================================================
-- Verify indexes and triggers were created successfully
DO $$
DECLARE
    player_stats_index_exists BOOLEAN := FALSE;
    team_stats_index_exists BOOLEAN := FALSE;
    player_stats_trigger_exists BOOLEAN := FALSE;
    team_stats_trigger_exists BOOLEAN := FALSE;
BEGIN
    -- Check player_stats lookup index
    SELECT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'player_stats' 
        AND indexname = 'idx_player_stats_player_gameweek_season_lookup'
    ) INTO player_stats_index_exists;
    
    IF player_stats_index_exists THEN
        RAISE NOTICE '✓ Lookup index idx_player_stats_player_gameweek_season_lookup exists';
    ELSE
        RAISE WARNING '✗ Lookup index idx_player_stats_player_gameweek_season_lookup not found';
    END IF;
    
    -- Check team_stats lookup index
    SELECT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'team_stats' 
        AND indexname = 'idx_team_stats_team_gameweek_season_lookup'
    ) INTO team_stats_index_exists;
    
    IF team_stats_index_exists THEN
        RAISE NOTICE '✓ Lookup index idx_team_stats_team_gameweek_season_lookup exists';
    ELSE
        RAISE WARNING '✗ Lookup index idx_team_stats_team_gameweek_season_lookup not found';
    END IF;
    
    -- Check player_stats trigger
    SELECT EXISTS (
        SELECT 1 FROM pg_trigger 
        WHERE tgname = 'trigger_check_player_stats_uniqueness'
    ) INTO player_stats_trigger_exists;
    
    IF player_stats_trigger_exists THEN
        RAISE NOTICE '✓ Trigger trigger_check_player_stats_uniqueness exists';
    ELSE
        RAISE WARNING '✗ Trigger trigger_check_player_stats_uniqueness not found';
    END IF;
    
    -- Check team_stats trigger
    SELECT EXISTS (
        SELECT 1 FROM pg_trigger 
        WHERE tgname = 'trigger_check_team_stats_uniqueness'
    ) INTO team_stats_trigger_exists;
    
    IF team_stats_trigger_exists THEN
        RAISE NOTICE '✓ Trigger trigger_check_team_stats_uniqueness exists';
    ELSE
        RAISE WARNING '✗ Trigger trigger_check_team_stats_uniqueness not found';
    END IF;
    
    -- Summary
    IF player_stats_index_exists AND team_stats_index_exists 
       AND player_stats_trigger_exists AND team_stats_trigger_exists THEN
        RAISE NOTICE '✓ All indexes and triggers created successfully';
    ELSE
        RAISE WARNING 'Some indexes or triggers may not have been created';
    END IF;
END $$;

-- ============================================================================
-- COMMENTS
-- ============================================================================
-- Add comments to indexes and functions
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'player_stats' 
        AND indexname = 'idx_player_stats_player_gameweek_season_lookup'
    ) THEN
        COMMENT ON INDEX idx_player_stats_player_gameweek_season_lookup IS 
            'Lookup index for fast queries on (player_id, gameweek, season). Uniqueness enforced by trigger.';
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'team_stats' 
        AND indexname = 'idx_team_stats_team_gameweek_season_lookup'
    ) THEN
        COMMENT ON INDEX idx_team_stats_team_gameweek_season_lookup IS 
            'Lookup index for fast queries on (team_id, gameweek, season). Uniqueness enforced by trigger.';
    END IF;
    
    COMMENT ON FUNCTION check_player_stats_uniqueness() IS 
        'Trigger function to enforce uniqueness on (player_id, gameweek, season) for player_stats table';
    
    COMMENT ON FUNCTION check_team_stats_uniqueness() IS 
        'Trigger function to enforce uniqueness on (team_id, gameweek, season) for team_stats table';
END $$;
