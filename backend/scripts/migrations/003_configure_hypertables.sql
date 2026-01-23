-- Migration: 003_configure_hypertables.sql
-- Description: Configure TimescaleDB hypertables with optimal settings
-- Dependencies: 
--   - Statistics tables created and converted to hypertables (subtask 1.3)
-- Date: 2026-01-18

-- ============================================================================
-- HYPERTABLE CONFIGURATION
-- ============================================================================
-- Configure chunk intervals for optimal partitioning
-- For FPL data: gameweeks are weekly, so 7 days is appropriate
-- This allows efficient querying by time ranges while keeping chunks manageable

-- Set chunk interval for player_stats (7 days = 1 week)
-- This aligns with gameweek frequency
DO $$
BEGIN
    -- Get current chunk interval
    DECLARE
        current_interval INTERVAL;
        target_interval INTERVAL := INTERVAL '7 days';
    BEGIN
        SELECT time_interval INTO current_interval
        FROM timescaledb_information.dimensions
        WHERE hypertable_name = 'player_stats' AND dimension_type = 'Time';
        
        -- Only update if different from target
        IF current_interval IS NULL OR current_interval != target_interval THEN
            PERFORM set_chunk_time_interval('player_stats', target_interval);
            RAISE NOTICE '✓ Set player_stats chunk interval to 7 days';
        ELSE
            RAISE NOTICE '⏭  player_stats chunk interval already set to 7 days';
        END IF;
    END;
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Could not set chunk interval for player_stats: %', SQLERRM;
END $$;

-- Set chunk interval for team_stats (7 days = 1 week)
DO $$
BEGIN
    DECLARE
        current_interval INTERVAL;
        target_interval INTERVAL := INTERVAL '7 days';
    BEGIN
        SELECT time_interval INTO current_interval
        FROM timescaledb_information.dimensions
        WHERE hypertable_name = 'team_stats' AND dimension_type = 'Time';
        
        IF current_interval IS NULL OR current_interval != target_interval THEN
            PERFORM set_chunk_time_interval('team_stats', target_interval);
            RAISE NOTICE '✓ Set team_stats chunk interval to 7 days';
        ELSE
            RAISE NOTICE '⏭  team_stats chunk interval already set to 7 days';
        END IF;
    END;
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Could not set chunk interval for team_stats: %', SQLERRM;
END $$;

-- ============================================================================
-- COMPRESSION POLICIES
-- ============================================================================
-- Enable compression for older data to save storage space
-- Compress data older than 30 days (approximately 4-5 gameweeks)
-- This keeps recent data uncompressed for fast queries while compressing historical data

-- Enable compression on player_stats
DO $$
BEGIN
    -- Check if compression is already enabled
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs
        WHERE proc_name = 'policy_compression'
        AND hypertable_name = 'player_stats'
    ) THEN
        -- Add compression policy: compress chunks older than 30 days
        PERFORM add_compression_policy('player_stats', INTERVAL '30 days');
        RAISE NOTICE '✓ Added compression policy for player_stats (30 days)';
    ELSE
        RAISE NOTICE '⏭  Compression policy already exists for player_stats';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Could not add compression policy for player_stats: %', SQLERRM;
END $$;

-- Enable compression on team_stats
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs
        WHERE proc_name = 'policy_compression'
        AND hypertable_name = 'team_stats'
    ) THEN
        PERFORM add_compression_policy('team_stats', INTERVAL '30 days');
        RAISE NOTICE '✓ Added compression policy for team_stats (30 days)';
    ELSE
        RAISE NOTICE '⏭  Compression policy already exists for team_stats';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Could not add compression policy for team_stats: %', SQLERRM;
END $$;

-- ============================================================================
-- COMPRESSION SETTINGS
-- ============================================================================
-- Configure which columns to compress and compression algorithm
-- Compress all numeric and integer columns for maximum space savings

-- Enable compression on hypertables with segment-by configuration
-- Segment-by columns are used to group related data together for better compression
DO $$
BEGIN
    -- Check if compression is already enabled
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'player_stats' AND compression_enabled = TRUE
    ) THEN
        -- Enable compression with segment-by columns
        -- This groups data by player_id, gameweek, season for better compression
        ALTER TABLE player_stats SET (
            timescaledb.compress = true,
            timescaledb.compress_segmentby = 'player_id, gameweek, season'
        );
        RAISE NOTICE '✓ Enabled compression for player_stats with segment-by columns';
    ELSE
        RAISE NOTICE '⏭  Compression already enabled for player_stats';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Could not enable compression for player_stats: %', SQLERRM;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'team_stats' AND compression_enabled = TRUE
    ) THEN
        ALTER TABLE team_stats SET (
            timescaledb.compress = true,
            timescaledb.compress_segmentby = 'team_id, gameweek, season'
        );
        RAISE NOTICE '✓ Enabled compression for team_stats with segment-by columns';
    ELSE
        RAISE NOTICE '⏭  Compression already enabled for team_stats';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Could not enable compression for team_stats: %', SQLERRM;
END $$;

-- ============================================================================
-- VERIFICATION
-- ============================================================================
-- Verify hypertable configuration
DO $$
DECLARE
    player_stats_configured BOOLEAN := FALSE;
    team_stats_configured BOOLEAN := FALSE;
    player_chunk_interval INTERVAL;
    team_chunk_interval INTERVAL;
BEGIN
    -- Check player_stats configuration
    SELECT 
        time_interval INTO player_chunk_interval
    FROM timescaledb_information.dimensions
    WHERE hypertable_name = 'player_stats' AND dimension_type = 'Time';
    
    IF player_chunk_interval IS NOT NULL THEN
        player_stats_configured := TRUE;
        RAISE NOTICE '✓ player_stats: chunk interval = %', player_chunk_interval;
    END IF;
    
    -- Check team_stats configuration
    SELECT 
        time_interval INTO team_chunk_interval
    FROM timescaledb_information.dimensions
    WHERE hypertable_name = 'team_stats' AND dimension_type = 'Time';
    
    IF team_chunk_interval IS NOT NULL THEN
        team_stats_configured := TRUE;
        RAISE NOTICE '✓ team_stats: chunk interval = %', team_chunk_interval;
    END IF;
    
    -- Summary
    IF player_stats_configured AND team_stats_configured THEN
        RAISE NOTICE '✓ Both hypertables configured successfully';
    ELSE
        RAISE WARNING 'Some hypertables may not be fully configured';
    END IF;
END $$;

COMMENT ON TABLE player_stats IS 'TimescaleDB hypertable optimized for time-series queries with 7-day chunk intervals and 30-day compression policy';
COMMENT ON TABLE team_stats IS 'TimescaleDB hypertable optimized for time-series queries with 7-day chunk intervals and 30-day compression policy';
