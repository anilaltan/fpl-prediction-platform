-- Migration: 006_create_third_party_data_cache_table.sql
-- Description: Create third_party_data_cache table for caching Understat/FBref data
-- Purpose: Pre-fetch and cache third-party data to avoid real-time scraping during ML predictions
-- Date: 2026-01-24

-- ============================================================================
-- THIRD_PARTY_DATA_CACHE TABLE
-- ============================================================================
-- Cache for third-party data (Understat/FBref) to avoid real-time scraping
-- Data is pre-fetched in background jobs and used during ML predictions
CREATE TABLE IF NOT EXISTS third_party_data_cache (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL REFERENCES players(id),
    season VARCHAR(9) NOT NULL DEFAULT '2025-26',

    -- Understat metrics
    understat_xg DECIMAL(6,3),
    understat_xa DECIMAL(6,3),
    understat_npxg DECIMAL(6,3),
    understat_xg_per_90 DECIMAL(6,3),
    understat_xa_per_90 DECIMAL(6,3),
    understat_npxg_per_90 DECIMAL(6,3),

    -- FBref defensive metrics
    fbref_blocks INTEGER DEFAULT 0,
    fbref_blocks_per_90 DECIMAL(6,3),
    fbref_interventions INTEGER DEFAULT 0,
    fbref_interventions_per_90 DECIMAL(6,3),
    fbref_tackles INTEGER DEFAULT 0,
    fbref_interceptions INTEGER DEFAULT 0,
    fbref_passes INTEGER DEFAULT 0,
    fbref_passes_per_90 DECIMAL(6,3),

    -- Metadata
    data_source VARCHAR(50),  -- 'understat', 'fbref', 'both'
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_third_party_cache_player_id ON third_party_data_cache(player_id);
CREATE INDEX IF NOT EXISTS idx_third_party_cache_season ON third_party_data_cache(season);
CREATE INDEX IF NOT EXISTS idx_third_party_cache_player_season ON third_party_data_cache(player_id, season);
CREATE INDEX IF NOT EXISTS idx_third_party_cache_last_updated ON third_party_data_cache(last_updated);

-- Unique constraint: one cache entry per player per season
CREATE UNIQUE INDEX IF NOT EXISTS uq_third_party_cache_player_season 
    ON third_party_data_cache(player_id, season);

-- Comments
COMMENT ON TABLE third_party_data_cache IS 'Cache for third-party data (Understat/FBref) to avoid real-time scraping during predictions';
COMMENT ON COLUMN third_party_data_cache.player_id IS 'Foreign key to players.id';
COMMENT ON COLUMN third_party_data_cache.data_source IS 'Source of data: understat, fbref, or both';
COMMENT ON COLUMN third_party_data_cache.confidence_score IS 'Matching confidence score (0.0-1.0) from entity resolution';
COMMENT ON COLUMN third_party_data_cache.last_updated IS 'Timestamp of last data update (used for cache invalidation)';

-- ============================================================================
-- VERIFICATION
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'third_party_data_cache'
    ) THEN
        RAISE NOTICE 'Table third_party_data_cache created successfully';
    ELSE
        RAISE EXCEPTION 'Failed to create third_party_data_cache table';
    END IF;
END $$;
