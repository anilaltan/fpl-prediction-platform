-- Migration: 005_create_market_intelligence_table.sql
-- Description: Create market_intelligence table for ownership arbitrage analysis
-- Dependencies: players table (001_create_core_reference_tables.sql)
-- Date: 2026-01-21

-- ============================================================================
-- MARKET INTELLIGENCE TABLE
-- ============================================================================
-- Table for storing ownership arbitrage analysis results
-- Identifies overvalued and undervalued assets based on ownership trends and xP
CREATE TABLE IF NOT EXISTS market_intelligence (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL REFERENCES players(id),
    gameweek INTEGER NOT NULL,
    season VARCHAR(9) NOT NULL DEFAULT '2025-26',
    
    -- Ranking metrics
    xp_rank INTEGER NOT NULL,  -- Rank by expected points (descending: highest xP = rank 1)
    ownership_rank INTEGER NOT NULL,  -- Rank by ownership percentage (descending: highest ownership = rank 1)
    
    -- Arbitrage score: (xp_rank - ownership_rank)
    -- Negative = Differential (high xP, low ownership)
    -- Positive = Overvalued (low xP, high ownership)
    arbitrage_score FLOAT NOT NULL,
    
    -- Category: 'Differential', 'Overvalued', or 'Neutral'
    category VARCHAR(50) NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for market intelligence queries
CREATE INDEX IF NOT EXISTS idx_market_intelligence_player_id ON market_intelligence(player_id);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_gameweek ON market_intelligence(gameweek);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_season ON market_intelligence(season);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_gameweek_season ON market_intelligence(gameweek, season);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_category ON market_intelligence(category);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_arbitrage_score ON market_intelligence(arbitrage_score);

-- Unique constraint: one record per player per gameweek per season
CREATE UNIQUE INDEX IF NOT EXISTS uq_market_intelligence_player_gameweek_season 
    ON market_intelligence(player_id, gameweek, season);

-- Comments for documentation
COMMENT ON TABLE market_intelligence IS 'Market Intelligence table - stores ownership arbitrage analysis results';
COMMENT ON COLUMN market_intelligence.xp_rank IS 'Rank by expected points (descending: highest xP = rank 1)';
COMMENT ON COLUMN market_intelligence.ownership_rank IS 'Rank by ownership percentage (descending: highest ownership = rank 1)';
COMMENT ON COLUMN market_intelligence.arbitrage_score IS 'Arbitrage score: (xp_rank - ownership_rank). Negative = Differential, Positive = Overvalued';
COMMENT ON COLUMN market_intelligence.category IS 'Category: Differential (low ownership, high xP), Overvalued (high ownership, low xP), or Neutral';

-- ============================================================================
-- VERIFICATION
-- ============================================================================
-- Verify table was created
DO $$
DECLARE
    table_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'market_intelligence'
    ) INTO table_exists;
    
    IF table_exists THEN
        RAISE NOTICE '✓ market_intelligence table created successfully';
    ELSE
        RAISE WARNING 'market_intelligence table was not created';
    END IF;
END $$;
