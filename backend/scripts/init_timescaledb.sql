-- Initialize TimescaleDB Extension
-- This script is automatically run when the database container is first created
-- It enables the TimescaleDB extension for time-series optimization

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Verify TimescaleDB is installed and working
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
    ) THEN
        RAISE EXCEPTION 'TimescaleDB extension failed to install';
    END IF;
    
    RAISE NOTICE 'TimescaleDB extension successfully enabled';
END $$;
