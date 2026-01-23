"""
Database Initialization Script
Initializes PostgreSQL database with TimescaleDB extension.

This script:
1. Creates the TimescaleDB extension if it doesn't exist
2. Verifies the extension is working correctly
3. Can be run manually or as part of deployment

Usage:
    python backend/scripts/init_database.py
"""
import sys
import os
import logging
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import DATABASE_URL, engine  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_timescaledb():
    """
    Initialize TimescaleDB extension in the database.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Connecting to database...")
        logger.info(
            f"Database URL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'hidden'}"
        )

        with engine.connect() as conn:
            # Check if TimescaleDB extension exists
            logger.info("Checking for TimescaleDB extension...")
            result = conn.execute(
                text(
                    """
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
                ) as extension_exists;
            """
                )
            )
            extension_exists = result.scalar()

            if extension_exists:
                logger.info("✓ TimescaleDB extension already exists")
            else:
                logger.info("Creating TimescaleDB extension...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
                conn.commit()
                logger.info("✓ TimescaleDB extension created successfully")

            # Verify TimescaleDB is working
            logger.info("Verifying TimescaleDB functionality...")
            result = conn.execute(
                text(
                    """
                SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';
            """
                )
            )
            version = result.scalar()

            if version:
                logger.info(f"✓ TimescaleDB version: {version}")
            else:
                logger.error("✗ TimescaleDB extension not found after creation")
                return False

            # Test TimescaleDB functions
            logger.info("Testing TimescaleDB functions...")
            conn.execute(text("SELECT timescaledb_post_restore();"))
            logger.info("✓ TimescaleDB functions are working")

            return True

    except OperationalError as e:
        logger.error(f"✗ Database connection error: {e}")
        logger.error("Make sure the database is running and accessible")
        return False
    except ProgrammingError as e:
        logger.error(f"✗ SQL error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False


def verify_timescaledb():
    """
    Verify TimescaleDB is properly installed and accessible.

    Returns:
        bool: True if verification passes, False otherwise
    """
    try:
        logger.info("Verifying TimescaleDB installation...")

        with engine.connect() as conn:
            # Check extension exists
            result = conn.execute(
                text(
                    """
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
                );
            """
                )
            )
            if not result.scalar():
                logger.error("✗ TimescaleDB extension not found")
                return False

            # Check version
            result = conn.execute(
                text(
                    """
                SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';
            """
                )
            )
            version = result.scalar()
            logger.info(f"✓ TimescaleDB version: {version}")

            # Check hypertables information is accessible
            result = conn.execute(
                text(
                    """
                SELECT COUNT(*) FROM timescaledb_information.hypertables;
            """
                )
            )
            hypertable_count = result.scalar()
            logger.info(f"✓ Current hypertables: {hypertable_count}")

            logger.info("✓ TimescaleDB verification passed")
            return True

    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False


def main():
    """Main function to initialize and verify database."""
    logger.info("=" * 60)
    logger.info("Database Initialization Script")
    logger.info("=" * 60)
    logger.info("")

    # Initialize TimescaleDB
    success = init_timescaledb()

    if success:
        logger.info("")
        logger.info("Verifying installation...")
        verify_success = verify_timescaledb()

        if verify_success:
            logger.info("")
            logger.info("=" * 60)
            logger.info("✓ Database initialization completed successfully!")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("")
            logger.error("=" * 60)
            logger.error("✗ Verification failed")
            logger.error("=" * 60)
            return 1
    else:
        logger.error("")
        logger.error("=" * 60)
        logger.error("✗ Database initialization failed")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
