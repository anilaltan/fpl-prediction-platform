"""
Database Migration Runner
Runs SQL migration scripts in order.

Usage:
    python backend/scripts/run_migration.py [migration_file]
    python backend/scripts/run_migration.py  # Runs all pending migrations
"""
import sys
import os
import logging
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import engine  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_migration_files():
    """Get all migration files sorted by name."""
    if not MIGRATIONS_DIR.exists():
        logger.error(f"Migrations directory not found: {MIGRATIONS_DIR}")
        return []

    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    logger.info(f"Found {len(migration_files)} migration files")
    return migration_files


def check_migration_applied(migration_name: str, conn) -> bool:
    """Check if a migration has already been applied."""
    try:
        # Create migrations tracking table if it doesn't exist
        conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                migration_name VARCHAR(255) UNIQUE NOT NULL,
                applied_at TIMESTAMP DEFAULT NOW()
            );
        """
            )
        )
        conn.commit()

        # Check if migration exists
        result = conn.execute(
            text(
                """
            SELECT COUNT(*) FROM schema_migrations 
            WHERE migration_name = :migration_name
        """
            ),
            {"migration_name": migration_name},
        )

        return result.scalar() > 0
    except Exception as e:
        logger.error(f"Error checking migration status: {e}")
        return False


def mark_migration_applied(migration_name: str, conn):
    """Mark a migration as applied."""
    try:
        conn.execute(
            text(
                """
            INSERT INTO schema_migrations (migration_name)
            VALUES (:migration_name)
            ON CONFLICT (migration_name) DO NOTHING
        """
            ),
            {"migration_name": migration_name},
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error marking migration as applied: {e}")


def run_migration(migration_file: Path):
    """Run a single migration file."""
    migration_name = migration_file.name

    try:
        logger.info(f"Reading migration: {migration_name}")
        with open(migration_file, "r") as f:
            sql_content = f.read()

        logger.info("Connecting to database...")
        with engine.connect() as conn:
            # Check if already applied
            if check_migration_applied(migration_name, conn):
                logger.info(f"⏭  Migration {migration_name} already applied, skipping")
                return True

            logger.info(f"Running migration: {migration_name}")

            # Execute migration
            conn.execute(text(sql_content))
            conn.commit()

            # Mark as applied
            mark_migration_applied(migration_name, conn)

            logger.info(f"✓ Migration {migration_name} applied successfully")
            return True

    except OperationalError as e:
        logger.error(f"✗ Database connection error: {e}")
        return False
    except ProgrammingError as e:
        logger.error(f"✗ SQL error in {migration_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error in {migration_name}: {e}")
        return False


def run_all_migrations():
    """Run all pending migrations in order."""
    logger.info("=" * 60)
    logger.info("Database Migration Runner")
    logger.info("=" * 60)
    logger.info("")

    migration_files = get_migration_files()

    if not migration_files:
        logger.warning("No migration files found")
        return 1

    success_count = 0
    for migration_file in migration_files:
        if run_migration(migration_file):
            success_count += 1
        else:
            logger.error(f"Migration failed: {migration_file.name}")
            return 1

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        f"✓ Successfully applied {success_count}/{len(migration_files)} migrations"
    )
    logger.info("=" * 60)
    return 0


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Run specific migration file
        migration_path = Path(sys.argv[1])
        if not migration_path.exists():
            logger.error(f"Migration file not found: {migration_path}")
            return 1

        success = run_migration(migration_path)
        return 0 if success else 1
    else:
        # Run all migrations
        return run_all_migrations()


if __name__ == "__main__":
    sys.exit(main())
