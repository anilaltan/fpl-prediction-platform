#!/usr/bin/env python3
"""
Standalone Startup Validation Script

Validates critical dependencies (ML models, database) before API deployment.
Can be used in Docker healthchecks, CI/CD pipelines, or manual verification.

Usage:
    # From host machine:
    docker compose exec backend python3 scripts/validate_startup.py
    
    # With verbose output:
    docker compose exec backend python3 scripts/validate_startup.py --verbose
    
    # With custom config:
    docker compose exec backend python3 scripts/validate_startup.py --config /path/to/config.json
    
    # In Docker healthcheck (from Dockerfile):
    CMD ["python3", "scripts/validate_startup.py"]

Exit codes:
    0: All validations passed
    1: One or more validations failed
"""

import asyncio
import argparse
import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.startup_validation import StartupValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {str(e)}")
        sys.exit(1)


def parse_checksums(checksum_string: str) -> dict:
    """
    Parse checksum string into dictionary.

    Format: "path1:checksum1,path2:checksum2"

    Args:
        checksum_string: Checksum string

    Returns:
        Dictionary mapping paths to checksums
    """
    checksums = {}
    if checksum_string:
        for pair in checksum_string.split(","):
            if ":" in pair:
                path, checksum = pair.split(":", 1)
                checksums[path.strip()] = checksum.strip()
    return checksums


async def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(
        description="Validate API startup dependencies (models, database)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file with validation settings",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        help="Explicit model file paths to validate (space-separated)",
    )
    parser.add_argument(
        "--model-checksums",
        type=str,
        help="Model checksums in format 'path1:checksum1,path2:checksum2'",
    )
    parser.add_argument(
        "--db-timeout",
        type=int,
        default=5,
        help="Database connection timeout in seconds (default: 5)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    model_paths = args.model_paths
    model_checksums = None
    db_timeout = args.db_timeout

    if args.config:
        config = load_config(args.config)
        model_paths = config.get("model_paths") or model_paths
        checksum_string = config.get("model_checksums")
        if checksum_string:
            model_checksums = parse_checksums(checksum_string)
        db_timeout = config.get("db_timeout", db_timeout)

    # Parse checksums from command line or environment
    if args.model_checksums:
        model_checksums = parse_checksums(args.model_checksums)
    elif not model_checksums:
        # Try environment variable
        checksum_env = os.getenv("MODEL_CHECKSUMS")
        if checksum_env:
            model_checksums = parse_checksums(checksum_env)

    # Run validation
    logger.info("Starting startup health validation...")
    try:
        validator = StartupValidator(
            model_paths=model_paths,
            model_checksums=model_checksums,
            db_timeout=db_timeout,
        )

        all_healthy, results = await validator.validate_all()

        # Print report
        report = validator.get_validation_report(results)
        print("\n" + report + "\n")

        if all_healthy:
            logger.info("✓ All validations passed. API is ready to start.")
            return 0
        else:
            logger.error("✗ One or more validations failed. API should not start.")
            # Print detailed errors
            for result in results:
                if not result.is_healthy():
                    logger.error(f"  - {result.name}: {result.error_message}")
            return 1

    except Exception as e:
        logger.critical(f"Validation script encountered an error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
