#!/usr/bin/env python3
"""
Standalone Pre-Deployment Validation Script

Validates critical dependencies (ML models, database) before API deployment.
This script can be run independently by operators to verify the deployment
environment without deploying the API.

Usage:
    # Basic usage (uses environment variables):
    docker compose exec backend python3 scripts/validate_deployment.py
    
    # With verbose output:
    docker compose exec backend python3 scripts/validate_deployment.py --verbose
    
    # With custom config file:
    docker compose exec backend python3 scripts/validate_deployment.py --config /path/to/config.json
    
    # With explicit model paths:
    docker compose exec backend python3 scripts/validate_deployment.py --model-paths /app/models/model1.pkl /app/models/model2.pkl
    
    # In CI/CD pipelines (non-interactive):
    docker compose exec backend python3 scripts/validate_deployment.py

Exit codes:
    0: All validations passed - deployment environment is ready
    1: One or more validations failed - deployment should not proceed

Configuration:
    The script loads configuration from:
    1. Command-line arguments (highest priority)
    2. Configuration file (--config flag)
    3. Environment variables (DATABASE_URL, MODEL_PATHS, MODEL_CHECKSUMS, etc.)
    4. Default values (lowest priority)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.startup_validation import ModelValidator, DatabaseValidator, ValidationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DeploymentValidator:
    """
    Standalone deployment validator that orchestrates model and database validation.
    
    This class provides a simple interface for pre-deployment health checks,
    reusing the ModelValidator and DatabaseValidator modules for consistency.
    """

    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        model_checksums: Optional[Dict[str, str]] = None,
        database_url: Optional[str] = None,
        db_timeout: float = 2.0,
        model_timeout: Optional[float] = None,
        verbose: bool = False,
    ):
        """
        Initialize deployment validator.

        Args:
            model_paths: Optional list of model file paths. If None, uses PLEngine to find latest model.
            model_checksums: Optional dictionary mapping model paths to expected SHA256 checksums
            database_url: Optional database URL. If None, uses DATABASE_URL from environment.
            db_timeout: Database connection timeout in seconds (default: 2.0)
            model_timeout: Optional timeout in seconds for model file operations
            verbose: Enable verbose logging output
        """
        self.model_validator = ModelValidator(
            model_paths=model_paths,
            checksums=model_checksums,
            timeout=model_timeout,
        )
        self.db_validator = DatabaseValidator(
            timeout=db_timeout,
            database_url=database_url,
        )
        self.verbose = verbose

    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validations synchronously.

        Returns:
            Tuple of (all_healthy: bool, results: List[ValidationResult])
        """
        results = []

        # Validate models
        if self.verbose:
            logger.info("Starting model validation...")
        model_result = self.model_validator.validate()
        results.append(model_result)
        if self.verbose:
            status_msg = "✓ PASSED" if model_result.is_healthy() else "✗ FAILED"
            logger.info(f"Model validation: {status_msg}")

        # Validate database
        if self.verbose:
            logger.info("Starting database validation...")
        db_result = self.db_validator.validate()
        results.append(db_result)
        if self.verbose:
            status_msg = "✓ PASSED" if db_result.is_healthy() else "✗ FAILED"
            logger.info(f"Database validation: {status_msg}")

        all_healthy = all(result.is_healthy() for result in results)
        return all_healthy, results

    def format_report(self, results: List[ValidationResult]) -> str:
        """
        Generate a human-readable validation report.

        Args:
            results: List of validation results

        Returns:
            Formatted report string with clear pass/fail indicators
        """
        lines = []
        lines.append("=" * 70)
        lines.append("PRE-DEPLOYMENT VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Individual validation results
        for result in results:
            if result.is_healthy():
                status_symbol = "✓"
                status_text = "PASS"
                status_color = "GREEN"
            else:
                status_symbol = "✗"
                status_text = "FAIL"
                status_color = "RED"

            lines.append(f"{status_symbol} {result.name}: {status_text}")
            lines.append(f"  Status: {result.status.upper()}")

            if result.error_message:
                lines.append(f"  Error: {result.error_message}")
                # Add fix instructions based on error type
                fix_instructions = self._get_fix_instructions(result)
                if fix_instructions:
                    lines.append(f"  Fix: {fix_instructions}")
            else:
                lines.append("  Details: All checks passed")

            lines.append("")  # Blank line between results

        # Overall status
        lines.append("-" * 70)
        all_healthy = all(r.is_healthy() for r in results)
        if all_healthy:
            lines.append("✓ OVERALL STATUS: READY FOR DEPLOYMENT")
            lines.append("  All critical dependencies are validated and ready.")
        else:
            failed_count = sum(1 for r in results if not r.is_healthy())
            lines.append("✗ OVERALL STATUS: NOT READY FOR DEPLOYMENT")
            lines.append(f"  {failed_count} of {len(results)} validation(s) failed.")
            lines.append("  Please fix the issues above before deploying.")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _get_fix_instructions(self, result: ValidationResult) -> Optional[str]:
        """
        Generate helpful fix instructions based on validation failure.

        Args:
            result: Validation result that failed

        Returns:
            Fix instruction string or None
        """
        error_msg = result.error_message or ""
        error_lower = error_msg.lower()

        if result.name == "ML Models":
            if "not found" in error_lower or "missing" in error_lower:
                return (
                    "Train models using: docker compose exec backend python3 scripts/train_ml_models.py. "
                    "Or specify model paths with --model-paths flag."
                )
            elif "checksum" in error_lower:
                return (
                    "Model file checksum mismatch. Re-train the model or update the expected checksum "
                    "in MODEL_CHECKSUMS environment variable or config file."
                )
            elif "permission" in error_lower or "unreadable" in error_lower:
                return (
                    "Check file permissions. Ensure the model files are readable by the application user. "
                    "Run: chmod 644 <model_path>"
                )
            else:
                return "Check model file integrity and ensure models are properly trained and saved."

        elif result.name == "Database":
            if "connection refused" in error_lower or "could not connect" in error_lower:
                return (
                    "Ensure the database is running: docker compose up -d db. "
                    "Check DATABASE_URL environment variable is correct."
                )
            elif "authentication" in error_lower or "password" in error_lower:
                return (
                    "Check database credentials in DATABASE_URL environment variable. "
                    "Verify username and password are correct."
                )
            elif "database.*not found" in error_lower or "does not exist" in error_lower:
                return (
                    "Create the database or check the database name in DATABASE_URL. "
                    "Run: docker compose exec db psql -U fpl_user -c 'CREATE DATABASE fpl_db;'"
                )
            elif "timeout" in error_lower:
                return (
                    "Database connection timed out. Check if database is accessible and responsive. "
                    "Increase timeout with --db-timeout flag if needed."
                )
            else:
                return (
                    "Check database connectivity and configuration. "
                    "Verify DATABASE_URL is set correctly and database is running."
                )

        return None


def load_config_file(config_path: str) -> Dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        SystemExit: If config file cannot be loaded
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            logger.info(f"Loaded configuration from: {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        logger.error("Please provide a valid config file path or use environment variables.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {str(e)}")
        logger.error("Please fix the JSON syntax in the config file.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        sys.exit(1)


def parse_checksums(checksum_string: str) -> Dict[str, str]:
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
                parts = pair.split(":", 1)
                if len(parts) == 2:
                    path, checksum = parts[0].strip(), parts[1].strip()
                    checksums[path] = checksum
            else:
                logger.warning(f"Invalid checksum format (expected 'path:checksum'): {pair}")
    return checksums


def load_configuration(args: argparse.Namespace) -> Dict:
    """
    Load configuration from multiple sources with priority order.

    Priority (highest to lowest):
    1. Command-line arguments
    2. Configuration file (if --config provided)
    3. Environment variables
    4. Default values

    Args:
        args: Parsed command-line arguments

    Returns:
        Configuration dictionary with all settings
    """
    config = {}

    # Start with defaults
    config["model_paths"] = None
    config["model_checksums"] = None
    config["database_url"] = None
    config["db_timeout"] = 2.0
    config["model_timeout"] = None

    # Load from config file if provided
    if args.config:
        file_config = load_config_file(args.config)
        config["model_paths"] = file_config.get("model_paths") or config["model_paths"]
        config["model_checksums"] = file_config.get("model_checksums") or config["model_checksums"]
        config["database_url"] = file_config.get("database_url") or config["database_url"]
        config["db_timeout"] = file_config.get("db_timeout", config["db_timeout"])
        config["model_timeout"] = file_config.get("model_timeout") or config["model_timeout"]

    # Override with environment variables
    if not args.model_paths:
        model_paths_env = os.getenv("MODEL_PATHS")
        if model_paths_env:
            config["model_paths"] = [p.strip() for p in model_paths_env.split(",") if p.strip()]

    checksum_env = os.getenv("MODEL_CHECKSUMS")
    if checksum_env and not config.get("model_checksums"):
        config["model_checksums"] = parse_checksums(checksum_env)

    if not config.get("database_url"):
        config["database_url"] = os.getenv("DATABASE_URL")

    db_timeout_env = os.getenv("DB_VALIDATION_TIMEOUT")
    if db_timeout_env:
        try:
            config["db_timeout"] = float(db_timeout_env)
        except ValueError:
            logger.warning(f"Invalid DB_VALIDATION_TIMEOUT value: {db_timeout_env}, using default")

    model_timeout_env = os.getenv("MODEL_VALIDATION_TIMEOUT")
    if model_timeout_env:
        try:
            config["model_timeout"] = float(model_timeout_env)
        except ValueError:
            logger.warning(f"Invalid MODEL_VALIDATION_TIMEOUT value: {model_timeout_env}")

    # Override with command-line arguments (highest priority)
    if args.model_paths:
        config["model_paths"] = args.model_paths

    if args.model_checksums:
        config["model_checksums"] = parse_checksums(args.model_checksums)

    if args.database_url:
        config["database_url"] = args.database_url

    if args.db_timeout:
        config["db_timeout"] = args.db_timeout

    if args.model_timeout:
        config["model_timeout"] = args.model_timeout

    return config


def main() -> int:
    """
    Main entry point for the validation script.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate deployment environment before API deployment. "
            "Checks ML models and database connectivity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed logging",
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to JSON configuration file (optional)",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        metavar="PATH",
        help="Explicit model file paths to validate (space-separated)",
    )
    parser.add_argument(
        "--model-checksums",
        type=str,
        metavar="STRING",
        help="Model checksums in format 'path1:checksum1,path2:checksum2'",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        metavar="URL",
        help="Database connection URL (overrides DATABASE_URL environment variable)",
    )
    parser.add_argument(
        "--db-timeout",
        type=float,
        metavar="SECONDS",
        help="Database connection timeout in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--model-timeout",
        type=float,
        metavar="SECONDS",
        help="Model file operation timeout in seconds (optional, for network filesystems)",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    # Load configuration
    try:
        config = load_configuration(args)
        if args.verbose:
            logger.debug(f"Configuration loaded: {config}")
    except Exception as e:
        logger.critical(f"Failed to load configuration: {str(e)}", exc_info=True)
        return 1

    # Validate configuration
    if not config.get("database_url") and not os.getenv("DATABASE_URL"):
        logger.error("Database URL not configured.")
        logger.error("Please set DATABASE_URL environment variable or use --database-url flag.")
        logger.error("Example: export DATABASE_URL='postgresql://user:pass@host:5432/dbname'")
        return 1

    # Initialize validator
    try:
        validator = DeploymentValidator(
            model_paths=config["model_paths"],
            model_checksums=config["model_checksums"],
            database_url=config["database_url"],
            db_timeout=config["db_timeout"],
            model_timeout=config["model_timeout"],
            verbose=args.verbose,
        )
    except Exception as e:
        logger.critical(f"Failed to initialize validator: {str(e)}", exc_info=True)
        return 1

    # Run validations
    logger.info("Starting pre-deployment validation...")
    try:
        all_healthy, results = validator.validate_all()

        # Print report
        report = validator.format_report(results)
        print("\n" + report + "\n")

        if all_healthy:
            logger.info("✓ All validations passed. Deployment environment is ready.")
            return 0
        else:
            logger.error("✗ One or more validations failed. Deployment should not proceed.")
            if args.verbose:
                for result in results:
                    if not result.is_healthy():
                        logger.error(f"  Failed: {result.name} - {result.error_message}")
            return 1

    except KeyboardInterrupt:
        logger.warning("Validation interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.critical(f"Validation script encountered an unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
