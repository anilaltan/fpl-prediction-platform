"""
Startup Validation Service
Validates critical dependencies (ML models, database) at API startup to prevent broken deployments.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from app.services.ml.engine import PLEngine
from app.services.ml.model_file_validator import ModelFileValidator, ModelValidationResult
from app.services.database_validator import DatabaseValidator as CoreDatabaseValidator, DatabaseValidationResult

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation check"""

    def __init__(self, name: str, status: str, error_message: Optional[str] = None):
        """
        Initialize validation result.

        Args:
            name: Name of the dependency being validated
            status: "healthy" or "unhealthy"
            error_message: Error message if validation failed
        """
        self.name = name
        self.status = status
        self.error_message = error_message
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/reporting"""
        return {
            "name": self.name,
            "status": self.status,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }

    def is_healthy(self) -> bool:
        """Check if validation passed"""
        return self.status == "healthy"


class ModelValidator:
    """
    Validates ML model files exist and are valid.
    
    Wrapper around ModelFileValidator that converts ModelValidationResult
    to ValidationResult for compatibility with StartupValidator.
    """

    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        checksums: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize model validator.

        Args:
            model_paths: List of required model file paths. If None, uses PLEngine to find latest model.
            checksums: Optional dictionary mapping model paths to expected SHA256 checksums
            timeout: Optional timeout in seconds for file operations
        """
        self.model_paths = model_paths
        self.checksums = checksums or {}
        self.timeout = timeout
        self.model_file_validator = ModelFileValidator(timeout=timeout)

    def validate(self) -> ValidationResult:
        """
        Validate that all required model files exist and are valid.

        Returns:
            ValidationResult with status and error message
        """
        try:
            # If no paths specified, use PLEngine to find the model
            if not self.model_paths:
                pl_engine = PLEngine()
                model_path = pl_engine.model_path
                if not model_path:
                    return ValidationResult(
                        "ML Models",
                        "unhealthy",
                        "No model path found. PLEngine could not locate any model files.",
                    )
                self.model_paths = [model_path]

            # Prepare model configs for ModelFileValidator
            model_configs = []
            for model_path in self.model_paths:
                config = {"path": model_path}
                if model_path in self.checksums:
                    config["checksum"] = self.checksums[model_path]
                model_configs.append(config)

            # Use ModelFileValidator to validate
            validation_results = self.model_file_validator.validate_models(model_configs)

            # Aggregate results
            errors = []
            for result in validation_results:
                if not result.is_valid:
                    error_msg = f"{result.model_path}: {result.error_message}"
                    if result.error_type:
                        error_msg = f"[{result.error_type}] {error_msg}"
                    errors.append(error_msg)

            if errors:
                return ValidationResult(
                    "ML Models",
                    "unhealthy",
                    "; ".join(errors),
                )

            return ValidationResult("ML Models", "healthy")

        except Exception as e:
            logger.exception("Unexpected error during model validation")
            return ValidationResult(
                "ML Models",
                "unhealthy",
                f"Unexpected error during model validation: {str(e)}",
            )


class DatabaseValidator:
    """
    Validates database connection and basic functionality.
    
    Wrapper around CoreDatabaseValidator that converts DatabaseValidationResult
    to ValidationResult for compatibility with StartupValidator.
    """

    def __init__(self, timeout: float = 2.0, database_url: Optional[str] = None):
        """
        Initialize database validator.

        Args:
            timeout: Connection timeout in seconds (default: 2.0)
            database_url: Optional database URL. If None, uses DATABASE_URL from environment.
        """
        self.core_validator = CoreDatabaseValidator(
            database_url=database_url,
            timeout=timeout,
        )

    def validate(self) -> ValidationResult:
        """
        Validate database connection and basic functionality.

        Returns:
            ValidationResult with status and error message
        """
        # Use the core database validator
        db_result = self.core_validator.validate()
        
        # Convert DatabaseValidationResult to ValidationResult
        if db_result.is_valid:
            status = "healthy"
            error_message = None
        else:
            status = "unhealthy"
            # Build detailed error message
            error_parts = [f"Error type: {db_result.error_type}"]
            if db_result.error_message:
                error_parts.append(db_result.error_message)
            if db_result.connection_time_ms is not None:
                error_parts.append(f"Connection time: {db_result.connection_time_ms:.2f}ms")
            error_message = " | ".join(error_parts)
        
        return ValidationResult("Database", status, error_message)


class StartupValidator:
    """
    Main validator that orchestrates all startup validations.
    
    Coordinates model and database validation checks during FastAPI application initialization.
    Ensures all critical dependencies are validated before the API accepts traffic.
    """

    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        model_checksums: Optional[Dict[str, str]] = None,
        db_timeout: float = 2.0,
        database_url: Optional[str] = None,
        model_timeout: Optional[float] = None,
        performance_budget_seconds: float = 5.0,
    ):
        """
        Initialize startup validator.

        Args:
            model_paths: Optional list of model file paths to validate. If None, uses PLEngine to find latest model.
            model_checksums: Optional dictionary mapping model paths to expected checksums
            db_timeout: Database connection timeout in seconds (default: 2.0)
            database_url: Optional database URL. If None, uses DATABASE_URL from environment.
            model_timeout: Optional timeout in seconds for model file operations
            performance_budget_seconds: Maximum time allowed for all validations (default: 5.0)
        """
        self.model_validator = ModelValidator(
            model_paths=model_paths,
            checksums=model_checksums,
            timeout=model_timeout,
        )
        self.db_validator = DatabaseValidator(timeout=db_timeout, database_url=database_url)
        self.performance_budget = performance_budget_seconds

    async def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validations with performance monitoring.
        
        Executes model validation first, then database validation, collecting all results.
        Aggregates validation results and determines overall startup success/failure.
        Ensures validation completes within the performance budget.

        Returns:
            Tuple of (all_healthy: bool, results: List[ValidationResult])
        """
        start_time = time.time()
        results = []

        try:
            # Validate models first
            logger.info("Starting model validation...")
            model_start = time.time()
            model_result = self.model_validator.validate()
            model_time = time.time() - model_start
            results.append(model_result)
            
            logger.info(
                f"Model validation completed in {model_time:.2f}s: {model_result.status} - {model_result.error_message or 'OK'}"
            )

            # Validate database
            logger.info("Starting database validation...")
            db_start = time.time()
            db_result = self.db_validator.validate()
            db_time = time.time() - db_start
            results.append(db_result)
            
            logger.info(
                f"Database validation completed in {db_time:.2f}s: {db_result.status} - {db_result.error_message or 'OK'}"
            )

            # Check performance budget
            total_time = time.time() - start_time
            if total_time > self.performance_budget:
                logger.warning(
                    f"Validation exceeded performance budget: {total_time:.2f}s > {self.performance_budget}s"
                )
            else:
                logger.info(f"All validations completed within budget: {total_time:.2f}s <= {self.performance_budget}s")

            all_healthy = all(result.is_healthy() for result in results)
            return all_healthy, results

        except Exception as e:
            logger.exception("Unexpected error during validation orchestration")
            # Create error result for the failed validation
            error_result = ValidationResult(
                "Validation Orchestrator",
                "unhealthy",
                f"Unexpected error during validation: {str(e)}",
            )
            results.append(error_result)
            return False, results

    def get_validation_report(self, results: List[ValidationResult]) -> str:
        """
        Generate a human-readable validation report with structured details.

        Args:
            results: List of validation results

        Returns:
            Human-readable report string with timestamps and detailed error information
        """
        lines = [
            "Startup Validation Report",
            "=" * 60,
            f"Timestamp: {datetime.utcnow().isoformat()}",
            "",
        ]
        
        for result in results:
            status_symbol = "✓" if result.is_healthy() else "✗"
            status_color = "HEALTHY" if result.is_healthy() else "UNHEALTHY"
            lines.append(f"{status_symbol} {result.name}: {status_color}")
            
            if result.timestamp:
                lines.append(f"  Timestamp: {result.timestamp.isoformat()}")
            
            if result.error_message:
                lines.append(f"  Error: {result.error_message}")
            else:
                lines.append("  Status: OK")
            
            lines.append("")  # Blank line between results
        
        lines.append("=" * 60)
        
        all_healthy = all(r.is_healthy() for r in results)
        overall_status = "HEALTHY" if all_healthy else "UNHEALTHY"
        lines.append(f"Overall Status: {overall_status}")
        
        if not all_healthy:
            failed_count = sum(1 for r in results if not r.is_healthy())
            lines.append(f"Failed Checks: {failed_count} of {len(results)}")

        return "\n".join(lines)
