"""
Model File Validator
Focused validation module for ML model files with existence and checksum verification.
Independent of FastAPI/startup context for easy testing and reuse.
"""

import os
import hashlib
import logging
import signal
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelValidationResult:
    """
    Result of validating a single model file.
    
    Attributes:
        model_path: Path to the model file that was validated
        is_valid: True if validation passed, False otherwise
        error_type: Type of error if validation failed (e.g., "missing", "checksum_mismatch", "empty", "unreadable", "permission_error", "symlink", "timeout")
        error_message: Detailed error message if validation failed
        file_size: Size of the file in bytes (None if file doesn't exist)
        actual_checksum: Calculated SHA-256 checksum (None if checksum wasn't calculated)
        expected_checksum: Expected SHA-256 checksum from configuration (None if not provided)
        timestamp: When the validation was performed
    """
    model_path: str
    is_valid: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    file_size: Optional[int] = None
    actual_checksum: Optional[str] = None
    expected_checksum: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/reporting."""
        return {
            "model_path": self.model_path,
            "is_valid": self.is_valid,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "file_size": self.file_size,
            "actual_checksum": self.actual_checksum,
            "expected_checksum": self.expected_checksum,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelFileValidator:
    """
    Validates ML model files for existence and integrity using checksums.
    
    This module is independent of FastAPI/startup context and can be used
    in standalone scripts, tests, or during API startup.
    """

    def __init__(self, timeout: Optional[float] = None):
        """
        Initialize model file validator.
        
        Args:
            timeout: Optional timeout in seconds for file operations.
                    If None, no timeout is enforced. Useful for network-mounted filesystems.
        """
        self.timeout = timeout

    def validate_models(
        self,
        model_configs: List[Dict[str, str]],
    ) -> List[ModelValidationResult]:
        """
        Validate multiple model files.
        
        Args:
            model_configs: List of model configurations. Each config should be a dict with:
                - "path" (required): File path to the model file
                - "checksum" (optional): Expected SHA-256 checksum as hex string
                
        Returns:
            List of ModelValidationResult objects, one per model configuration.
            All validations are performed even if some fail, allowing complete reporting.
            
        Example:
            >>> validator = ModelFileValidator(timeout=30.0)
            >>> configs = [
            ...     {"path": "/app/models/model1.pkl", "checksum": "abc123..."},
            ...     {"path": "/app/models/model2.pkl"},
            ... ]
            >>> results = validator.validate_models(configs)
            >>> for result in results:
            ...     print(f"{result.model_path}: {'OK' if result.is_valid else result.error_message}")
        """
        results = []
        
        for config in model_configs:
            model_path = config.get("path")
            expected_checksum = config.get("checksum")
            
            if not model_path:
                results.append(ModelValidationResult(
                    model_path="<unknown>",
                    is_valid=False,
                    error_type="invalid_config",
                    error_message="Model configuration missing 'path' field",
                ))
                continue
            
            result = self._validate_single_model(model_path, expected_checksum)
            results.append(result)
        
        return results

    def _validate_single_model(
        self,
        model_path: str,
        expected_checksum: Optional[str] = None,
    ) -> ModelValidationResult:
        """
        Validate a single model file.
        
        Args:
            model_path: Path to the model file
            expected_checksum: Optional expected SHA-256 checksum as hex string
            
        Returns:
            ModelValidationResult with detailed validation information
        """
        # Validate checksum format if provided
        if expected_checksum is not None:
            checksum_validation = self._validate_checksum_format(expected_checksum)
            if not checksum_validation["valid"]:
                return ModelValidationResult(
                    model_path=model_path,
                    is_valid=False,
                    error_type="invalid_checksum_format",
                    error_message=checksum_validation["error"],
                    expected_checksum=expected_checksum,
                )
        
        # Check file existence
        if not os.path.exists(model_path):
            return ModelValidationResult(
                model_path=model_path,
                is_valid=False,
                error_type="missing",
                error_message=f"Model file not found: {model_path}",
            )
        
        # Check if it's a symlink (we want actual files, not symlinks)
        if os.path.islink(model_path):
            return ModelValidationResult(
                model_path=model_path,
                is_valid=False,
                error_type="symlink",
                error_message=f"Model file is a symlink, not a regular file: {model_path}",
                file_size=os.path.getsize(model_path) if os.path.exists(model_path) else None,
            )
        
        # Check file size (empty files are invalid)
        try:
            file_size = os.path.getsize(model_path)
            if file_size == 0:
                return ModelValidationResult(
                    model_path=model_path,
                    is_valid=False,
                    error_type="empty",
                    error_message=f"Model file is empty (0 bytes): {model_path}",
                    file_size=0,
                )
        except OSError as e:
            return ModelValidationResult(
                model_path=model_path,
                is_valid=False,
                error_type="permission_error",
                error_message=f"Cannot access model file size: {model_path} - {str(e)}",
            )
        
        # Check file readability and calculate checksum
        try:
            actual_checksum = self._calculate_checksum_with_timeout(model_path)
        except PermissionError as e:
            return ModelValidationResult(
                model_path=model_path,
                is_valid=False,
                error_type="permission_error",
                error_message=f"Permission denied reading model file: {model_path} - {str(e)}",
                file_size=file_size,
            )
        except TimeoutError as e:
            return ModelValidationResult(
                model_path=model_path,
                is_valid=False,
                error_type="timeout",
                error_message=f"Timeout reading model file: {model_path} - {str(e)}",
                file_size=file_size,
            )
        except IOError as e:
            return ModelValidationResult(
                model_path=model_path,
                is_valid=False,
                error_type="unreadable",
                error_message=f"Model file is not readable: {model_path} - {str(e)}",
                file_size=file_size,
            )
        except Exception as e:
            return ModelValidationResult(
                model_path=model_path,
                is_valid=False,
                error_type="unexpected_error",
                error_message=f"Unexpected error validating model file: {model_path} - {str(e)}",
                file_size=file_size,
            )
        
        # Verify checksum if expected value provided
        if expected_checksum is not None:
            if actual_checksum.lower() != expected_checksum.lower():
                return ModelValidationResult(
                    model_path=model_path,
                    is_valid=False,
                    error_type="checksum_mismatch",
                    error_message=(
                        f"Model checksum mismatch for {model_path}: "
                        f"expected {expected_checksum}, got {actual_checksum}"
                    ),
                    file_size=file_size,
                    actual_checksum=actual_checksum,
                    expected_checksum=expected_checksum,
                )
        
        # All checks passed
        return ModelValidationResult(
            model_path=model_path,
            is_valid=True,
            file_size=file_size,
            actual_checksum=actual_checksum,
            expected_checksum=expected_checksum,
        )

    def _validate_checksum_format(self, checksum: str) -> Dict[str, any]:
        """
        Validate that a checksum string is a valid hex string.
        
        Args:
            checksum: Checksum string to validate
            
        Returns:
            Dict with "valid" (bool) and "error" (str, if invalid) keys
        """
        if not isinstance(checksum, str):
            return {
                "valid": False,
                "error": f"Checksum must be a string, got {type(checksum).__name__}",
            }
        
        if not checksum:
            return {
                "valid": False,
                "error": "Checksum cannot be empty",
            }
        
        # SHA-256 produces 64 hex characters
        if len(checksum) != 64:
            return {
                "valid": False,
                "error": f"Checksum must be 64 hex characters (SHA-256), got {len(checksum)} characters",
            }
        
        # Check that all characters are valid hex
        try:
            int(checksum, 16)
        except ValueError:
            return {
                "valid": False,
                "error": f"Checksum contains invalid hex characters: {checksum}",
            }
        
        return {"valid": True}

    def _calculate_checksum_with_timeout(self, file_path: str) -> str:
        """
        Calculate SHA-256 checksum of a file with optional timeout.
        
        Uses streaming reads to handle large files efficiently.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA-256 checksum as hex string
            
        Raises:
            PermissionError: If file cannot be read due to permissions
            TimeoutError: If operation exceeds timeout
            IOError: If file cannot be read
        """
        if self.timeout is not None:
            # Use signal-based timeout (Unix only)
            # For Windows, we'll rely on the file operation itself
            if hasattr(signal, "SIGALRM"):
                # Unix/Linux: Use signal-based timeout
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Checksum calculation exceeded timeout of {self.timeout}s")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))
                try:
                    checksum = self._calculate_checksum_streaming(file_path)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                return checksum
            else:
                # Windows: No signal support, but we can still use streaming
                # The timeout would need to be handled at a higher level
                logger.warning(
                    "Timeout requested on Windows platform. Signal-based timeout not available. "
                    "Consider handling timeout at a higher level."
                )
        
        return self._calculate_checksum_streaming(file_path)

    @staticmethod
    def _calculate_checksum_streaming(file_path: str) -> str:
        """
        Calculate SHA-256 checksum using streaming reads.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA-256 checksum as hex string
            
        Raises:
            PermissionError: If file cannot be read due to permissions
            IOError: If file cannot be read
        """
        sha256_hash = hashlib.sha256()
        chunk_size = 4096  # 4KB chunks for efficient streaming
        
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    sha256_hash.update(chunk)
        except PermissionError:
            raise
        except IOError as e:
            raise IOError(f"Cannot read file {file_path}: {str(e)}") from e
        
        return sha256_hash.hexdigest()
