"""
Centralized Exception Handling for FPL Prediction Platform
Provides standardized error responses across the API.
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class AppException(Exception):
    """
    Base exception class for application-specific errors.
    
    All custom exceptions should inherit from this class to ensure
    consistent error handling and response formatting.
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize application exception.
        
        Args:
            message: Human-readable error message
            status_code: HTTP status code (default: 500)
            error_code: Machine-readable error code (e.g., "VALIDATION_ERROR")
            details: Additional error details dictionary
        """
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "INTERNAL_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(AppException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="VALIDATION_ERROR",
            details=details
        )


class NotFoundError(AppException):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource: str, identifier: Optional[str] = None):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            details={"resource": resource, "identifier": identifier}
        )


class DatabaseError(AppException):
    """Raised when a database operation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="DATABASE_ERROR",
            details=details
        )


class ExternalAPIError(AppException):
    """Raised when an external API call fails."""
    
    def __init__(self, service: str, message: str, details: Optional[Dict[str, Any]] = None):
        full_message = f"External API error ({service}): {message}"
        super().__init__(
            message=full_message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code="EXTERNAL_API_ERROR",
            details={"service": service, **details} if details else {"service": service}
        )


class ModelError(AppException):
    """Raised when ML model operations fail."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        full_message = f"Model error"
        if model_name:
            full_message += f" ({model_name})"
        full_message += f": {message}"
        super().__init__(
            message=full_message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="MODEL_ERROR",
            details={"model_name": model_name, **details} if details else {"model_name": model_name}
        )


class RateLimitError(AppException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_ERROR",
            details=details
        )


def create_error_response(
    exception: AppException,
    include_traceback: bool = False
) -> JSONResponse:
    """
    Create standardized error response from AppException.
    
    Args:
        exception: AppException instance
        include_traceback: Whether to include traceback in response (default: False for security)
    
    Returns:
        JSONResponse with standardized error format
    """
    response_data = {
        "error": {
            "code": exception.error_code,
            "message": exception.message,
            "status_code": exception.status_code
        }
    }
    
    # Add details if present
    if exception.details:
        response_data["error"]["details"] = exception.details
    
    # Include traceback only in development/debug mode
    if include_traceback:
        import traceback
        response_data["error"]["traceback"] = traceback.format_exc()
    
    return JSONResponse(
        status_code=exception.status_code,
        content=response_data
    )


def handle_app_exception(exception: AppException) -> JSONResponse:
    """
    Handle AppException and return standardized response.
    
    Args:
        exception: AppException instance
    
    Returns:
        JSONResponse with error details
    """
    logger.error(
        f"AppException: {exception.error_code} - {exception.message}",
        extra={"error_code": exception.error_code, "details": exception.details}
    )
    return create_error_response(exception, include_traceback=False)


def handle_generic_exception(exception: Exception) -> JSONResponse:
    """
    Handle generic exceptions and convert to standardized format.
    
    Args:
        exception: Generic Exception instance
    
    Returns:
        JSONResponse with error details
    """
    logger.error(f"Unhandled exception: {str(exception)}", exc_info=True)
    
    # Convert to AppException
    app_exception = AppException(
        message="An unexpected error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_ERROR",
        details={"original_error": str(exception)}
    )
    
    return create_error_response(app_exception, include_traceback=False)


def handle_http_exception(exception: HTTPException) -> JSONResponse:
    """
    Handle FastAPI HTTPException and convert to standardized format.
    
    Args:
        exception: HTTPException instance
    
    Returns:
        JSONResponse with standardized error format
    """
    logger.warning(f"HTTPException: {exception.status_code} - {exception.detail}")
    
    response_data = {
        "error": {
            "code": "HTTP_ERROR",
            "message": exception.detail,
            "status_code": exception.status_code
        }
    }
    
    return JSONResponse(
        status_code=exception.status_code,
        content=response_data
    )
