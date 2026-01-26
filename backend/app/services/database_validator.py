"""
Database Connection Validator
Focused validation module for database connectivity with configurable timeout and clear error reporting.
Independent of FastAPI/startup context for easy testing and reuse.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from datetime import datetime
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import (
    SQLAlchemyError,
    OperationalError,
    DisconnectionError,
)
from sqlalchemy.pool import QueuePool
import psycopg2
from psycopg2 import OperationalError as Psycopg2OperationalError

logger = logging.getLogger(__name__)


@dataclass
class DatabaseValidationResult:
    """
    Result of validating database connection.
    
    Attributes:
        database_url: Database connection URL (with credentials masked)
        is_valid: True if validation passed, False otherwise
        error_type: Type of error if validation failed (e.g., "connection_refused", "timeout", 
                   "authentication_failure", "database_not_found", "unknown_error")
        error_message: Detailed error message if validation failed
        connection_time_ms: Time taken to establish connection in milliseconds (None if failed)
        query_time_ms: Time taken to execute health check query in milliseconds (None if failed)
        timestamp: When the validation was performed
    """
    database_url: str
    is_valid: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    connection_time_ms: Optional[float] = None
    query_time_ms: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/reporting."""
        return {
            "database_url": self.database_url,
            "is_valid": self.is_valid,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "connection_time_ms": self.connection_time_ms,
            "query_time_ms": self.query_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class DatabaseValidator:
    """
    Validates database connection with configurable timeout and clear error reporting.
    
    This module is independent of FastAPI/startup context and can be used
    in standalone scripts, tests, or during API startup.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        timeout: float = 2.0,
    ):
        """
        Initialize database validator.
        
        Args:
            database_url: Database connection URL. If None, uses DATABASE_URL from environment.
            timeout: Connection timeout in seconds (default: 2.0). Used for both connection
                    and query execution timeout.
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://fpl_user:fpl_password@localhost:5432/fpl_db"
        )
        self.timeout = timeout
        self._masked_url = self._mask_credentials(self.database_url)

    def validate(self) -> DatabaseValidationResult:
        """
        Validate database connection synchronously.
        
        Attempts to establish a connection and execute a simple health check query.
        Returns structured validation results with specific error types.
        
        Returns:
            DatabaseValidationResult with detailed validation information
            
        Example:
            >>> validator = DatabaseValidator(timeout=2.0)
            >>> result = validator.validate()
            >>> if result.is_valid:
            ...     print(f"Database connection OK ({result.connection_time_ms}ms)")
            ... else:
            ...     print(f"Database connection failed: {result.error_message}")
        """
        import time
        
        start_time = time.time()
        engine = None
        connection = None
        
        try:
            # Create engine with timeout configuration
            engine = self._create_engine_with_timeout()
            
            # Attempt connection
            connection_start = time.time()
            try:
                connection = engine.connect()
                connection_time = (time.time() - connection_start) * 1000  # Convert to ms
            except Exception as conn_error:
                connection_time = None
                error_type, error_message = self._classify_connection_error(conn_error)
                return DatabaseValidationResult(
                    database_url=self._masked_url,
                    is_valid=False,
                    error_type=error_type,
                    error_message=error_message,
                    connection_time_ms=connection_time,
                    timestamp=datetime.utcnow(),
                )
            
            # Execute health check query
            query_start = time.time()
            try:
                result = connection.execute(text("SELECT 1"))
                result.fetchone()
                query_time = (time.time() - query_start) * 1000  # Convert to ms
            except Exception as query_error:
                query_time = None
                error_type, error_message = self._classify_query_error(query_error)
                return DatabaseValidationResult(
                    database_url=self._masked_url,
                    is_valid=False,
                    error_type=error_type,
                    error_message=error_message,
                    connection_time_ms=connection_time,
                    query_time_ms=query_time,
                    timestamp=datetime.utcnow(),
                )
            
            # All checks passed
            total_time = (time.time() - start_time) * 1000
            return DatabaseValidationResult(
                database_url=self._masked_url,
                is_valid=True,
                connection_time_ms=connection_time,
                query_time_ms=query_time,
                timestamp=datetime.utcnow(),
            )
            
        except Exception as e:
            # Unexpected error
            error_type, error_message = self._classify_connection_error(e)
            return DatabaseValidationResult(
                database_url=self._masked_url,
                is_valid=False,
                error_type=error_type,
                error_message=error_message,
                timestamp=datetime.utcnow(),
            )
        finally:
            # Ensure connection cleanup
            if connection:
                try:
                    connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {str(e)}")
            
            if engine:
                try:
                    engine.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing engine: {str(e)}")

    async def validate_async(self) -> DatabaseValidationResult:
        """
        Validate database connection asynchronously.
        
        Uses async SQLAlchemy engine for async/await pattern support.
        Returns structured validation results with specific error types.
        
        Returns:
            DatabaseValidationResult with detailed validation information
            
        Example:
            >>> validator = DatabaseValidator(timeout=2.0)
            >>> result = await validator.validate_async()
            >>> if result.is_valid:
            ...     print(f"Database connection OK ({result.connection_time_ms}ms)")
        """
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        import time
        import asyncio
        
        start_time = time.time()
        engine = None
        connection = None
        
        try:
            # Convert to async URL if needed
            async_url = self._convert_to_async_url(self.database_url)
            
            # Create async engine with timeout
            engine = create_async_engine(
                async_url,
                pool_pre_ping=True,
                connect_args={
                    "command_timeout": self.timeout,
                    "server_settings": {
                        "statement_timeout": str(int(self.timeout * 1000)),  # Convert to milliseconds
                    },
                },
                poolclass=QueuePool,
                pool_size=1,
                max_overflow=0,
            )
            
            # Attempt connection
            connection_start = time.time()
            try:
                connection = await asyncio.wait_for(
                    engine.connect(),
                    timeout=self.timeout,
                )
                connection_time = (time.time() - connection_start) * 1000
            except asyncio.TimeoutError:
                connection_time = None
                return DatabaseValidationResult(
                    database_url=self._masked_url,
                    is_valid=False,
                    error_type="timeout",
                    error_message=f"Connection timeout after {self.timeout}s",
                    connection_time_ms=connection_time,
                    timestamp=datetime.utcnow(),
                )
            except Exception as conn_error:
                connection_time = None
                error_type, error_message = self._classify_connection_error(conn_error)
                return DatabaseValidationResult(
                    database_url=self._masked_url,
                    is_valid=False,
                    error_type=error_type,
                    error_message=error_message,
                    connection_time_ms=connection_time,
                    timestamp=datetime.utcnow(),
                )
            
            # Execute health check query
            query_start = time.time()
            try:
                result = await asyncio.wait_for(
                    connection.execute(text("SELECT 1")),
                    timeout=self.timeout,
                )
                await result.fetchone()
                query_time = (time.time() - query_start) * 1000
            except asyncio.TimeoutError:
                query_time = None
                return DatabaseValidationResult(
                    database_url=self._masked_url,
                    is_valid=False,
                    error_type="timeout",
                    error_message=f"Query execution timeout after {self.timeout}s",
                    connection_time_ms=connection_time,
                    query_time_ms=query_time,
                    timestamp=datetime.utcnow(),
                )
            except Exception as query_error:
                query_time = None
                error_type, error_message = self._classify_query_error(query_error)
                return DatabaseValidationResult(
                    database_url=self._masked_url,
                    is_valid=False,
                    error_type=error_type,
                    error_message=error_message,
                    connection_time_ms=connection_time,
                    query_time_ms=query_time,
                    timestamp=datetime.utcnow(),
                )
            
            # All checks passed
            return DatabaseValidationResult(
                database_url=self._masked_url,
                is_valid=True,
                connection_time_ms=connection_time,
                query_time_ms=query_time,
                timestamp=datetime.utcnow(),
            )
            
        except Exception as e:
            # Unexpected error
            error_type, error_message = self._classify_connection_error(e)
            return DatabaseValidationResult(
                database_url=self._masked_url,
                is_valid=False,
                error_type=error_type,
                error_message=error_message,
                timestamp=datetime.utcnow(),
            )
        finally:
            # Ensure connection cleanup
            if connection:
                try:
                    await connection.close()
                except Exception as e:
                    logger.warning(f"Error closing async connection: {str(e)}")
            
            if engine:
                try:
                    await engine.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing async engine: {str(e)}")

    def _create_engine_with_timeout(self) -> Engine:
        """
        Create SQLAlchemy engine with timeout configuration.
        
        Returns:
            Configured SQLAlchemy Engine instance
        """
        return create_engine(
            self.database_url,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": int(self.timeout),
                "options": f"-c statement_timeout={int(self.timeout * 1000)}",  # milliseconds
            },
            poolclass=QueuePool,
            pool_size=1,
            max_overflow=0,
        )

    def _classify_connection_error(self, error: Exception) -> Tuple[str, str]:
        """
        Classify connection error into specific error types.
        
        Args:
            error: Exception that occurred during connection
            
        Returns:
            Tuple of (error_type, error_message)
        """
        error_str = str(error).lower()
        error_repr = repr(error)
        
        # Connection refused / Network unreachable
        if any(term in error_str for term in [
            "connection refused",
            "could not connect",
            "network is unreachable",
            "no route to host",
            "connection timed out",
            "timeout expired",
        ]) or isinstance(error, (Psycopg2OperationalError, OperationalError)):
            if "connection refused" in error_str or "could not connect" in error_str:
                return (
                    "connection_refused",
                    f"Database connection refused. Check if database is running and accessible at the configured host/port.",
                )
            elif "timeout" in error_str or "timed out" in error_str:
                return (
                    "timeout",
                    f"Connection timeout after {self.timeout}s. Database may be slow or unreachable.",
                )
            else:
                return (
                    "connection_refused",
                    f"Database connection failed: {str(error)}",
                )
        
        # Authentication failure
        if any(term in error_str for term in [
            "authentication failed",
            "password authentication failed",
            "invalid password",
            "access denied",
            "permission denied",
        ]):
            return (
                "authentication_failure",
                "Database authentication failed. Check username and password credentials.",
            )
        
        # Database not found
        if any(term in error_str for term in [
            "database.*does not exist",
            "database.*not found",
            "unknown database",
        ]):
            return (
                "database_not_found",
                "Database not found. Check if the database name is correct and the database exists.",
            )
        
        # Timeout errors
        if isinstance(error, TimeoutError) or "timeout" in error_str:
            return (
                "timeout",
                f"Connection or operation timeout after {self.timeout}s.",
            )
        
        # Disconnection errors
        if isinstance(error, DisconnectionError):
            return (
                "connection_lost",
                "Database connection was lost during operation.",
            )
        
        # Generic SQLAlchemy errors
        if isinstance(error, SQLAlchemyError):
            return (
                "database_error",
                f"Database error: {str(error)}",
            )
        
        # Unknown error
        return (
            "unknown_error",
            f"Unexpected error during database connection: {str(error)}",
        )

    def _classify_query_error(self, error: Exception) -> Tuple[str, str]:
        """
        Classify query execution error into specific error types.
        
        Args:
            error: Exception that occurred during query execution
            
        Returns:
            Tuple of (error_type, error_message)
        """
        error_str = str(error).lower()
        
        # Timeout errors
        if isinstance(error, TimeoutError) or "timeout" in error_str:
            return (
                "timeout",
                f"Query execution timeout after {self.timeout}s.",
            )
        
        # Connection lost during query
        if isinstance(error, DisconnectionError) or "connection" in error_str and "lost" in error_str:
            return (
                "connection_lost",
                "Database connection was lost during query execution.",
            )
        
        # Generic SQLAlchemy errors
        if isinstance(error, SQLAlchemyError):
            return (
                "query_error",
                f"Query execution error: {str(error)}",
            )
        
        # Unknown error
        return (
            "unknown_error",
            f"Unexpected error during query execution: {str(error)}",
        )

    def _mask_credentials(self, url: str) -> str:
        """
        Mask credentials in database URL for logging/error messages.
        
        Args:
            url: Database connection URL
            
        Returns:
            URL with credentials masked
        """
        try:
            parsed = urlparse(url)
            if parsed.password:
                # Replace password with ***
                masked = parsed._replace(password="***")
                return masked.geturl()
            return url
        except Exception:
            # If parsing fails, return a safe version
            if "@" in url:
                # Mask password if present
                parts = url.split("@")
                if len(parts) == 2:
                    auth_part = parts[0]
                    if ":" in auth_part:
                        user_part = auth_part.split(":")[0]
                        return f"{user_part}:***@{parts[1]}"
            return url

    def _convert_to_async_url(self, url: str) -> str:
        """
        Convert synchronous database URL to async format.
        
        Args:
            url: Synchronous database URL (e.g., postgresql://...)
            
        Returns:
            Async database URL (e.g., postgresql+asyncpg://...)
        """
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql+psycopg2://"):
            return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        # Already async or unknown format
        return url
