"""
Pytest configuration and fixtures for the FPL Prediction Platform.

This module provides:
- Session-scoped fixtures for database setup/teardown
- Function-scoped fixtures for database sessions (with rollback)
- Async client fixture for FastAPI endpoint testing
"""

import os
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import Session, sessionmaker
from httpx import AsyncClient

# Set test mode before importing app modules
os.environ["MODE"] = "TEST"

from app.core.config import settings
from app.database import Base


# ============================================================================
# Database Setup/Teardown (Session-scoped)
# ============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for the test session.
    Required for async fixtures in pytest-asyncio.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """
    Get the test database URL from settings.
    If DATABASE_URL is set in environment, use it (for Docker compatibility).
    
    Returns:
        Test database connection string.
    """
    # Check if DATABASE_URL is set in environment (Docker sets this)
    env_db_url = os.getenv("DATABASE_URL")
    if env_db_url:
        # Use the same database URL but ensure it's for test database
        # Replace database name with test database name (only at the end, after the last /)
        # Format: postgresql://user:pass@host:port/dbname
        import re
        # Replace /fpl_db at the end of the URL (before any query params)
        # This ensures we don't replace fpl_db in the username
        test_url = re.sub(r'/fpl_db(\?|$)', r'/fpl_test_db\1', env_db_url)
        if test_url != env_db_url:
            return test_url
        # If no replacement happened, check if it ends with /fpl_db
        if env_db_url.endswith("/fpl_db"):
            return env_db_url[:-7] + "/fpl_test_db"
        # If no fpl_db in URL, just use it as-is (might already be test DB)
        return env_db_url
    
    # Fall back to settings (for local development)
    return settings.database_url


@pytest.fixture(scope="session")
def sync_test_engine(test_database_url: str):
    """
    Create a synchronous test database engine (session-scoped).
    
    Args:
        test_database_url: Test database connection string.
        
    Yields:
        SQLAlchemy sync engine for test database.
    """
    # Convert async URL to sync if needed
    sync_url = test_database_url.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(sync_url, pool_pre_ping=True, echo=False)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup: Drop all tables
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="session")
def async_test_engine(test_database_url: str, event_loop):
    """
    Create an asynchronous test database engine (session-scoped).
    
    Args:
        test_database_url: Test database connection string.
        event_loop: Session-scoped event loop.
        
    Yields:
        SQLAlchemy async engine for test database.
    """
    # Ensure async URL format
    async_url = test_database_url
    if async_url.startswith("postgresql://"):
        async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")
    
    engine = create_async_engine(async_url, echo=False, pool_pre_ping=True)
    
    # Create all tables using sync metadata (works with async engine)
    # We'll create tables in the first async_db_session usage
    yield engine
    
    # Cleanup: Dispose engine
    # Use the event loop to properly dispose
    async def cleanup():
        await engine.dispose()
    
    event_loop.run_until_complete(cleanup())


# ============================================================================
# Database Sessions (Function-scoped with rollback)
# ============================================================================


@pytest.fixture
def sync_db_session(sync_test_engine) -> Generator[Session, None, None]:
    """
    Provide a synchronous database session with automatic rollback.
    
    Each test gets a fresh session that rolls back all changes after the test.
    
    Yields:
        SQLAlchemy sync session.
    """
    connection = sync_test_engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
async def async_db_session(
    async_test_engine,
    sync_test_engine,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide an asynchronous database session with automatic rollback.
    
    Each test gets a fresh session that rolls back all changes after the test.
    Tables are created on first use if they don't exist.
    
    Yields:
        SQLAlchemy async session.
    """
    # Ensure tables exist (use sync engine for this)
    Base.metadata.create_all(bind=sync_test_engine)
    
    async_session_maker = async_sessionmaker(
        bind=async_test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_maker() as session:
        # Begin a nested transaction
        transaction = await session.begin()
        
        try:
            yield session
        finally:
            # Rollback the transaction
            await transaction.rollback()
            await session.close()


# ============================================================================
# FastAPI Test Client
# ============================================================================


@pytest.fixture
async def async_client(sync_db_session: Session) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an async HTTP client for testing FastAPI endpoints.
    
    The client uses the test database session and automatically
    handles dependency injection overrides.
    
    Yields:
        httpx.AsyncClient configured for the FastAPI app.
    """
    # Import app only when needed to avoid async issues during import
    from app.main import app
    from app.database import get_db
    
    # Override the get_db dependency to use test session
    def override_get_db():
        try:
            yield sync_db_session
        finally:
            pass  # Session cleanup handled by sync_db_session fixture
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    # Clean up overrides
    app.dependency_overrides.clear()


# ============================================================================
# Helper Fixtures
# ============================================================================


@pytest.fixture
def test_settings():
    """
    Provide test settings configuration.
    
    Yields:
        Settings instance configured for testing.
    """
    return settings
