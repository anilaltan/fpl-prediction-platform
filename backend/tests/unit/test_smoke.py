"""
Smoke tests to verify the testing infrastructure is correctly configured.

These tests validate:
- Basic pytest functionality
- Database connection and fixtures
- Async test support
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session


def test_basic_math():
    """Simple test to verify pytest is working."""
    assert 1 + 1 == 2


def test_sync_db_session(sync_db_session: Session):
    """
    Test that the sync database session fixture works correctly.
    
    Verifies:
    - Session is created
    - Can execute a simple query
    - Session is properly configured
    """
    # Execute a simple query
    result = sync_db_session.execute(text("SELECT 1 as value"))
    row = result.fetchone()
    
    assert row is not None
    assert row.value == 1


@pytest.mark.asyncio
async def test_async_db_session(async_db_session: AsyncSession):
    """
    Test that the async database session fixture works correctly.
    
    Verifies:
    - Async session is created
    - Can execute a simple async query
    - Session is properly configured
    """
    # Execute a simple async query
    result = await async_db_session.execute(text("SELECT 1 as value"))
    row = result.fetchone()
    
    assert row is not None
    assert row.value == 1


@pytest.mark.asyncio
async def test_async_client(async_client):
    """
    Test that the async HTTP client fixture works correctly.
    
    Verifies:
    - Client is created
    - Can make requests to the FastAPI app
    - App is properly configured
    """
    # Make a simple request to a health endpoint (if it exists)
    # For now, just verify the client is created
    assert async_client is not None
    assert async_client.base_url == "http://test"


def test_settings_fixture(test_settings):
    """
    Test that the settings fixture works correctly.
    
    Verifies:
    - Settings are loaded
    - Test mode is correctly set
    - Database URL is configured
    """
    assert test_settings is not None
    assert test_settings.is_testing is True
    assert test_settings.database_url is not None
    assert "test" in test_settings.database_url.lower() or test_settings.MODE == "TEST"
