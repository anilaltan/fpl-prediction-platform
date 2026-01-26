"""
Configuration settings for the FPL Prediction Platform.
Supports testing, development, and production environments.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """
    Application settings with environment-aware configuration.
    
    Supports three modes:
    - TEST: Uses TEST_DATABASE_URL for isolated test database
    - DEVELOPMENT: Uses DATABASE_URL for development database
    - PRODUCTION: Uses DATABASE_URL for production database
    """
    
    # Environment mode: TEST, DEVELOPMENT, or PRODUCTION
    MODE: str = os.getenv("MODE", "DEVELOPMENT").upper()
    
    # Database URLs
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://fpl_user:fpl_password@localhost:5432/fpl_db"
    )
    TEST_DATABASE_URL: Optional[str] = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://fpl_user:fpl_password@localhost:5432/fpl_test_db"
    )
    
    # Application settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # FPL API settings
    FPL_EMAIL: Optional[str] = os.getenv("FPL_EMAIL")
    FPL_PASSWORD: Optional[str] = os.getenv("FPL_PASSWORD")
    
    class Config:
        """Pydantic configuration."""
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @property
    def database_url(self) -> str:
        """
        Get the appropriate database URL based on the current mode.
        
        Returns:
            Database URL string for the current environment.
        """
        if self.MODE == "TEST":
            return self.TEST_DATABASE_URL or self.DATABASE_URL.replace(
                "/fpl_db", "/fpl_test_db"
            )
        return self.DATABASE_URL
    
    @property
    def is_testing(self) -> bool:
        """Check if running in test mode."""
        return self.MODE == "TEST"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.MODE == "DEVELOPMENT"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.MODE == "PRODUCTION"


# Global settings instance
settings = Settings()
