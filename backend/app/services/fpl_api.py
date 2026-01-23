"""
FPL API Service - Compatibility Wrapper
This file maintains backward compatibility while the codebase migrates to the new modular structure.
All imports should eventually be updated to use: from app.services.fpl import FPLAPIService
"""
# Import from new modular structure
from app.services.fpl import FPLAPIService

# Re-export for backward compatibility
__all__ = ["FPLAPIService"]
