"""
ML Engine - Compatibility Wrapper
This file maintains backward compatibility while the codebase migrates to the new modular structure.
All imports should eventually be updated to use: from app.services.ml import PLEngine
"""
# Import from new modular structure
from app.services.ml import PLEngine
from app.services.ml.strategies.xmins_strategy import XMinsStrategy as XMinsModel
from app.services.ml.strategies.attack_strategy import AttackStrategy as AttackModel
from app.services.ml.strategies.defense_strategy import DefenseStrategy as DefenseModel

# Re-export for backward compatibility
__all__ = ['PLEngine', 'XMinsModel', 'AttackModel', 'DefenseModel']
