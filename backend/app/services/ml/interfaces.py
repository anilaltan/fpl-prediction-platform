"""
Model Interface
Abstract interface for all ML models with lazy loading support.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelInterface(ABC):
    """
    Abstract interface for all ML models.
    
    All model strategies must implement this interface to ensure
    consistent lazy loading, memory management, and prediction API.
    """
    
    @abstractmethod
    async def load(self, model_path: Optional[str] = None) -> None:
        """
        Load model into memory from file or initialize empty model.
        
        Args:
            model_path: Optional path to model file. If None, initializes empty model.
        
        Raises:
            FileNotFoundError: If model_path is provided but file doesn't exist
        """
        pass
    
    @abstractmethod
    async def unload(self) -> None:
        """
        Unload model from memory and call gc.collect() to free RAM.
        
        This is critical for 4GB RAM constraint - models should only
        be loaded during inference and unloaded immediately after.
        """
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Make prediction using the model.
        
        Returns:
            Dictionary with prediction results
        """
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded in memory.
        
        Returns:
            True if model is loaded, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """
        Check if model has been trained.
        
        Returns:
            True if model is trained, False otherwise
        """
        pass
