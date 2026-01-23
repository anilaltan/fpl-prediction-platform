"""
Model Loader
Manages model lifecycle: load on demand, unload immediately after use.
Critical for 4GB RAM constraint.
"""
import asyncio
import gc
import logging
from typing import Optional, List
from .interfaces import ModelInterface

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Manages model lifecycle: load on demand, unload immediately after use.
    
    Ensures models are only in memory during inference and are immediately
    unloaded via gc.collect() to save RAM in the 4GB constraint environment.
    """
    
    def __init__(self):
        """Initialize model loader."""
        self._load_lock = asyncio.Lock()
        self._loaded_models: List[ModelInterface] = []
    
    async def load_model(
        self,
        strategy: ModelInterface,
        model_path: Optional[str] = None
    ) -> None:
        """
        Load model, ensuring previous model is unloaded if needed.
        
        Args:
            strategy: Model strategy implementing ModelInterface
            model_path: Optional path to model file
        """
        async with self._load_lock:
            # Unload any other models to save memory
            await self._unload_other_models(strategy)
            
            # Load the requested model
            if not strategy.is_loaded:
                await strategy.load(model_path)
                if strategy.is_loaded:
                    self._loaded_models.append(strategy)
                    logger.debug(f"Loaded model: {type(strategy).__name__}")
    
    async def unload_model(self, strategy: ModelInterface) -> None:
        """
        Unload model and force garbage collection.
        
        Args:
            strategy: Model strategy to unload
        """
        async with self._load_lock:
            if strategy.is_loaded:
                await strategy.unload()
                if strategy in self._loaded_models:
                    self._loaded_models.remove(strategy)
                
                # Force garbage collection to free RAM immediately
                gc.collect()
                logger.debug(f"Unloaded model: {type(strategy).__name__}")
    
    async def unload_all(self) -> None:
        """Unload all loaded models and force garbage collection."""
        async with self._load_lock:
            for strategy in list(self._loaded_models):
                await strategy.unload()
            self._loaded_models.clear()
            gc.collect()
            logger.debug("Unloaded all models")
    
    async def _unload_other_models(self, current_strategy: ModelInterface) -> None:
        """
        Unload all models except the current one to save memory.
        
        Args:
            current_strategy: Strategy to keep loaded
        """
        for strategy in list(self._loaded_models):
            if strategy is not current_strategy and strategy.is_loaded:
                await strategy.unload()
                self._loaded_models.remove(strategy)
                gc.collect()
