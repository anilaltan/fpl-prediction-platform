"""
Model Loader
Manages model lifecycle: load on demand, unload immediately after use.
Critical for 4GB RAM constraint.
"""

import asyncio
import gc
import logging
import os
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
        self, strategy: ModelInterface, model_path: Optional[str] = None
    ) -> None:
        """
        Load model, ensuring previous model is unloaded if needed.

        Args:
            strategy: Model strategy implementing ModelInterface
            model_path: Optional path to model file
        """
        async with self._load_lock:
            # IMPORTANT: Don't unload the strategy we're about to load
            # Only unload OTHER strategies that are different instances
            # Check by object identity to ensure we never unload the current strategy
            await self._unload_other_models(strategy)

            # Load the requested model
            # Always try to load if model_path is provided
            # (trained models from pickle need to be loaded into memory)
            if model_path:
                logger.info(f"[ModelLoader] Loading {type(strategy).__name__} from {model_path}")
                logger.info(f"[ModelLoader] Model file exists: {os.path.exists(model_path)}")
                logger.info(f"[ModelLoader] Strategy state before load: is_loaded={getattr(strategy, 'is_loaded', 'N/A')}")
                try:
                    await strategy.load(model_path)
                    logger.info(f"[ModelLoader] Strategy.load() completed for {type(strategy).__name__}")
                    
                    # Verify load was actually successful by checking strategy-specific attributes
                    load_successful = False
                    if hasattr(strategy, 'is_loaded'):
                        is_loaded = strategy.is_loaded
                        if hasattr(strategy, 'is_trained') and hasattr(strategy, 'model'):
                            # For xMins strategy
                            is_trained = strategy.is_trained
                            model_exists = strategy.model is not None
                            load_successful = is_loaded and is_trained and model_exists
                            logger.info(
                                f"After load - {type(strategy).__name__}: "
                                f"is_trained={is_trained}, model={'exists' if model_exists else 'None'}, "
                                f"is_loaded={is_loaded}, model_type={type(strategy.model).__name__ if model_exists else 'None'}"
                            )
                        elif hasattr(strategy, 'xg_trained') and hasattr(strategy, 'xg_model') and hasattr(strategy, 'xa_model'):
                            # For Attack strategy
                            xg_trained = strategy.xg_trained
                            xa_trained = strategy.xa_trained
                            xg_exists = strategy.xg_model is not None
                            xa_exists = strategy.xa_model is not None
                            load_successful = is_loaded and xg_trained and xa_trained and xg_exists and xa_exists
                            logger.info(
                                f"After load - {type(strategy).__name__}: "
                                f"xg_trained={xg_trained}, xa_trained={xa_trained}, "
                                f"xg_model={'exists' if xg_exists else 'None'}, "
                                f"xa_model={'exists' if xa_exists else 'None'}, "
                                f"is_loaded={is_loaded}"
                            )
                        else:
                            # For other strategies (e.g., Defense)
                            load_successful = is_loaded
                            logger.info(
                                f"After load - {type(strategy).__name__}: is_loaded={is_loaded}"
                            )
                    
                    if not load_successful:
                        error_msg = (
                            f"Load reported success but model verification failed for {type(strategy).__name__}. "
                            f"is_loaded={is_loaded}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    if strategy not in self._loaded_models:
                        self._loaded_models.append(strategy)
                    logger.info(f"Successfully loaded and verified model: {type(strategy).__name__}")
                except Exception as e:
                    logger.error(
                        f"Failed to load {type(strategy).__name__} from {model_path}: {str(e)}",
                        exc_info=True
                    )
                    raise
            elif strategy not in self._loaded_models:
                # Model is already loaded/trained, just add to list
                self._loaded_models.append(strategy)
                logger.debug(f"Model already available: {type(strategy).__name__}")

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
            current_strategy: Strategy to keep loaded (never unload this one)
        """
        # Only unload strategies that are in the loaded list AND are different instances
        # Use object identity (id) to ensure we never accidentally unload the current strategy
        current_strategy_id = id(current_strategy)
        for strategy in list(self._loaded_models):
            strategy_id = id(strategy)
            # Only unload if it's a DIFFERENT strategy instance (not the one we're loading)
            if strategy_id != current_strategy_id and strategy.is_loaded:
                logger.debug(f"[ModelLoader] Unloading other model: {type(strategy).__name__} (id={strategy_id}, current_id={current_strategy_id})")
                await strategy.unload()
                self._loaded_models.remove(strategy)
                gc.collect()
            elif strategy_id == current_strategy_id:
                logger.debug(f"[ModelLoader] Skipping unload of current strategy: {type(strategy).__name__} (same id={strategy_id})")
