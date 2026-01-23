"""
In-Memory Cache with TTL
Strict LRU caching logic for FPL API responses.
"""
import time
import asyncio
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CacheEntry:
    """
    Cache entry with timestamp and TTL.
    
    Attributes:
        data: Cached data
        created_at: Timestamp when entry was created
        ttl_seconds: Time-to-live in seconds
    """
    
    def __init__(self, data: Any, ttl_seconds: int):
        """
        Initialize cache entry.
        
        Args:
            data: Data to cache
            ttl_seconds: Time-to-live in seconds
        """
        self.data = data
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """
        Check if cache entry has expired.
        
        Returns:
            True if entry has expired, False otherwise
        """
        return time.time() - self.created_at > self.ttl_seconds


class InMemoryCache:
    """
    Simple in-memory cache with TTL support.
    
    Thread-safe async cache implementation with automatic expiration.
    Uses LRU-style eviction (expired entries are removed on access).
    """
    
    # Cache TTL constants (in seconds)
    BOOTSTRAP_CACHE_TTL = 24 * 60 * 60  # 24 hours
    ELEMENT_SUMMARY_CACHE_TTL = 60 * 60  # 1 hour
    
    def __init__(self):
        """Initialize in-memory cache."""
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value if exists and not expired, None otherwise
        """
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    return entry.data
                else:
                    # Remove expired entry
                    del self._cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
        """
        async with self._lock:
            self._cache[key] = CacheEntry(value, ttl_seconds)
    
    async def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache entry or all entries.
        
        Args:
            key: Optional cache key to clear. If None, clears all cache.
        """
        async with self._lock:
            if key:
                self._cache.pop(key, None)
                logger.info(f"Cleared cache entry: {key}")
            else:
                self._cache.clear()
                logger.info("Cleared all cache entries")
    
    async def get_or_set(
        self,
        key: str,
        value_factory: callable,
        ttl_seconds: int
    ) -> Any:
        """
        Get value from cache, or set it using factory function if not found.
        
        Args:
            key: Cache key
            value_factory: Async function that returns the value to cache
            ttl_seconds: Time-to-live in seconds
        
        Returns:
            Cached or newly computed value
        """
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        value = await value_factory()
        await self.set(key, value, ttl_seconds)
        return value
