"""
Caching utilities for performance optimization.

This module provides various caching mechanisms including TTL cache, LRU cache,
and async-safe caching for database operations, API calls, and memory retrieval.
"""

import asyncio
import hashlib
import time
from typing import Any, Dict, Optional, Callable, Union
from collections import OrderedDict
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path


class TTLCache:
    """Time-To-Live cache implementation."""

    def __init__(self, maxsize: int = 1000, ttl: int = 300):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: OrderedDict = OrderedDict()

    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []

        for key, data in self._cache.items():
            if current_time - data['timestamp'] > self.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)

    def _ensure_capacity(self):
        """Ensure cache doesn't exceed maxsize."""
        while len(self._cache) >= self.maxsize:
            # Remove least recently used item
            oldest_key = next(iter(self._access_times))
            self._cache.pop(oldest_key, None)
            self._access_times.pop(oldest_key, None)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self._cleanup_expired()

        if key in self._cache:
            # Update access time
            self._access_times.move_to_end(key)
            return self._cache[key]['value']

        return None

    def set(self, key: str, value: Any):
        """Set value in cache."""
        self._cleanup_expired()
        self._ensure_capacity()

        self._cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        self._access_times[key] = time.time()

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._cleanup_expired()
        return {
            'size': len(self._cache),
            'maxsize': self.maxsize,
            'ttl': self.ttl,
            'hit_ratio': getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
        }


class LRUCache:
    """Least Recently Used cache implementation."""

    def __init__(self, maxsize: int = 500):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        return None

    def set(self, key: str, value: Any):
        """Set value in cache."""
        if key in self._cache:
            # Update existing key
            self._cache[key] = value
            self._cache.move_to_end(key)
        else:
            # Add new key
            if len(self._cache) >= self.maxsize:
                # Remove least recently used
                self._cache.popitem(last=False)

            self._cache[key] = value

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        return {
            'size': len(self._cache),
            'maxsize': self.maxsize,
            'hits': self._hits,
            'misses': self._misses,
            'hit_ratio': self._hits / max(total_requests, 1)
        }


class AsyncRateLimiter:
    """Async rate limiter for API calls."""

    def __init__(self, rate: int = 100, period: int = 60):
        self.rate = rate
        self.period = period
        self._calls = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire rate limit permission."""
        async with self._lock:
            now = time.time()

            # Remove old calls outside the period
            self._calls = [call_time for call_time in self._calls
                          if now - call_time < self.period]

            # Check if we're under the rate limit
            if len(self._calls) >= self.rate:
                # Calculate wait time
                oldest_call = min(self._calls)
                wait_time = self.period - (now - oldest_call)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()

            # Record this call
            self._calls.append(now)


class PersistentCache:
    """Persistent cache that survives application restarts."""

    def __init__(self, cache_dir: Path, ttl: int = 3600):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            # Check TTL
            if time.time() - data['timestamp'] > self.ttl:
                cache_path.unlink(missing_ok=True)
                return None

            return data['value']

        except Exception:
            # Clean up corrupted cache file
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any):
        """Set value in persistent cache."""
        cache_path = self._get_cache_path(key)

        data = {
            'value': value,
            'timestamp': time.time()
        }

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Failed to write cache: {e}")

    def clear(self):
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink(missing_ok=True)


def cache_key_from_args(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Create a deterministic string from args and kwargs
    key_data = {
        'args': str(args),
        'kwargs': sorted(kwargs.items())
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


def async_cache(cache_instance: Union[TTLCache, LRUCache], key_func: Optional[Callable] = None):
    """Decorator for async function caching."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = cache_key_from_args(*args, **kwargs)

            # Try to get from cache
            cached_result = cache_instance.get(key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_instance.set(key, result)
            return result

        return wrapper
    return decorator


class BatchProcessor:
    """Batch processor for efficient bulk operations."""

    def __init__(self, batch_size: int = 100, flush_interval: float = 5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._batch = []
        self._last_flush = time.time()
        self._lock = asyncio.Lock()

    async def add(self, item: Any, process_func: Callable):
        """Add item to batch and process if needed."""
        async with self._lock:
            self._batch.append(item)

            # Check if we should flush
            should_flush = (
                len(self._batch) >= self.batch_size or
                time.time() - self._last_flush > self.flush_interval
            )

            if should_flush:
                await self._flush(process_func)

    async def _flush(self, process_func: Callable):
        """Flush current batch."""
        if not self._batch:
            return

        batch_to_process = self._batch.copy()
        self._batch.clear()
        self._last_flush = time.time()

        try:
            await process_func(batch_to_process)
        except Exception as e:
            print(f"Batch processing failed: {e}")
            # Re-add items to batch for retry
            self._batch.extend(batch_to_process)

    async def force_flush(self, process_func: Callable):
        """Force flush current batch."""
        async with self._lock:
            await self._flush(process_func)