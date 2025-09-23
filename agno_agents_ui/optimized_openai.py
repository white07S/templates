"""
Optimized OpenAI client with caching, rate limiting, and batch processing.

This module provides enhanced OpenAI API clients that implement aggressive caching,
intelligent rate limiting, and batch processing for improved performance.
"""

import hashlib
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import AsyncOpenAI
import tiktoken

from config import settings
from cache_utils import TTLCache, LRUCache, AsyncRateLimiter, PersistentCache


class OptimizedEmbeddingClient:
    """Optimized OpenAI embedding client with caching and batching."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        # Caching layers
        self.embedding_cache = TTLCache(maxsize=2000, ttl=7200)  # 2-hour cache
        self.persistent_cache = PersistentCache(
            cache_dir=settings.data_dir / "embedding_cache",
            ttl=86400  # 24-hour persistent cache
        )

        # Rate limiting
        self.rate_limiter = AsyncRateLimiter(rate=3000, period=60)  # OpenAI limits

        # Performance tracking
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'batch_calls': 0,
            'total_tokens_processed': 0
        }

    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text with caching."""
        cache_key = self._generate_cache_key(text)

        # Try memory cache first
        cached_embedding = self.embedding_cache.get(cache_key)
        if cached_embedding is not None:
            self._stats['cache_hits'] += 1
            return cached_embedding

        # Try persistent cache
        cached_embedding = self.persistent_cache.get(cache_key)
        if cached_embedding is not None:
            self._stats['cache_hits'] += 1
            # Also store in memory cache for faster access
            self.embedding_cache.set(cache_key, cached_embedding)
            return cached_embedding

        # Cache miss - generate embedding
        self._stats['cache_misses'] += 1
        self._stats['api_calls'] += 1

        await self.rate_limiter.acquire()

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Update stats
            self._stats['total_tokens_processed'] += self.tokenizer.encode(text).__len__()

            # Cache the result
            self.embedding_cache.set(cache_key, embedding)
            self.persistent_cache.set(cache_key, embedding)

            return embedding

        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(self.dimension, dtype=np.float32)

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 50
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with intelligent batching."""
        if not texts:
            return []

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._generate_cache_key(text)

            # Try memory cache
            cached_embedding = self.embedding_cache.get(cache_key)
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
                self._stats['cache_hits'] += 1
                continue

            # Try persistent cache
            cached_embedding = self.persistent_cache.get(cache_key)
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
                self.embedding_cache.set(cache_key, cached_embedding)  # Store in memory too
                self._stats['cache_hits'] += 1
                continue

            # Need to generate
            uncached_texts.append(text)
            uncached_indices.append(i)
            self._stats['cache_misses'] += 1

        # Process uncached texts in batches
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                batch_indices = uncached_indices[batch_start:batch_end]

                batch_embeddings = await self._generate_batch_api_call(batch_texts)

                # Cache and store results
                for j, embedding in enumerate(batch_embeddings):
                    text = batch_texts[j]
                    original_index = batch_indices[j]
                    cache_key = self._generate_cache_key(text)

                    self.embedding_cache.set(cache_key, embedding)
                    self.persistent_cache.set(cache_key, embedding)
                    embeddings.append((original_index, embedding))

        # Sort by original order and return
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]

    async def _generate_batch_api_call(self, texts: List[str]) -> List[np.ndarray]:
        """Make batch API call to OpenAI."""
        await self.rate_limiter.acquire()

        try:
            self._stats['api_calls'] += 1
            self._stats['batch_calls'] += 1

            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            embeddings = [
                np.array(data.embedding, dtype=np.float32)
                for data in response.data
            ]

            # Update token stats
            total_tokens = sum(len(self.tokenizer.encode(text)) for text in texts)
            self._stats['total_tokens_processed'] += total_tokens

            return embeddings

        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [np.zeros(self.dimension, dtype=np.float32) for _ in texts]

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        return {
            **self._stats,
            'hit_ratio': self._stats['cache_hits'] / max(total_requests, 1),
            'avg_tokens_per_call': self._stats['total_tokens_processed'] / max(self._stats['api_calls'], 1),
            'cache_efficiency': self._stats['cache_hits'] / max(total_requests, 1) * 100
        }


class OptimizedChatClient:
    """Optimized OpenAI chat client with response caching."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.chat_model

        # Response caching (shorter TTL since responses can be more dynamic)
        self.response_cache = TTLCache(maxsize=500, ttl=1800)  # 30-minute cache
        self.compression_cache = TTLCache(maxsize=200, ttl=3600)  # 1-hour cache for compressions

        # Rate limiting
        self.rate_limiter = AsyncRateLimiter(rate=500, period=60)  # Conservative limits

        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'tokens_saved': 0
        }

    def _generate_cache_key(self, messages: List[Dict], **kwargs) -> str:
        """Generate cache key for chat completion."""
        # Create deterministic key from messages and parameters
        cache_data = {
            'model': self.model,
            'messages': messages,
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens'),
            'tools': kwargs.get('tools', [])
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()

    async def create_completion(
        self,
        messages: List[Dict],
        use_cache: bool = True,
        **kwargs
    ) -> Any:
        """Create chat completion with optional caching."""
        if use_cache:
            cache_key = self._generate_cache_key(messages, **kwargs)

            # Try cache first
            cached_response = self.response_cache.get(cache_key)
            if cached_response is not None:
                self._stats['cache_hits'] += 1
                return cached_response

        self._stats['cache_misses'] += 1
        self._stats['api_calls'] += 1

        await self.rate_limiter.acquire()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )

            # Cache the response if caching is enabled
            if use_cache:
                self.response_cache.set(cache_key, response)

            return response

        except Exception as e:
            print(f"Error creating chat completion: {e}")
            raise

    async def compress_text(
        self,
        text: str,
        target_ratio: float = 5.0,
        use_cache: bool = True
    ) -> str:
        """Compress text with caching for repeated compressions."""
        if len(text) < 100:
            return text

        if use_cache:
            cache_key = hashlib.sha256(f"{text}:{target_ratio}".encode()).hexdigest()

            cached_result = self.compression_cache.get(cache_key)
            if cached_result is not None:
                self._stats['cache_hits'] += 1
                return cached_result

        self._stats['cache_misses'] += 1

        prompt = f"""Compress the following text to approximately {int(100/target_ratio)}% of its original length while preserving all key information and semantic meaning.
        Focus on removing redundancy and verbose language while keeping facts, entities, and relationships intact.

        Original text:
        {text}

        Compressed version:"""

        messages = [
            {"role": "system", "content": "You are a text compression expert. Preserve semantic meaning while reducing verbosity."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self.create_completion(
                messages=messages,
                max_tokens=int(len(text.split()) / target_ratio),
                temperature=0.3,
                use_cache=False  # Don't cache the API call itself
            )

            compressed_text = response.choices[0].message.content.strip()

            # Cache the compression result
            if use_cache:
                self.compression_cache.set(cache_key, compressed_text)

            return compressed_text

        except Exception as e:
            print(f"Error compressing text: {e}")
            return text

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        return {
            **self._stats,
            'hit_ratio': self._stats['cache_hits'] / max(total_requests, 1),
            'response_cache_stats': self.response_cache.stats(),
            'compression_cache_stats': self.compression_cache.stats()
        }