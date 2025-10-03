"""
Utility functions for the advanced search algorithm.
"""
import asyncio
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from openai import AsyncOpenAI
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm
import config

class EmbeddingClient:
    """Optimized async OpenAI embedding client with batching and caching."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
        self.cache_path = Path(config.EMBEDDINGS_CACHE_PATH)
        self.cache = self._load_cache()
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

    def _load_cache(self) -> Dict[str, np.ndarray]:
        """Load embeddings cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
        return {}

    def _save_cache(self):
        """Save embeddings cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string."""
        return hashlib.md5(f"{text}_{self.model}".encode()).hexdigest()

    async def _get_embedding_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts from OpenAI API."""
        async with self.semaphore:
            try:
                response = await self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                return [np.array(data.embedding, dtype=np.float32) for data in response.data]
            except Exception as e:
                print(f"Error getting embeddings: {e}")
                # Return zero vectors on error
                return [np.zeros(config.VECTOR_DIMENSIONS, dtype=np.float32) for _ in texts]

    async def get_embeddings(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """
        Get embeddings for a list of texts with caching and batching.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar

        Returns:
            List of numpy arrays containing embeddings
        """
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        # Check cache first
        for i, text in enumerate(texts):
            if not text or text.strip() == "":
                results[i] = np.zeros(config.VECTOR_DIMENSIONS, dtype=np.float32)
                continue

            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                results[i] = self.cache[cache_key]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Process uncached texts in batches
        if texts_to_embed:
            batches = [texts_to_embed[i:i + config.EMBEDDING_BATCH_SIZE]
                      for i in range(0, len(texts_to_embed), config.EMBEDDING_BATCH_SIZE)]

            if show_progress:
                pbar = async_tqdm(total=len(texts_to_embed), desc="Getting embeddings")

            tasks = []
            for batch in batches:
                tasks.append(self._get_embedding_batch(batch))

            batch_results = await asyncio.gather(*tasks)

            # Flatten batch results and update cache
            flat_results = []
            for batch_result in batch_results:
                flat_results.extend(batch_result)

            for idx, text, embedding in zip(indices_to_embed, texts_to_embed, flat_results):
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding
                results[idx] = embedding
                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

            # Save cache after processing
            self._save_cache()

        return results

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text string."""
        embeddings = await self.get_embeddings([text], show_progress=False)
        return embeddings[0]


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[int]:
    """
    Implement Reciprocal Rank Fusion (RRF) for combining multiple rankings.

    Args:
        rankings: List of rankings, where each ranking is a list of document IDs
        k: Constant for RRF (default 60)

    Returns:
        Combined ranking of document IDs
    """
    scores = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)

    # Sort by score descending, then by doc_id for stability
    sorted_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [doc_id for doc_id, _ in sorted_docs]


def handle_edge_cases(value: Any) -> str:
    """
    Handle edge cases in data values (NaN, None, empty strings, etc.)

    Args:
        value: The value to process

    Returns:
        Cleaned string representation
    """
    if value is None:
        return ""

    # Handle pandas specific cases
    if hasattr(value, '__module__') and 'pandas' in str(value.__module__):
        import pandas as pd
        if pd.isna(value):
            return ""

    # Convert to string and handle edge cases
    str_value = str(value).strip()

    # Handle common null representations
    if str_value.lower() in ['nan', 'none', 'null', 'na', 'n/a']:
        return ""

    return str_value


def prepare_text_for_indexing(row: Dict[str, Any], columns: List[str]) -> str:
    """
    Prepare text from multiple columns for indexing.

    Args:
        row: Dictionary representing a data row
        columns: List of column names to include

    Returns:
        Combined text string for indexing
    """
    text_parts = []

    for col in columns:
        if col in row:
            cleaned = handle_edge_cases(row[col])
            if cleaned:
                text_parts.append(cleaned)

    return " ".join(text_parts)


async def batch_process_async(items: List[Any], process_func, batch_size: int, desc: str = "Processing"):
    """
    Process items in batches asynchronously with progress tracking.

    Args:
        items: List of items to process
        process_func: Async function to process each batch
        batch_size: Size of each batch
        desc: Description for progress bar

    Returns:
        List of processed results
    """
    results = []
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    pbar = async_tqdm(total=len(items), desc=desc)

    for batch in batches:
        batch_results = await process_func(batch)
        results.extend(batch_results)
        pbar.update(len(batch))

    pbar.close()
    return results