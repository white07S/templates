"""
Optimized OpenAI Embeddings Client
Supports batch processing, caching, and fallback to random vectors for testing
"""

import numpy as np
from typing import List, Dict, Any, Optional
import os
import json
import hashlib
# Removed ThreadPoolExecutor import - no longer needed for sequential processing
import pickle
from tqdm import tqdm
import time

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class EmbeddingsClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "text-embedding-3-large",
        dimension: int = 4096,
        cache_dir: str = "cache/embeddings",
        use_cache: bool = True,
        max_batch_size: int = 100,
        max_workers: int = 1,  # Changed to sequential processing
        batch_delay: float = 0.1  # Add delay between batches to avoid rate limits
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_url = api_url or os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
        self.model = model
        self.dimension = dimension
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.batch_delay = batch_delay

        # Initialize OpenAI client if available
        self.client = None
        if OpenAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)

        # Create cache directory
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, f"{model}_{dimension}.pkl")
            self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, np.ndarray]:
        """Load embeddings cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
        return {}

    def _save_cache(self):
        """Save embeddings cache to disk"""
        if self.use_cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.sha256(f"{self.model}:{self.dimension}:{text}".encode()).hexdigest()

    def _generate_random_embedding(self, dimension: int = None) -> np.ndarray:
        """Generate random embedding vector for testing"""
        dim = dimension or self.dimension
        # Generate unit vector for consistency
        vec = np.random.randn(dim)
        return vec / np.linalg.norm(vec)

    def _get_openai_embedding(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings from OpenAI API"""
        if not self.client:
            # Fallback to random embeddings if no client
            return [self._generate_random_embedding() for _ in texts]

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
                dimensions=self.dimension
            )
            return [np.array(data.embedding) for data in response.data]
        except Exception as e:
            print(f"Warning: OpenAI API error: {e}. Using random embeddings.")
            return [self._generate_random_embedding() for _ in texts]

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text"""
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                return self.cache[cache_key]

        embedding = self._get_openai_embedding([text])[0]

        if self.use_cache:
            self.cache[cache_key] = embedding

        return embedding

    def get_embeddings_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
        desc: str = "Generating embeddings"
    ) -> List[np.ndarray]:
        """Get embeddings for multiple texts with batch processing"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        if self.use_cache:
            for i, text in enumerate(texts):
                if text is None or text == "":
                    embeddings.append(None)
                    continue

                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    embeddings.append(self.cache[cache_key])
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = [t for t in texts if t is not None and t != ""]
            uncached_indices = [i for i, t in enumerate(texts) if t is not None and t != ""]
            embeddings = [None] * len(texts)

        # Process uncached texts
        if uncached_texts:
            # Split into batches
            batches = [
                uncached_texts[i:i + self.max_batch_size]
                for i in range(0, len(uncached_texts), self.max_batch_size)
            ]

            # Process batches sequentially
            if show_progress:
                pbar = tqdm(total=len(uncached_texts), desc=desc)

            batch_idx = 0
            for batch_num, batch in enumerate(batches):
                # Add delay between batches (except for the first one)
                if batch_num > 0 and self.batch_delay > 0:
                    time.sleep(self.batch_delay)

                # Process batch sequentially
                batch_embeddings = self._get_openai_embedding(batch)
                batch_size = len(batch_embeddings)

                for i, embedding in enumerate(batch_embeddings):
                    idx = uncached_indices[batch_idx + i]
                    embeddings[idx] = embedding

                    # Update cache
                    if self.use_cache:
                        text = uncached_texts[batch_idx + i]
                        cache_key = self._get_cache_key(text)
                        self.cache[cache_key] = embedding

                batch_idx += batch_size

                if show_progress:
                    pbar.update(batch_size)

            if show_progress:
                pbar.close()

        # Replace None with zero vectors for consistency
        for i, emb in enumerate(embeddings):
            if emb is None:
                embeddings[i] = np.zeros(self.dimension)

        # Save cache periodically
        if self.use_cache and uncached_texts:
            self._save_cache()

        return embeddings

    def clear_cache(self):
        """Clear the embeddings cache"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print(f"Cache cleared for {self.model}_{self.dimension}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_file": self.cache_file,
            "cache_enabled": self.use_cache,
            "model": self.model,
            "dimension": self.dimension
        }

# Convenience function for testing
def create_test_client(dimension: int = 4096) -> EmbeddingsClient:
    """Create a test client that uses random embeddings"""
    return EmbeddingsClient(
        api_key=None,  # No API key for testing
        dimension=dimension,
        use_cache=True
    )