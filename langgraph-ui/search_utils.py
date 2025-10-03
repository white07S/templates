"""
Utility functions for the search engine including optimized async OpenAI embedding client.
"""
import asyncio
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm

# OpenAI imports
try:
    from openai import AsyncOpenAI
except ImportError:
    print("Please install openai: uv add openai")
    raise

from config import SearchConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=SearchConfig.LOG_LEVEL, format=SearchConfig.LOG_FORMAT)


class EmbeddingCache:
    """Simple file-based cache for embeddings."""
    
    def __init__(self, cache_dir: str = SearchConfig.CACHE_PATH):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, np.ndarray]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    logger.info(f"Loaded {len(cache)} cached embeddings")
                    return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get_hash(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        if not SearchConfig.ENABLE_CACHE:
            return None
        hash_key = self.get_hash(text)
        return self.cache.get(hash_key)
    
    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        if not SearchConfig.ENABLE_CACHE:
            return
        hash_key = self.get_hash(text)
        self.cache[hash_key] = embedding
    
    def batch_get(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Get multiple embeddings from cache."""
        results = {}
        for text in texts:
            cached = self.get(text)
            if cached is not None:
                results[text] = cached
        return results
    
    def batch_set(self, embeddings: Dict[str, np.ndarray]):
        """Store multiple embeddings in cache."""
        for text, embedding in embeddings.items():
            self.set(text, embedding)
        self._save_cache()


class OptimizedEmbeddingClient:
    """Optimized async OpenAI embedding client with batching and caching."""
    
    def __init__(self, api_key: str = SearchConfig.OPENAI_API_KEY):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = SearchConfig.OPENAI_MODEL
        self.batch_size = SearchConfig.OPENAI_BATCH_SIZE
        self.cache = EmbeddingCache()
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    
    async def _get_embedding_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts from OpenAI."""
        async with self.semaphore:
            for attempt in range(SearchConfig.OPENAI_MAX_RETRIES):
                try:
                    response = await self.client.embeddings.create(
                        input=texts,
                        model=self.model
                    )
                    embeddings = [np.array(item.embedding) for item in response.data]
                    return embeddings
                except Exception as e:
                    if attempt < SearchConfig.OPENAI_MAX_RETRIES - 1:
                        await asyncio.sleep(SearchConfig.OPENAI_RETRY_DELAY * (2 ** attempt))
                    else:
                        logger.error(f"Failed to get embeddings after {SearchConfig.OPENAI_MAX_RETRIES} attempts: {e}")
                        raise
    
    async def get_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Get embeddings for a list of texts with caching and batching.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Check cache
        cached_embeddings = self.cache.batch_get(texts)
        texts_to_embed = [t for t in texts if t not in cached_embeddings]
        
        if texts_to_embed:
            # Process in batches
            all_embeddings = []
            batches = [
                texts_to_embed[i:i + self.batch_size]
                for i in range(0, len(texts_to_embed), self.batch_size)
            ]
            
            # Create progress bar
            if show_progress:
                pbar = tqdm(total=len(texts_to_embed), desc="Generating embeddings", unit="text")
            
            # Process batches concurrently
            tasks = []
            for batch in batches:
                task = asyncio.create_task(self._get_embedding_batch(batch))
                tasks.append((batch, task))
            
            # Collect results
            new_embeddings = {}
            for batch, task in tasks:
                embeddings = await task
                for text, embedding in zip(batch, embeddings):
                    new_embeddings[text] = embedding
                    if show_progress:
                        pbar.update(len(batch))
            
            if show_progress:
                pbar.close()
            
            # Cache new embeddings
            self.cache.batch_set(new_embeddings)
            
            # Merge with cached
            cached_embeddings.update(new_embeddings)
        
        # Return in original order
        result = np.array([cached_embeddings[text] for text in texts])
        return result
    
    async def get_single_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        embeddings = await self.get_embeddings([text], show_progress=False)
        return embeddings[0]


def clean_text(text: Any) -> str:
    """Clean and normalize text for indexing."""
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string
    text = str(text)
    
    # Basic cleaning
    text = text.strip()
    text = ' '.join(text.split())  # Normalize whitespace
    
    return text


def handle_edge_cases(value: Any) -> Any:
    """Handle edge cases in data values."""
    import pandas as pd
    import numpy as np
    
    # Handle NaN, None, empty strings
    if pd.isna(value) or value is None:
        return ""
    
    # Handle numpy types
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    
    # Handle dates
    if pd.api.types.is_datetime64_any_dtype(type(value)):
        return str(value)
    
    # Convert to string for text fields
    return str(value)


class DataProcessor:
    """Process and prepare data for indexing."""
    
    @staticmethod
    def load_parquet(file_path: str, chunk_size: Optional[int] = None):
        """Load parquet file with optional chunking."""
        import pandas as pd
        import pyarrow.parquet as pq
        
        if chunk_size:
            # Read in chunks for large files
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                yield df
        else:
            # Read entire file
            df = pd.read_parquet(file_path)
            yield df
    
    @staticmethod
    def prepare_text_for_indexing(df, columns: List[str]) -> Dict[int, str]:
        """Prepare text from multiple columns for indexing."""
        texts = {}
        for idx, row in df.iterrows():
            combined_text = []
            for col in columns:
                if col in row:
                    value = handle_edge_cases(row[col])
                    if value:
                        combined_text.append(f"{col}: {value}")
            texts[idx] = " ".join(combined_text)
        return texts
    
    @staticmethod
    def validate_dataframe(df) -> bool:
        """Validate dataframe for indexing."""
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        if len(df.columns) < 1:
            logger.error("DataFrame has no columns")
            return False
        
        # Check for at least one non-null value
        if df.isna().all().all():
            logger.error("DataFrame contains only null values")
            return False
        
        return True


class RRFScorer:
    """Reciprocal Rank Fusion scorer for combining search results."""
    
    def __init__(self, k: int = SearchConfig.RRF_K):
        self.k = k
    
    def score(self, rankings: List[List[tuple]]) -> List[tuple]:
        """
        Apply RRF to multiple ranked lists.
        
        Args:
            rankings: List of ranked results, each is a list of (doc_id, score) tuples
        
        Returns:
            Combined ranked list of (doc_id, rrf_score) tuples
        """
        rrf_scores = {}
        
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1.0 / (self.k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_results


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Start a timer for a metric."""
        self.metrics[name] = {'start': time.time()}
    
    def end_timer(self, name: str):
        """End a timer and calculate duration."""
        if name in self.metrics:
            self.metrics[name]['end'] = time.time()
            self.metrics[name]['duration'] = self.metrics[name]['end'] - self.metrics[name]['start']
            return self.metrics[name]['duration']
        return 0
    
    def log_metrics(self):
        """Log all collected metrics."""
        logger.info("Performance Metrics:")
        for name, data in self.metrics.items():
            if 'duration' in data:
                logger.info(f"  {name}: {data['duration']:.3f}s")
    
    def get_total_time(self) -> float:
        """Get total time for all metrics."""
        return sum(m.get('duration', 0) for m in self.metrics.values())


# Import pandas here to avoid circular imports
import pandas as pd