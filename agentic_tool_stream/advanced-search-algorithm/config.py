"""
Configuration file for the advanced search algorithm.
"""
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
DATA_PATH: str = os.getenv("DATA_PATH", "data/data.parquet")
INDEX_PATH: str = os.getenv("INDEX_PATH", "index/")
EMBEDDINGS_CACHE_PATH: str = os.getenv("EMBEDDINGS_CACHE_PATH", "cache/embeddings_cache.pkl")

# Embedding configuration
EMBEDDINGS_COLUMNS: List[str] = eval(os.getenv("EMBEDDINGS_COLUMNS", '["column1", "column2"]'))
VECTOR_DIMENSIONS: int = int(os.getenv("VECTOR_DIMENSIONS", "1536"))  # OpenAI ada-002 default

# OpenAI configuration
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "text-embedding-ada-002")

# Indexing configuration
REINDEX: bool = os.getenv("REINDEX", "False").lower() in ("true", "1", "yes")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
PARALLEL_WORKERS: int = int(os.getenv("PARALLEL_WORKERS", "4"))
TANTIVY_HEAP_SIZE: int = int(os.getenv("TANTIVY_HEAP_SIZE", "100000000"))  # 100MB default

# Search configuration
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "100"))
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))

# Performance requirements
MAX_SEARCH_TIME_MS: int = 150  # Maximum allowed search time in milliseconds

# Batch processing for embeddings
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

# Create necessary directories
def ensure_directories():
    """Ensure all required directories exist."""
    for path in [INDEX_PATH, Path(EMBEDDINGS_CACHE_PATH).parent]:
        Path(path).mkdir(parents=True, exist_ok=True)

ensure_directories()