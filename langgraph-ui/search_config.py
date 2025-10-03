"""
Configuration file for the advanced search engine.
"""
import os
from pathlib import Path
from typing import List, Optional

class SearchConfig:
    """Configuration class for search engine settings."""
    
    # Data paths
    DATA_PATH: str = os.getenv("DATA_PATH", "data/input.parquet")
    INDEX_PATH: str = os.getenv("INDEX_PATH", "data/indices")
    CACHE_PATH: str = os.getenv("CACHE_PATH", "data/cache")
    
    # Tantivy index path
    TANTIVY_INDEX_PATH: str = os.path.join(INDEX_PATH, "tantivy")
    
    # FAISS index path
    FAISS_INDEX_PATH: str = os.path.join(INDEX_PATH, "faiss")
    
    # Embedding settings
    EMBEDDINGS_COLUMNS: List[str] = ["column_1", "column_2"]  # Columns to generate embeddings for
    VECTOR_DIMENSIONS: int = 1536  # OpenAI text-embedding-3-small dimensions
    
    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "text-embedding-3-small"
    OPENAI_BATCH_SIZE: int = 100  # Batch size for embedding requests
    OPENAI_MAX_RETRIES: int = 3
    OPENAI_RETRY_DELAY: float = 1.0
    
    # Indexing settings
    REINDEX: bool = True  # Set to False to use existing indices
    CHUNK_SIZE: int = 1000  # Process data in chunks
    PARALLEL_WORKERS: int = os.cpu_count() or 4  # Number of parallel workers
    
    # Tantivy settings
    TANTIVY_HEAP_SIZE: Optional[int] = None  # Will be calculated based on file size
    TANTIVY_NUM_THREADS: int = 4
    TANTIVY_MEMORY_BUDGET: int = 1_000_000_000  # 1GB
    
    # Search settings
    DEFAULT_TOP_K: int = 10  # Default number of results to return
    RRF_K: int = 60  # Reciprocal Rank Fusion constant
    SEARCH_TIMEOUT: float = 0.15  # 150ms timeout for search operations
    
    # Performance settings
    USE_MEMORY_MAP: bool = True  # Use memory mapping for large files
    ENABLE_CACHE: bool = True  # Enable embedding cache
    CACHE_TTL: int = 86400 * 30  # Cache TTL in seconds (30 days)
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Check if data path exists
        if not Path(cls.DATA_PATH).exists():
            errors.append(f"Data path does not exist: {cls.DATA_PATH}")
        
        # Check OpenAI API key
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")
        
        # Check vector dimensions
        if cls.VECTOR_DIMENSIONS <= 0:
            errors.append("VECTOR_DIMENSIONS must be positive")
        
        # Check chunk size
        if cls.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be positive")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.INDEX_PATH,
            cls.TANTIVY_INDEX_PATH,
            cls.FAISS_INDEX_PATH,
            cls.CACHE_PATH
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def calculate_tantivy_heap_size(cls, file_size_mb: float) -> int:
        """Calculate Tantivy heap size based on file size."""
        # Rule of thumb: 4x file size for optimal performance
        return int(file_size_mb * 4 * 1024 * 1024)
    
    @classmethod
    def get_search_columns(cls, df) -> List[str]:
        """Get columns suitable for text search from dataframe."""
        # Filter columns that contain text data
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains text (not all nulls)
                if df[col].notna().any():
                    text_columns.append(col)
        return text_columns
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

# Create directories on import
SearchConfig.create_directories()