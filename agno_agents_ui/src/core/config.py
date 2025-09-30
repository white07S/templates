"""Configuration models for the search engine."""

from typing import List, Optional
from pydantic import BaseModel, Field
import os


class IndexConfig(BaseModel):
    """Configuration for indexing pipeline."""

    # OpenAI settings
    openai_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_api_base: Optional[str] = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")

    # Index paths
    index_path: str = Field(default="data/index", description="Path where index files will be stored")
    data_path: str = Field(default="data/raw/data.parquet", description="Path of raw data parquet file")

    # Embedding configuration
    embedding_columns: List[str] = Field(default_factory=list, description="Columns to convert to embeddings")
    vector_dimension: int = Field(default=4096, description="Dimension of embedding vectors")

    # Processing flags
    re_index: bool = Field(default=False, description="Force re-indexing even if index exists")
    batch_size: int = Field(default=1000, description="Batch size for processing")

    # Cache settings
    embedding_cache_path: str = Field(default="data/index/embedding_cache.pkl", description="Path to embedding cache")

    class Config:
        validate_assignment = True
        extra = "forbid"


class SearchConfig(BaseModel):
    """Configuration for search operations."""

    index_path: str = Field(default="data/index", description="Path where index files are stored")
    top_k: int = Field(default=5, description="Number of top results to return")
    vector_dimension: int = Field(default=4096, description="Dimension of embedding vectors")

    class Config:
        validate_assignment = True
        extra = "forbid"