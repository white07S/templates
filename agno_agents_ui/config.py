"""Configuration management for the chat backend system."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings and configuration."""
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    chat_model: str = Field(default="gpt-4o", env="CHAT_MODEL")
    embedding_model: str = Field(default="text-embedding-3-large", env="EMBEDDING_MODEL")

    # Model Context Windows
    model_context_window: int = Field(default=128000, env="MODEL_CONTEXT_WINDOW")
    embedding_dimension: int = Field(default=3072, env="EMBEDDING_DIMENSION")

    # Token Budget Configuration
    max_conversation_tokens: int = Field(default=8000, env="MAX_CONVERSATION_TOKENS")
    max_memory_tokens: int = Field(default=4000, env="MAX_MEMORY_TOKENS")
    system_prompt_tokens: int = Field(default=500, env="SYSTEM_PROMPT_TOKENS")

    # Memory Configuration
    memory_compression_ratio: float = Field(default=5.0, env="MEMORY_COMPRESSION_RATIO")
    memory_importance_threshold: float = Field(default=0.7, env="MEMORY_IMPORTANCE_THRESHOLD")
    temporal_decay_lambda: float = Field(default=0.5, env="TEMPORAL_DECAY_LAMBDA")

    # Storage Configuration
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    tinydb_conversations_path: Path = Field(default=Path("./data/conversations.json"))
    tinydb_memories_path: Path = Field(default=Path("./data/memories.json"))
    tantivy_index_path: Path = Field(default=Path("./data/tantivy_index"))
    faiss_index_path: Path = Field(default=Path("./data/faiss_index"))

    # Retrieval Configuration
    hybrid_search_k: int = Field(default=60, env="HYBRID_SEARCH_K")
    max_search_results: int = Field(default=20, env="MAX_SEARCH_RESULTS")
    lexical_weight: float = Field(default=0.3, env="LEXICAL_WEIGHT")
    semantic_weight: float = Field(default=0.7, env="SEMANTIC_WEIGHT")

    # Performance Configuration
    enable_async_memory_processing: bool = Field(default=True)
    memory_extraction_batch_size: int = Field(default=10)
    faiss_nprobe: int = Field(default=10)
    faiss_nclusters: int = Field(default=100)

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")

    # ReAct Agent Configuration
    enable_react_agent: bool = Field(default=True, env="ENABLE_REACT_AGENT")
    react_complexity_threshold: float = Field(default=0.7, env="REACT_COMPLEXITY_THRESHOLD")
    react_max_actions: int = Field(default=10, env="REACT_MAX_ACTIONS")
    react_parallel_limit: int = Field(default=5, env="REACT_PARALLEL_LIMIT")

    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.tantivy_index_path.mkdir(exist_ok=True, parents=True)
        self.faiss_index_path.mkdir(exist_ok=True, parents=True)

# Global settings instance
settings = Settings()
settings.setup_directories()