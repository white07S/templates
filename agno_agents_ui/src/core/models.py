"""Data models for search engine."""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from enum import Enum


class SearchMode(str, Enum):
    """Search modes supported by the engine."""

    DATAFRAME = "dataframe"
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    """Request model for search operations."""

    mode: SearchMode
    column_value_pairs: Optional[Dict[str, Any]] = Field(default=None)
    keywords: Optional[List[str]] = Field(default=None)
    vector: Optional[List[float]] = Field(default=None)
    top_k: int = Field(default=5)

    class Config:
        use_enum_values = True


class SearchResult(BaseModel):
    """Result model for search operations."""

    status: str  # "success" or "failed"
    count: int
    results: List[Dict[str, Any]]
    time_taken: float  # in seconds

    class Config:
        validate_assignment = True