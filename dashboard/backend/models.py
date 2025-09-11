from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class DatasetType(str, Enum):
    EXTERNAL_LOSS = "external_loss"
    INTERNAL_LOSS = "internal_loss"
    ISSUES = "issues"
    CONTROLS = "controls"

class DatasetRecord(BaseModel):
    id: int
    dataset_type: DatasetType
    description: str
    ai_taxonomy: str
    current_erms_taxonomy: str
    raw_meta_data: Optional[Dict[str, Any]] = None
    ai_root_cause: Optional[Dict[str, Any]] = None
    ai_enrichment: Optional[Dict[str, Any]] = None

class DatasetRecordSummary(BaseModel):
    id: int
    description: str
    ai_taxonomy: str
    current_erms_taxonomy: str

class PaginatedResponse(BaseModel):
    data: List[DatasetRecordSummary]
    total: int
    page: int
    page_size: int
    total_pages: int

class DatasetStats(BaseModel):
    dataset_type: DatasetType
    total_records: int
    unique_ai_taxonomies: int
    unique_erms_taxonomies: int

class FeedbackType(str, Enum):
    DROPDOWN = "dropdown"
    TEXT = "text"
    RATING = "rating"
    APPROVAL = "approval"
    ISSUE = "issue"

class FeedbackSubmission(BaseModel):
    record_id: Optional[int] = None  # Will be set by the API endpoint
    feedback_type: FeedbackType
    value: str
    additional_notes: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: str
    record_id: int
    feedback_type: FeedbackType
    value: str
    additional_notes: Optional[str] = None
    timestamp: datetime
    
class SearchFilters(BaseModel):
    search_query: Optional[str] = None
    ai_taxonomy: Optional[str] = None
    current_erms_taxonomy: Optional[str] = None