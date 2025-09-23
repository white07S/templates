"""Data models and schemas for the chat backend system."""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid

class MessageRole(str, Enum):
    """Enum for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MemoryType(str, Enum):
    """Types of memories in the system."""
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"

class IntentType(str, Enum):
    """Types of user query intents."""
    META_QUERY = "meta_query"
    SEMANTIC_SEARCH = "semantic_search"
    TEMPORAL_QUERY = "temporal_query"
    MEMORY_RETRIEVAL = "memory_retrieval"
    GENERAL = "general"

class Message(BaseModel):
    """Single message in a conversation."""
    model_config = ConfigDict(use_enum_values=True)

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    embedding_id: Optional[int] = None
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConversationDocument(BaseModel):
    """Document schema for conversation storage in TinyDB."""
    session_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Message] = Field(default_factory=list)
    memory_state: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        data['timestamp'] = self.timestamp.isoformat()
        data['messages'] = [
            {
                **msg,
                'timestamp': msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp']
            } for msg in data['messages']
        ]
        return data

class Memory(BaseModel):
    """Memory document schema."""
    model_config = ConfigDict(use_enum_values=True)

    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    memory_type: MemoryType
    content: str
    compressed_content: Optional[str] = None
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    access_frequency: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    faiss_id: Optional[int] = None
    keywords: List[str] = Field(default_factory=list)
    source_session_id: Optional[str] = None
    entity_references: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        data['last_modified'] = self.last_modified.isoformat()
        # memory_type will already be a string due to use_enum_values=True
        return data

class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    session_id: str
    user_id: str
    message: str
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    session_id: str
    user_id: str
    response: str
    memories_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    """Result from hybrid search."""
    content: str
    score: float
    source: Literal["lexical", "semantic", "hybrid"]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = None

class TokenBudget(BaseModel):
    """Token budget allocation for context assembly."""
    conversation_tokens: int
    memory_tokens: int
    system_tokens: int
    total_budget: int

    @property
    def remaining_tokens(self) -> int:
        return self.total_budget - (self.conversation_tokens + self.memory_tokens + self.system_tokens)

class PerformanceMetrics(BaseModel):
    """Performance monitoring metrics."""
    operation: str
    avg_time_ms: float
    total_calls: int
    total_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)