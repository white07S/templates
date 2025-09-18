from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime, timezone
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[Dict[str, Any]] = None


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]


class ToolResult(BaseModel):
    tool_call_id: str
    result: Any
    error: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: UUID
    user_id: str
    message: str
    reasoning_mode: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: float = 0.7
    max_tokens: int = 1000


class ChatResponse(BaseModel):
    session_id: UUID
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    role: MessageRole = MessageRole.ASSISTANT
    tool_calls: Optional[List[ToolCall]] = None
    memory_hints: Optional[List[str]] = None
    reasoning_steps: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionInfo(BaseModel):
    session_id: UUID
    user_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int
    memory_count: int


class MemoryItem(BaseModel):
    id: str
    text: str
    kind: str  # "profile" | "episodic"
    user_id: str
    session_id: Optional[UUID] = None
    created_at: datetime
    last_seen_at: datetime
    importance: float = 0.5
    frequency: int = 1
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class ReasoningStep(BaseModel):
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))