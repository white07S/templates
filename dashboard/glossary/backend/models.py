from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class Term(BaseModel):
    id: Optional[int] = None
    term: str
    definition: str
    synonyms: List[str] = Field(default_factory=list)
    createdAt: datetime = Field(default_factory=datetime.now)
    updatedAt: datetime = Field(default_factory=datetime.now)

class TermCreate(BaseModel):
    term: str
    definition: str
    synonyms: Optional[List[str]] = Field(default_factory=list)
    user_id: str

class TermUpdate(BaseModel):
    term: Optional[str] = None
    definition: Optional[str] = None
    synonyms: Optional[List[str]] = None
    user_id: str

class TermResponse(BaseModel):
    id: int
    term: str
    definition: str
    synonyms: List[str]
    createdAt: datetime
    updatedAt: datetime