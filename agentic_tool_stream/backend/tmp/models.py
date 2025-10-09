from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Azure Auth Models
class AuthStartResponse(BaseModel):
    session_id: str
    user_code: str
    verification_uri: str

class AuthStatusResponse(BaseModel):
    status: str  # 'pending', 'completed', 'timeout', 'error'
    authorized: Optional[bool] = None
    email: Optional[str] = None
    user_name: Optional[str] = None
    message: Optional[str] = None

class AuthCompleteRequest(BaseModel):
    session_id: str
    fingerprint: str

# Token Models
class TokenResponse(BaseModel):
    token_type: str = "bearer"
    token_expires_at: str
    roles: List[str]
    user: "UserResponse"

class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[str] = None
    roles: List[str] = []
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    fingerprint: Optional[str] = None
    token_expires_at: Optional[str] = None
    azure_object_id: Optional[str] = None

# User Models
class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    roles: List[str]
    is_active: bool
    created_at: str
    last_login: Optional[str] = None

class UserCreate(BaseModel):
    email: str
    username: str
    roles: List[str]

# Device Models
class DeviceInfo(BaseModel):
    device_id: str
    fingerprint: str
    created_at: str
    last_used_at: Optional[str] = None


TokenResponse.model_rebuild()
