from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class TaskStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"

class TaskSubmit(BaseModel):
    username: str
    secret_code: str
    data_type: str
    tasks: List[str]

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus

class DataType(BaseModel):
    name: str
    tasks: List[str]

class AvailableTasksResponse(BaseModel):
    data_types: List[DataType]

class UserTaskInfo(BaseModel):
    task_id: str
    data_type: str
    tasks: List[str]
    status: TaskStatus
    submitted_at: datetime
    file_name: str

class UserTasksResponse(BaseModel):
    tasks: List[UserTaskInfo]