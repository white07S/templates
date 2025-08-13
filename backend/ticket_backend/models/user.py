from pydantic import BaseModel
from typing import List, Dict, Any

class DataTask(BaseModel):
    data_type: str
    tasks: List[str]

class User(BaseModel):
    username: str
    secret_code: str = "123456789"
    authorized_data_tasks: List[DataTask]

class UserCreate(BaseModel):
    username: str
    secret_code: str = "123456789"
    authorized_data_tasks: List[DataTask]

class UserRemove(BaseModel):
    username: str

class UserAuth(BaseModel):
    username: str
    secret_code: str

class APIResponse(BaseModel):
    status: str
    message: str