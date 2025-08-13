import json
import os
from typing import Dict, List, Optional
from ticket_backend.models.user import User, DataTask

class AuthManager:
    def __init__(self, users_file: str = "data/users.json"):
        self.users_file = users_file
        self.users: Dict[str, User] = {}
        self._load_users()
    
    def _load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    users_data = json.load(f)
                    for username, user_data in users_data.items():
                        self.users[username] = User(**user_data)
            except (json.JSONDecodeError, FileNotFoundError):
                self.users = {}
    
    def _save_users(self):
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        with open(self.users_file, 'w') as f:
            users_data = {
                username: user.dict() 
                for username, user in self.users.items()
            }
            json.dump(users_data, f, indent=2)
    
    def add_user(self, user: User) -> bool:
        self.users[user.username] = user
        self._save_users()
        return True
    
    def remove_user(self, username: str) -> bool:
        if username in self.users:
            del self.users[username]
            self._save_users()
            return True
        return False
    
    def authenticate_user(self, username: str, secret_code: str) -> Optional[User]:
        user = self.users.get(username)
        if user and user.secret_code == secret_code:
            return user
        return None
    
    def is_user_authorized(self, username: str, secret_code: str, data_type: str, tasks: List[str]) -> bool:
        user = self.authenticate_user(username, secret_code)
        if not user:
            return False
        
        for data_task in user.authorized_data_tasks:
            if data_task.data_type == data_type:
                return all(task in data_task.tasks for task in tasks)
        
        return False

auth_manager = AuthManager()