import os
from typing import Set

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10080  # 7 days
REFRESH_TOKEN_EXPIRE_MINUTES = 129600  # 90 days

# Database
DATABASE_FILE = "auth_system.db"

# Azure Login Configuration
AZ_LOGIN_TIMEOUT = 300

# RBAC Configuration
ALLOWED_EMAILS: Set[str] = {
    "alice@example.com",
    "bob@example.com",
    "preetam.sharma@outlook.com",
    "sharmapreetam.uk@outlook.com"
}

# Role to email mapping
ROLE_MAPPING = {
    "preetam.sharma@outlook.com": ["admin", "user"],
    "sharmapreetam.uk@outlook.com": ["user"],
    "alice@example.com": ["user"],
    "bob@example.com": ["user"]
}