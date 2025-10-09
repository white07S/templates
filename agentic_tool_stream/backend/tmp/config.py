import os
from pathlib import Path
from typing import Dict, List

# Azure CLI session isolation
AZURE_CONFIG_BASE_DIR = Path(os.getenv("AZURE_CONFIG_BASE_DIR", "/tmp/azcfg"))

# Azure login
AZ_LOGIN_TIMEOUT = int(os.getenv("AZ_LOGIN_TIMEOUT", "300"))

# Microsoft Graph access
GRAPH_RESOURCE = os.getenv("GRAPH_RESOURCE", "https://graph.microsoft.com")
GRAPH_TOKEN_TTL_MINUTES = int(os.getenv("GRAPH_TOKEN_TTL_MINUTES", "1440"))  # 24 hours

# Database (TinyDB JSON file)
DATABASE_FILE = os.getenv("DATABASE_FILE", "auth_system.json")

# Roles mapped to Azure AD group IDs
AZURE_ROLE_GROUP_MAPPING: Dict[str, List[str]] = {
    "admin": [gid for gid in os.getenv("AZURE_ADMIN_GROUP_IDS", "").split(",") if gid],
    "user": [gid for gid in os.getenv("AZURE_USER_GROUP_IDS", "").split(",") if gid],
}

# Default role assigned when no group mapping matches
DEFAULT_ROLE = os.getenv("DEFAULT_ROLE", "user").strip() or "user"
