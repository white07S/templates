import logging
from typing import List, Optional
from fastapi import HTTPException, status, Cookie, Depends
from models import TokenData
from auth import decode_token, verify_fingerprint
from database import db

logger = logging.getLogger(__name__)

async def get_current_user(
    access_token: Optional[str] = Cookie(None),
    fingerprint: Optional[str] = Cookie(None)
) -> TokenData:
    """
    Dependency to get current authenticated user from cookie.
    Validates token and fingerprint.
    """
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = decode_token(access_token)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify fingerprint
    if fingerprint and token_data.fingerprint:
        if not verify_fingerprint(token_data.fingerprint, fingerprint):
            logger.warning(f"Fingerprint mismatch for user {token_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Device fingerprint mismatch",
            )
    
    # Verify user exists and is active
    user = db.get_user_by_id(token_data.user_id)
    if not user or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    return token_data

async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """Dependency to ensure user is active"""
    return current_user

class RoleChecker:
    """Dependency class to check if user has required roles"""
    
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles
    
    async def __call__(self, current_user: TokenData = Depends(get_current_user)) -> TokenData:
        """Check if user has any of the required roles"""
        if not any(role in current_user.roles for role in self.allowed_roles):
            logger.warning(
                f"User {current_user.email} with roles {current_user.roles} "
                f"attempted to access resource requiring {self.allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(self.allowed_roles)}"
            )
        return current_user

# Predefined role checkers
require_admin = RoleChecker(["admin"])
require_user = RoleChecker(["user", "admin"])  # Both user and admin can access

def has_permission(user: TokenData, required_roles: List[str]) -> bool:
    """Check if user has any of the required roles"""
    return any(role in user.roles for role in required_roles)