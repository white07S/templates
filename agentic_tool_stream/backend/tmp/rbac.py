import logging
from datetime import datetime
from typing import List, Optional

from fastapi import Cookie, Depends, HTTPException, status

from auth import hash_token, verify_fingerprint
from database import db
from models import TokenData

logger = logging.getLogger(__name__)


async def get_current_user(
    graph_access_token: Optional[str] = Cookie(None),
    fingerprint: Optional[str] = Cookie(None),
) -> TokenData:
    """
    Resolve the current authenticated user using the stored Graph access token.
    Validates token existence, expiration, and browser fingerprint.
    """
    if not graph_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_hash = hash_token(graph_access_token)
    session = db.get_session_by_token_hash(token_hash)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expires_at_str = session.get("token_expires_at")
    if expires_at_str:
        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.utcnow() > expires_at:
                db.delete_session_by_token_hash(token_hash)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        except ValueError:
            logger.warning(
                "Invalid token_expires_at format for session %s", session.get("id")
            )

    stored_fingerprint = session.get("fingerprint")
    if stored_fingerprint and fingerprint:
        if not verify_fingerprint(stored_fingerprint, fingerprint):
            logger.warning(
                "Fingerprint mismatch for user session %s", session.get("id")
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Device fingerprint mismatch",
            )

    user = db.get_user_by_id(session["user_id"])
    if not user or not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return TokenData(
        email=user["email"],
        user_id=user["id"],
        roles=user["roles"],
        session_id=session["id"],
        device_id=session.get("device_id"),
        fingerprint=stored_fingerprint,
        token_expires_at=expires_at_str,
        azure_object_id=session.get("azure_object_id"),
    )


async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user),
) -> TokenData:
    """Dependency to ensure user is active."""
    return current_user


class RoleChecker:
    """Dependency class to check if user has required roles."""

    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    async def __call__(
        self, current_user: TokenData = Depends(get_current_user)
    ) -> TokenData:
        """Check if user has any of the required roles."""
        if not any(role in current_user.roles for role in self.allowed_roles):
            logger.warning(
                "User %s with roles %s attempted to access resource requiring %s",
                current_user.email,
                current_user.roles,
                self.allowed_roles,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(self.allowed_roles)}",
            )
        return current_user


# Predefined role checkers
require_admin = RoleChecker(["admin"])
require_user = RoleChecker(["user", "admin"])  # Both user and admin can access


def has_permission(user: TokenData, required_roles: List[str]) -> bool:
    """Check if user has any of the required roles."""
    return any(role in user.roles for role in required_roles)
