"""
Test routes to demonstrate RBAC (Role-Based Access Control)
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict
from models import TokenData
from rbac import get_current_user, require_admin, require_user

router = APIRouter(prefix="/api/test", tags=["Testing"])

# ============= PUBLIC ROUTE =============

@router.get("/public")
async def public_endpoint():
    """Public endpoint - No authentication required"""
    return {
        "message": "This is a public endpoint accessible to everyone",
        "access_level": "public",
        "authentication_required": False
    }

# ============= USER LEVEL ROUTES =============

@router.get("/user/profile")
async def get_user_profile(current_user: TokenData = Depends(require_user)):
    """User endpoint - Requires user or admin role"""
    return {
        "message": "User profile data retrieved successfully",
        "access_level": "user",
        "user": {
            "email": current_user.email,
            "user_id": current_user.user_id,
            "roles": current_user.roles
        },
        "authentication_required": True,
        "allowed_roles": ["user", "admin"]
    }

@router.post("/user/data")
async def create_user_data(
    data: Dict,
    current_user: TokenData = Depends(require_user)
):
    """User POST endpoint - Requires user or admin role"""
    return {
        "message": "Data created successfully",
        "access_level": "user",
        "created_by": current_user.email,
        "data_received": data,
        "authentication_required": True,
        "allowed_roles": ["user", "admin"]
    }

# ============= ADMIN ONLY ROUTES =============

@router.get("/admin/users")
async def get_all_users(current_user: TokenData = Depends(require_admin)):
    """Admin endpoint - Requires admin role only"""
    return {
        "message": "All users list retrieved successfully",
        "access_level": "admin",
        "accessed_by": {
            "email": current_user.email,
            "roles": current_user.roles
        },
        "users": [
            {"id": 1, "email": "user1@example.com", "role": "user"},
            {"id": 2, "email": "admin@example.com", "role": "admin"},
            {"id": 3, "email": current_user.email, "role": "admin"}
        ],
        "authentication_required": True,
        "allowed_roles": ["admin"]
    }

@router.post("/admin/settings")
async def update_settings(
    settings: Dict,
    current_user: TokenData = Depends(require_admin)
):
    """Admin POST endpoint - Requires admin role only"""
    return {
        "message": "System settings updated successfully",
        "access_level": "admin",
        "updated_by": current_user.email,
        "settings_changed": settings,
        "authentication_required": True,
        "allowed_roles": ["admin"]
    }

@router.get("/admin/stats")
async def get_system_stats(current_user: TokenData = Depends(require_admin)):
    """Admin endpoint - Get system statistics"""
    return {
        "message": "System statistics retrieved successfully",
        "access_level": "admin",
        "stats": {
            "total_users": 156,
            "active_sessions": 42,
            "total_requests_today": 8934,
            "server_uptime": "99.9%",
            "database_size_mb": 245
        },
        "accessed_by": current_user.email,
        "authentication_required": True,
        "allowed_roles": ["admin"]
    }

# ============= ROLE INFORMATION ENDPOINT =============

@router.get("/roles/info")
async def get_role_info(current_user: TokenData = Depends(get_current_user)):
    """Get information about current user's roles and accessible endpoints"""
    is_admin = "admin" in current_user.roles
    is_user = "user" in current_user.roles or is_admin

    return {
        "message": "Role information retrieved successfully",
        "current_user": {
            "email": current_user.email,
            "user_id": current_user.user_id,
            "roles": current_user.roles,
            "is_admin": is_admin,
            "is_user": is_user
        },
        "accessible_endpoints": {
            "public": [
                {"method": "GET", "path": "/api/test/public", "description": "Public endpoint"}
            ],
            "user_level": [
                {"method": "GET", "path": "/api/test/user/profile", "description": "Get user profile", "accessible": is_user},
                {"method": "POST", "path": "/api/test/user/data", "description": "Create user data", "accessible": is_user}
            ],
            "admin_only": [
                {"method": "GET", "path": "/api/test/admin/users", "description": "Get all users", "accessible": is_admin},
                {"method": "POST", "path": "/api/test/admin/settings", "description": "Update settings", "accessible": is_admin},
                {"method": "GET", "path": "/api/test/admin/stats", "description": "Get system stats", "accessible": is_admin}
            ]
        }
    }
