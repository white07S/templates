from fastapi import FastAPI, Request, HTTPException, Depends, Response, Cookie
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional

import config
from models import (
    AuthStartResponse, AuthStatusResponse, AuthCompleteRequest,
    TokenResponse, UserResponse
)
from auth import (
    auth_sessions, run_az_login, create_access_token, create_refresh_token,
    decode_token, hash_token, set_auth_cookies, clear_auth_cookies, verify_fingerprint
)
from rbac import get_current_user, require_admin, require_user
from database import db
from test_routes import router as test_router

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Azure Auth with RBAC")

# Include test routes
app.include_router(test_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= AZURE AUTH API =============

@app.post("/api/authorize/start", response_model=AuthStartResponse)
async def start_authorization():
    """Start the Azure device code authorization process"""
    session_id = str(uuid.uuid4())
    logger.info(f"Starting authorization session {session_id}")
    
    auth_sessions[session_id] = {
        "status": "starting",
        "created_at": datetime.now(),
        "user_code": None,
        "verification_uri": None
    }
    
    asyncio.create_task(run_az_login(session_id))
    
    # Wait for code/URI
    for _ in range(50):
        await asyncio.sleep(0.2)
        session = auth_sessions.get(session_id, {})
        if session.get("user_code") and session.get("verification_uri"):
            return AuthStartResponse(
                session_id=session_id,
                user_code=session["user_code"],
                verification_uri=session["verification_uri"]
            )
    
    raise HTTPException(status_code=500, detail="Failed to start authorization")

@app.get("/api/authorize/status", response_model=AuthStatusResponse)
async def check_authorization_status(session_id: str):
    """Check the status of an authorization session"""
    session = auth_sessions.get(session_id)
    
    if not session:
        return AuthStatusResponse(status="error", message="Session not found")
    
    status = session.get("status", "unknown")
    
    if status == "completed":
        return AuthStatusResponse(
            status="completed",
            authorized=session.get("authorized"),
            email=session.get("email"),
            user_name=session.get("user_name"),
            message=session.get("message")
        )
    elif status in ["starting", "waiting_for_user"]:
        return AuthStatusResponse(status="pending")
    elif status == "timeout":
        return AuthStatusResponse(status="timeout", message=session.get("message"))
    elif status == "error":
        return AuthStatusResponse(status="error", message=session.get("message"))
    else:
        return AuthStatusResponse(status="pending")

@app.post("/api/authorize/complete", response_model=TokenResponse)
async def complete_authorization(
    request: AuthCompleteRequest,
    response: Response
):
    """Complete authorization and issue JWT tokens"""
    session = auth_sessions.get(request.session_id)
    
    if not session or session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Invalid or incomplete session")
    
    if not session.get("authorized"):
        raise HTTPException(status_code=403, detail="User not authorized")
    
    email = session.get("email")
    user_name = session.get("user_name")
    roles = session.get("roles", ["user"])
    
    # Create or get user
    user = db.get_user_by_email(email)
    if not user:
        user_id = db.create_user(email, user_name, roles)
        user = db.get_user_by_id(user_id)
    else:
        # Update roles in case they changed in config
        db.update_user_roles(user["id"], roles)
        db.update_last_login(user["id"])
        user = db.get_user_by_id(user["id"])
    
    # Generate device ID
    device_id = str(uuid.uuid4())
    
    # Create tokens
    token_data = {
        "sub": email,
        "user_id": user["id"],
        "roles": roles,
        "device_id": device_id,
        "fingerprint": request.fingerprint
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    # Store tokens in database
    access_expires_at = (datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)).isoformat()
    refresh_expires_at = (datetime.utcnow() + timedelta(minutes=config.REFRESH_TOKEN_EXPIRE_MINUTES)).isoformat()
    
    db.create_device_token(
        user_id=user["id"],
        username=user["username"],
        device_id=device_id,
        fingerprint=request.fingerprint,
        access_token_hash=hash_token(access_token),
        refresh_token_hash=hash_token(refresh_token),
        access_expires_at=access_expires_at,
        refresh_expires_at=refresh_expires_at
    )
    
    # Set cookies
    set_auth_cookies(response, access_token, refresh_token)
    
    # Also set fingerprint cookie
    response.set_cookie(
        key="fingerprint",
        value=request.fingerprint,
        httponly=False,  # Needs to be readable by JS
        max_age=config.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax"
    )
    
    # Clean up session
    del auth_sessions[request.session_id]
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(**user)
    )

@app.post("/api/auth/refresh")
async def refresh_access_token(
    response: Response,
    refresh_token: Optional[str] = Cookie(None),
    fingerprint: Optional[str] = Cookie(None)
):
    """Refresh access token using refresh token"""
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token required")
    
    token_data = decode_token(refresh_token)
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    # Verify fingerprint
    if not verify_fingerprint(token_data.fingerprint, fingerprint):
        raise HTTPException(status_code=401, detail="Fingerprint mismatch")
    
    # Verify device token exists
    device_token = db.get_device_token(token_data.device_id)
    if not device_token:
        raise HTTPException(status_code=401, detail="Device token not found")
    
    # Create new access token
    new_token_data = {
        "sub": token_data.email,
        "user_id": token_data.user_id,
        "roles": token_data.roles,
        "device_id": token_data.device_id,
        "fingerprint": token_data.fingerprint
    }
    
    new_access_token = create_access_token(new_token_data)
    access_expires_at = (datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)).isoformat()
    
    # Update database
    db.update_device_token(
        device_id=token_data.device_id,
        access_token_hash=hash_token(new_access_token),
        access_expires_at=access_expires_at
    )
    
    # Set new access token cookie
    response.set_cookie(
        key="access_token",
        value=new_access_token,
        httponly=True,
        max_age=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax"
    )
    
    return {"message": "Token refreshed successfully"}

@app.post("/api/auth/logout")
async def logout(
    response: Response,
    access_token: Optional[str] = Cookie(None)
):
    """Logout and invalidate tokens"""
    if access_token:
        token_data = decode_token(access_token)
        if token_data and token_data.device_id:
            db.delete_device_token(token_data.device_id)
    
    clear_auth_cookies(response)
    response.delete_cookie("fingerprint")
    
    return {"message": "Logged out successfully"}

# ============= API ENDPOINTS =============

@app.get("/api/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information"""
    user = db.get_user_by_id(current_user.user_id)
    return UserResponse(**user)

@app.get("/api/check-auth")
async def check_auth(current_user = Depends(get_current_user)):
    """Check if user is authenticated"""
    return {
        "authenticated": True,
        "email": current_user.email,
        "roles": current_user.roles
    }

@app.get("/api/admin/sessions")
async def get_all_sessions(current_user = Depends(require_admin)):
    """Get all active sessions (Admin only)"""
    sessions = db.get_all_device_tokens()
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }

@app.get("/api/me/devices")
async def get_my_devices(current_user = Depends(get_current_user)):
    """Get all devices for current user"""
    devices = db.get_user_devices(current_user.user_id)
    return {
        "total_devices": len(devices),
        "devices": devices
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)