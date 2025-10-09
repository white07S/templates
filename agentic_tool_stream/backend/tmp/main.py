import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

import config
from auth import (
    auth_sessions,
    clear_auth_cookies,
    hash_token,
    run_az_login,
    set_auth_cookies,
)
from database import db
from models import (
    AuthCompleteRequest,
    AuthStartResponse,
    AuthStatusResponse,
    TokenResponse,
    UserResponse,
)
from rbac import get_current_user, require_admin, require_user
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
    """Start the Azure device code authorization process."""
    session_id = str(uuid.uuid4())
    logger.info("Starting authorization session %s", session_id)

    config.AZURE_CONFIG_BASE_DIR.mkdir(parents=True, exist_ok=True)
    session_config_dir = config.AZURE_CONFIG_BASE_DIR / session_id
    session_config_dir.mkdir(parents=True, exist_ok=True)

    auth_sessions[session_id] = {
        "status": "starting",
        "created_at": datetime.utcnow().isoformat(),
        "user_code": None,
        "verification_uri": None,
    }

    asyncio.create_task(run_az_login(session_id, session_config_dir))

    for _ in range(50):
        await asyncio.sleep(0.2)
        session = auth_sessions.get(session_id, {})
        if session.get("user_code") and session.get("verification_uri"):
            return AuthStartResponse(
                session_id=session_id,
                user_code=session["user_code"],
                verification_uri=session["verification_uri"],
            )

    raise HTTPException(status_code=500, detail="Failed to start authorization")


@app.get("/api/authorize/status", response_model=AuthStatusResponse)
async def check_authorization_status(session_id: str):
    """Check the status of an authorization session."""
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
            message=session.get("message"),
        )
    if status in {"starting", "waiting_for_user"}:
        return AuthStatusResponse(status="pending")
    if status == "timeout":
        return AuthStatusResponse(status="timeout", message=session.get("message"))
    if status == "error":
        return AuthStatusResponse(status="error", message=session.get("message"))

    return AuthStatusResponse(status="pending")


@app.post("/api/authorize/complete", response_model=TokenResponse)
async def complete_authorization(request: AuthCompleteRequest, response: Response):
    """Complete authorization and persist session tied to Graph token."""
    session = auth_sessions.get(request.session_id)

    if not session or session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Invalid or incomplete session")

    if not session.get("authorized"):
        raise HTTPException(status_code=403, detail="User not authorized")

    graph_token = session.get("graph_token")
    if not graph_token:
        raise HTTPException(
            status_code=400, detail="Graph token missing from session response"
        )

    fingerprint = request.fingerprint
    if not fingerprint:
        raise HTTPException(status_code=400, detail="Fingerprint required")

    email = session.get("email")
    user_name = session.get("user_name")
    roles = session.get("roles", [])
    azure_object_id = session.get("azure_object_id")
    azure_tenant_id = session.get("azure_tenant_id")
    token_expires_at = session.get("graph_token_expires_at")
    original_expires_on = session.get("graph_token_original_expires_on")

    if not email or not azure_object_id:
        raise HTTPException(
            status_code=400, detail="Azure user identity information incomplete"
        )

    user_record = db.create_or_update_user(
        azure_object_id=azure_object_id,
        email=email,
        username=user_name or email.split("@")[0],
        roles=roles,
    )

    device_id = str(uuid.uuid4())
    token_hash = hash_token(graph_token)

    db.create_session(
        user_id=user_record["id"],
        username=user_record["username"],
        device_id=device_id,
        fingerprint=fingerprint,
        token_hash=token_hash,
        token_expires_at=token_expires_at,
        azure_object_id=azure_object_id,
        azure_tenant_id=azure_tenant_id,
        azure_token_expires_on=original_expires_on,
    )

    set_auth_cookies(response, graph_token)

    response.set_cookie(
        key="fingerprint",
        value=fingerprint,
        httponly=False,
        max_age=config.GRAPH_TOKEN_TTL_MINUTES * 60,
        samesite="lax",
    )

    auth_sessions.pop(request.session_id, None)

    return TokenResponse(
        token_expires_at=token_expires_at,
        user=UserResponse(**user_record),
        roles=user_record["roles"],
    )


@app.post("/api/auth/logout")
async def logout(response: Response, graph_access_token: Optional[str] = Cookie(None)):
    """Logout the current user and invalidate stored session."""
    if graph_access_token:
        token_hash = hash_token(graph_access_token)
        db.delete_session_by_token_hash(token_hash)

    clear_auth_cookies(response)
    response.delete_cookie("fingerprint")

    return {"message": "Logged out successfully"}


# ============= API ENDPOINTS =============


@app.get("/api/me", response_model=UserResponse)
async def get_current_user_info(current_user=Depends(get_current_user)):
    """Get current user information."""
    user = db.get_user_by_id(current_user.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**user)


@app.get("/api/check-auth")
async def check_auth(current_user=Depends(get_current_user)):
    """Check if user is authenticated."""
    return {
        "authenticated": True,
        "email": current_user.email,
        "roles": current_user.roles,
        "token_expires_at": current_user.token_expires_at,
    }


@app.get("/api/admin/sessions")
async def get_all_sessions(current_user=Depends(require_admin)):
    """Get all active sessions (Admin only)."""
    sessions = db.list_sessions()
    return {
        "total_sessions": len(sessions),
        "sessions": sessions,
    }


@app.get("/api/me/devices")
async def get_my_devices(current_user=Depends(get_current_user)):
    """Get all devices for current user."""
    devices = db.get_user_devices(current_user.user_id)
    return {
        "total_devices": len(devices),
        "devices": devices,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
