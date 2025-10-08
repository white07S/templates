import asyncio
import json
import re
import uuid
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional
import jwt
from jwt.exceptions import InvalidTokenError
from fastapi import HTTPException, status, Cookie, Response
import config
from models import TokenData

logger = logging.getLogger(__name__)

# In-memory storage for Azure authorization sessions
auth_sessions: Dict[str, Dict] = {}

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=config.REFRESH_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        roles: list = payload.get("roles", [])
        device_id: str = payload.get("device_id")
        fingerprint: str = payload.get("fingerprint")
        
        if email is None or user_id is None:
            return None
            
        return TokenData(
            email=email,
            user_id=user_id,
            roles=roles,
            device_id=device_id,
            fingerprint=fingerprint
        )
    except InvalidTokenError as e:
        logger.warning(f"Token decode error: {e}")
        return None

def hash_token(token: str) -> str:
    """Hash token for storage"""
    return hashlib.sha256(token.encode()).hexdigest()

def verify_fingerprint(stored_fingerprint: str, provided_fingerprint: str) -> bool:
    """Verify browser fingerprint matches"""
    return stored_fingerprint == provided_fingerprint

def set_auth_cookies(response: Response, access_token: str, refresh_token: str):
    """Set authentication cookies"""
    # Set access token cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=False  # Set to True in production with HTTPS
    )
    
    # Set refresh token cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        max_age=config.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=False
    )

def clear_auth_cookies(response: Response):
    """Clear authentication cookies"""
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")

async def run_az_login(session_id: str):
    """
    Runs `az login --use-device-code --output json` as a subprocess,
    parses the device code + URL, and processes the final auth result.
    """
    logger.debug(f"Starting Azure login process for session {session_id}")
    
    proc = await asyncio.create_subprocess_exec(
        "az", "login", "--use-device-code", "--output", "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    user_code = None
    verification_uri = None
    json_lines = []
    seen_first_json = False

    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="ignore").strip()
            logger.debug(f"Azure CLI output: {text}")

            if not seen_first_json:
                code_match = re.search(r"code\s+([A-Z0-9\-]+)", text)
                url_match = re.search(r"(https?://\S+)", text)
                if code_match:
                    user_code = code_match.group(1)
                if url_match:
                    verification_uri = url_match.group(1)
                if user_code and verification_uri:
                    auth_sessions[session_id].update({
                        "user_code": user_code,
                        "verification_uri": verification_uri,
                        "status": "waiting_for_user"
                    })
                    seen_first_json = True
                    continue

                if text.startswith("[") or text.startswith("{"):
                    seen_first_json = True

            if seen_first_json:
                json_lines.append(text)

        try:
            await asyncio.wait_for(proc.wait(), timeout=config.AZ_LOGIN_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            auth_sessions[session_id].update({
                "status": "timeout",
                "message": "Login timed out"
            })
            return

        if not json_lines:
            auth_sessions[session_id].update({
                "status": "error",
                "message": "No JSON from az login"
            })
            return

        raw_json = "\n".join(json_lines)
        
        try:
            accounts = json.loads(raw_json)
        except json.JSONDecodeError:
            auth_sessions[session_id].update({
                "status": "error",
                "message": "Malformed JSON from az login"
            })
            return

        if not isinstance(accounts, list) or not accounts:
            auth_sessions[session_id].update({
                "status": "error",
                "message": "Unexpected login response"
            })
            return

        email = accounts[0].get("user", {}).get("name")
        user_name = email.split('@')[0] if email else None
        
        if not email:
            auth_sessions[session_id].update({
                "status": "error",
                "message": "Email not found in response"
            })
            return

        if email in config.ALLOWED_EMAILS:
            roles = config.ROLE_MAPPING.get(email, ["user"])
            auth_sessions[session_id].update({
                "status": "completed",
                "authorized": True,
                "email": email,
                "user_name": user_name,
                "roles": roles,
                "message": f"Authorization successful, welcome {user_name}"
            })
            logger.info(f"Authorization successful for {email} with roles {roles}")
        else:
            auth_sessions[session_id].update({
                "status": "completed",
                "authorized": False,
                "email": email,
                "user_name": user_name,
                "message": "Unauthorized user"
            })

    finally:
        if proc.returncode is None:
            proc.kill()