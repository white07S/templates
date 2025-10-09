import asyncio
import json
import logging
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from fastapi import Response

import config

logger = logging.getLogger(__name__)

# In-memory storage for Azure authorization sessions
auth_sessions: Dict[str, Dict] = {}


class AzureLoginError(Exception):
    """Raised when the Azure CLI flow cannot be completed."""


def hash_token(token: str) -> str:
    """Return a SHA256 hash of the provided token."""
    import hashlib

    return hashlib.sha256(token.encode()).hexdigest()


def verify_fingerprint(stored_fingerprint: str, provided_fingerprint: str) -> bool:
    """Ensure the browser fingerprint matches the stored value."""
    return stored_fingerprint == provided_fingerprint


def set_auth_cookies(response: Response, access_token: str):
    """Set HttpOnly cookies for the Graph access token."""
    max_age = config.GRAPH_TOKEN_TTL_MINUTES * 60
    response.set_cookie(
        key="graph_access_token",
        value=access_token,
        httponly=True,
        max_age=max_age,
        samesite="lax",
        secure=False,  # Set to True when HTTPS is enforced
    )


def clear_auth_cookies(response: Response):
    """Remove authentication cookies from the response."""
    response.delete_cookie("graph_access_token")
    response.delete_cookie("fingerprint")


async def _run_az_command(command: list, env: Dict[str, str]) -> str:
    """Execute an Azure CLI command and return stdout."""
    logger.debug("Executing command: %s", " ".join(command))
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        error_output = stderr.decode() or stdout.decode()
        logger.error("Command failed: %s", error_output.strip())
        raise AzureLoginError(error_output.strip())
    return stdout.decode()


def _resolve_roles(group_ids: Optional[list]) -> Dict[str, list]:
    """Map Azure AD group IDs to application roles."""
    matched_roles = []
    group_ids = set(group_ids or [])

    for role, configured_groups in config.AZURE_ROLE_GROUP_MAPPING.items():
        for gid in configured_groups:
            if gid and gid in group_ids:
                matched_roles.append(role)
                break

    if not matched_roles and config.DEFAULT_ROLE:
        matched_roles.append(config.DEFAULT_ROLE)

    return {"roles": sorted(set(matched_roles))}


async def run_az_login(session_id: str, config_dir: Path):
    """
    Runs `az login --use-device-code --output json` in an isolated AZURE_CONFIG_DIR,
    parses the device code + URL, and stores the resulting authentication details.
    """
    logger.debug("Starting Azure login process for session %s", session_id)

    env = os.environ.copy()
    env["AZURE_CONFIG_DIR"] = str(config_dir)

    proc = await asyncio.create_subprocess_exec(
        "az",
        "login",
        "--use-device-code",
        "--output",
        "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
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
            logger.debug("Azure CLI output: %s", text)

            if not seen_first_json:
                code_match = re.search(r"code\s+([A-Z0-9\-]+)", text)
                url_match = re.search(r"(https?://\S+)", text)
                if code_match:
                    user_code = code_match.group(1)
                if url_match:
                    verification_uri = url_match.group(1)
                if user_code and verification_uri:
                    auth_sessions[session_id].update(
                        {
                            "user_code": user_code,
                            "verification_uri": verification_uri,
                            "status": "waiting_for_user",
                        }
                    )
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
            auth_sessions[session_id].update(
                {
                    "status": "timeout",
                    "message": "Login timed out",
                }
            )
            return

        if not json_lines:
            auth_sessions[session_id].update(
                {
                    "status": "error",
                    "message": "No JSON payload returned from az login",
                }
            )
            return

        raw_json = "\n".join(json_lines)

        try:
            accounts = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse az login output: %s", exc)
            auth_sessions[session_id].update(
                {
                    "status": "error",
                    "message": "Malformed JSON from az login",
                }
            )
            return

        if not isinstance(accounts, list) or not accounts:
            auth_sessions[session_id].update(
                {
                    "status": "error",
                    "message": "Unexpected login response",
                }
            )
            return

        # Gather additional user context
        try:
            user_info_raw = await _run_az_command(
                ["az", "ad", "signed-in-user", "show", "--output", "json"],
                env,
            )
            user_info = json.loads(user_info_raw)
            azure_object_id = user_info.get("id") or user_info.get("objectId")
            email = user_info.get("userPrincipalName") or user_info.get("mail")
            user_name = user_info.get("mailNickname") or (
                email.split("@")[0] if email else None
            )

            member_of_raw = await _run_az_command(
                [
                    "az",
                    "rest",
                    "--method",
                    "GET",
                    "--uri",
                    f"https://graph.microsoft.com/v1.0/users/{azure_object_id}/memberOf",
                    "--headers",
                    "ConsistencyLevel=eventual",
                ],
                env,
            )
            member_of = json.loads(member_of_raw)
            group_ids = [
                entry.get("id")
                for entry in member_of.get("value", [])
                if entry.get("@odata.type") == "#microsoft.graph.group"
            ]

            roles_info = _resolve_roles(group_ids)
            roles = roles_info["roles"]

            if not email or not roles:
                auth_sessions[session_id].update(
                    {
                        "status": "completed",
                        "authorized": False,
                        "email": email,
                        "message": "User not authorized for any roles",
                    }
                )
                return

            access_token_raw = await _run_az_command(
                [
                    "az",
                    "account",
                    "get-access-token",
                    "--resource",
                    config.GRAPH_RESOURCE,
                    "--output",
                    "json",
                ],
                env,
            )
            token_payload = json.loads(access_token_raw)
            graph_token = token_payload.get("accessToken")
            tenant_id = token_payload.get("tenant")
            graph_token_expires_on = token_payload.get("expiresOn")
            custom_expiry = (
                datetime.utcnow() + timedelta(minutes=config.GRAPH_TOKEN_TTL_MINUTES)
            ).isoformat()

            if not graph_token:
                auth_sessions[session_id].update(
                    {
                        "status": "error",
                        "message": "Failed to fetch Graph access token",
                    }
                )
                return

            auth_sessions[session_id].update(
                {
                    "status": "completed",
                    "authorized": True,
                    "email": email,
                    "user_name": user_name,
                    "roles": roles,
                    "azure_object_id": azure_object_id,
                    "azure_tenant_id": tenant_id,
                    "graph_token": graph_token,
                    "graph_token_expires_at": custom_expiry,
                    "graph_token_original_expires_on": graph_token_expires_on,
                    "message": f"Authorization successful, welcome {user_name}",
                    "group_ids": group_ids,
                }
            )
            logger.info(
                "Authorization successful for %s (roles=%s)",
                email,
                roles,
            )
        except AzureLoginError as exc:
            auth_sessions[session_id].update(
                {
                    "status": "error",
                    "message": str(exc),
                }
            )
        except json.JSONDecodeError as exc:
            auth_sessions[session_id].update(
                {
                    "status": "error",
                    "message": f"Unable to parse Azure CLI response: {exc}",
                }
            )
    finally:
        if proc.returncode is None:
            proc.kill()
        # Clean up the AZURE_CONFIG_DIR for this session
        shutil.rmtree(config_dir, ignore_errors=True)
