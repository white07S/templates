"""
Azure Authentication Service with device code flow
"""
import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

AZURE_ROLE_GROUP_MAPPING: Dict[str, List[str]] = {
    "admin": [gid.strip() for gid in os.getenv("AZURE_ADMIN_GROUP_IDS", "").split(",") if gid.strip()],
    "user": [gid.strip() for gid in os.getenv("AZURE_USER_GROUP_IDS", "").split(",") if gid.strip()],
}

class AzureAuthService:
    def __init__(self, settings, session_manager, token_manager):
        self.settings = settings
        self.session_manager = session_manager
        self.token_manager = token_manager
        self.active_auth_processes = {}

    @staticmethod
    def _normalize_timestamp(timestamp: Optional[str]) -> Optional[str]:
        """Convert various Azure timestamp formats to ISO 8601."""
        if not timestamp:
            return None

        formats = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt).isoformat()
            except ValueError:
                continue

        return timestamp

    async def start_device_code_auth(self, client_id: Optional[str] = None) -> dict:
        """Start the device code authentication flow"""
        session_id = str(uuid.uuid4())
        session_dir = self.session_manager.create_session_dir(session_id)

        # Create environment for the subprocess
        env = os.environ.copy()
        env["AZURE_CONFIG_DIR"] = str(session_dir)

        # Ensure Azure CLI can write to the session directory
        # Set HOME to session dir as fallback for Azure CLI
        env["HOME"] = str(session_dir)

        logger.info(f"Starting auth for session {session_id} in directory {session_dir}")
        logger.info(f"AZURE_CONFIG_DIR set to: {env['AZURE_CONFIG_DIR']}")

        try:
            # Start the az login process with device code
            proc = await asyncio.create_subprocess_exec(
                "az",
                "login",
                "--use-device-code",
                "--output",
                "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Store the process reference
            self.active_auth_processes[session_id] = {
                "process": proc,
                "env": env,
                "started_at": datetime.utcnow(),
                "session_dir": session_dir,
                "status": "pending"
            }

            # Parse device code information from stderr
            device_code_info = await self._parse_device_code_output(proc)

            if not device_code_info:
                raise Exception("Failed to get device code information")

            # Create initial session data
            session_data = {
                "session_id": session_id,
                "status": "pending",
                "device_code": device_code_info["user_code"],
                "verification_uri": device_code_info["verification_uri"],
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(minutes=15)).isoformat()
            }

            self.session_manager.save_session(session_id, session_data)

            # Start monitoring the auth process
            asyncio.create_task(self._monitor_auth_process(session_id))

            return {
                "session_id": session_id,
                "user_code": device_code_info["user_code"],
                "verification_uri": device_code_info["verification_uri"],
                "expires_at": datetime.utcnow() + timedelta(minutes=15)
            }

        except Exception as e:
            logger.error(f"Failed to start authentication: {e}")
            # Clean up on failure
            if session_id in self.active_auth_processes:
                del self.active_auth_processes[session_id]
            self.session_manager.delete_session(session_id)
            raise

    async def _parse_device_code_output(self, proc) -> Optional[dict]:
        """Parse the device code information from az login output"""
        try:
            # Read initial output to get device code
            stderr_output = ""
            user_code = None
            verification_uri = None

            # Read stderr line by line to get device code info
            while True:
                try:
                    line = await asyncio.wait_for(
                        proc.stderr.readline(),
                        timeout=5.0
                    )
                    if not line:
                        break

                    line_str = line.decode('utf-8')
                    stderr_output += line_str

                    # Look for device code pattern
                    if "enter the code" in line_str.lower():
                        code_match = re.search(r'([A-Z0-9]{9})', line_str)
                        if code_match:
                            user_code = code_match.group(1)

                    # Look for verification URL
                    if "https://microsoft.com/devicelogin" in line_str:
                        verification_uri = "https://microsoft.com/devicelogin"

                    # If we have both, we can return
                    if user_code and verification_uri:
                        return {
                            "user_code": user_code,
                            "verification_uri": verification_uri
                        }

                except asyncio.TimeoutError:
                    # If we have the info, return it
                    if user_code and verification_uri:
                        return {
                            "user_code": user_code,
                            "verification_uri": verification_uri
                        }
                    continue

            # Fallback: try to extract from full output
            if "enter the code" in stderr_output.lower():
                code_match = re.search(r'([A-Z0-9]{9})', stderr_output)
                if code_match:
                    user_code = code_match.group(1)

            return {
                "user_code": user_code or "UNKNOWN",
                "verification_uri": verification_uri or "https://microsoft.com/devicelogin"
            }

        except Exception as e:
            logger.error(f"Failed to parse device code output: {e}")
            return None

    async def _monitor_auth_process(self, session_id: str):
        """Monitor the authentication process"""
        try:
            auth_info = self.active_auth_processes.get(session_id)
            if not auth_info:
                return

            proc = auth_info["process"]
            env = auth_info["env"]

            # Wait for the process to complete
            stdout, stderr = await proc.communicate()

            # Check if authentication was successful
            if proc.returncode == 0:
                # Authentication successful, get user info
                try:
                    # Parse the login output
                    login_data = json.loads(stdout.decode('utf-8'))

                    # Get access token and user information
                    user_info = await self._get_user_info(env)

                    # Update session with user info
                    session_data = self.session_manager.get_session(session_id)
                    normalized_expiry = self._normalize_timestamp(user_info.get("token_expires_on"))
                    if normalized_expiry:
                        user_info["token_expires_on"] = normalized_expiry
                    session_data.update({
                        "status": "completed",
                        "authenticated": True,
                        "user_info": user_info,
                        "login_data": login_data[0] if login_data else {},
                        "token_expires_at": normalized_expiry
                    })

                    self.session_manager.save_session(session_id, session_data)
                    auth_info["status"] = "completed"

                except Exception as e:
                    logger.error(f"Failed to get user info: {e}")
                    auth_info["status"] = "error"
                    session_data = self.session_manager.get_session(session_id)
                    session_data["status"] = "error"
                    session_data["error"] = str(e)
                    self.session_manager.save_session(session_id, session_data)

            else:
                # Authentication failed
                auth_info["status"] = "error"
                session_data = self.session_manager.get_session(session_id)
                session_data["status"] = "error"
                session_data["error"] = stderr.decode('utf-8')
                self.session_manager.save_session(session_id, session_data)

        except Exception as e:
            logger.error(f"Error monitoring auth process: {e}")
            if session_id in self.active_auth_processes:
                self.active_auth_processes[session_id]["status"] = "error"

    async def _get_user_info(self, env: dict) -> dict:
        """Get user information after successful authentication"""
        try:
            # Get access token with tenant ID
            token_cmd = ["az", "account", "get-access-token"]
            if self.settings.TENANT_ID:
                token_cmd.extend(["--tenant", self.settings.TENANT_ID])

            token_output = await self._run_az_command(token_cmd, env)
            token_data = json.loads(token_output)
            expires_iso = self._normalize_timestamp(token_data.get("expiresOn"))
            if expires_iso:
                token_data["expiresOn"] = expires_iso

            # Get account information
            account_output = await self._run_az_command(
                ["az", "account", "show", "--output", "json"],
                env
            )
            account_data = json.loads(account_output)

            # Get signed-in Azure AD user details
            user_info_raw = await self._run_az_command(
                ["az", "ad", "signed-in-user", "show", "--output", "json"],
                env,
            )
            user_info = json.loads(user_info_raw)

            azure_object_id = user_info.get("id") or user_info.get("objectId")
            email = user_info.get("userPrincipalName") or user_info.get("mail") or ""
            user_name = (
                user_info.get("mailNickname")
                or (email.split("@")[0] if email else None)
                or account_data.get("user", {}).get("name")
            )
            user_type = user_info.get("userType") or account_data.get("user", {}).get("type", "")

            # Retrieve group memberships for role resolution
            group_ids: List[str] = []
            profile_data: dict = {}
            groups_result: Optional[dict] = None
            if azure_object_id:
                member_of_raw = await self._run_az_command(
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

                groups_result = await self._get_user_groups(
                    azure_object_id,
                    env
                )
                if groups_result:
                    profile_data = groups_result.get("profile", {})
                    if not group_ids:
                        group_ids = groups_result.get("group_ids", [])
            else:
                logger.warning("Azure object ID not found for signed-in user")

            roles_info = self._resolve_roles(group_ids)

            # For work/school accounts, get additional info from Graph API
            user_info = {
                "email": email,
                "user_name": user_name or "unknown",
                "user_type": user_type,
                "tenant_id": account_data.get("tenantId"),
                "subscription_id": account_data.get("id"),
                "token_expires_on": expires_iso,
                "access_token": token_data.get("accessToken"),
                "object_id": azure_object_id,
                "group_ids": group_ids,
                "roles": roles_info.get("roles", []),
                "role_details": roles_info.get("matched_groups", []),
                "profile": profile_data,
            }

            # Get user's Azure AD object ID and groups if it's a work account
            if user_type == "user" and not user_info.get("object_id"):
                try:
                    # Get user's object ID from token claims
                    import base64
                    token_parts = token_data.get("accessToken", "").split(".")
                    if len(token_parts) >= 2:
                        # Decode the token payload
                        payload = base64.urlsafe_b64decode(
                            token_parts[1] + "=" * (4 - len(token_parts[1]) % 4)
                        )
                        claims = json.loads(payload)
                        user_info["object_id"] = claims.get("oid") or claims.get("sub")

                except Exception as e:
                    logger.warning(f"Could not get additional user info: {e}")

            # If we still don't have group IDs, attempt legacy retrieval
            if not group_ids and user_info.get("object_id"):
                try:
                    if not groups_result:
                        groups_result = await self._get_user_groups(
                            user_info["object_id"],
                            env
                        )
                    fallback_groups = groups_result.get("group_ids", []) if groups_result else []
                    user_info["group_ids"] = fallback_groups
                    legacy_roles = self._resolve_roles(fallback_groups)
                    user_info["roles"] = legacy_roles.get("roles", user_info.get("roles", []))
                    user_info["role_details"] = legacy_roles.get("matched_groups", user_info.get("role_details", []))
                    if not user_info.get("profile") and groups_result:
                        user_info["profile"] = groups_result.get("profile", {})
                except Exception as e:
                    logger.warning(f"Could not retrieve group memberships via legacy path: {e}")

            return user_info

        except Exception as e:
            logger.error(f"Failed to get user information: {e}")
            raise

    async def _get_user_groups(self, object_id: str, env: dict) -> dict:
        """Get user's group memberships and profile information from Microsoft Graph"""
        group_ids: List[str] = []
        profile_data: dict = {}

        try:
            # Use az rest to call Microsoft Graph API for group memberships
            member_of_output = await self._run_az_command([
                "az", "rest",
                "--method", "GET",
                "--uri", f"https://graph.microsoft.com/v1.0/users/{object_id}/memberOf",
                "--headers", "ConsistencyLevel=eventual"
            ], env)

            member_of_data = json.loads(member_of_output)

            group_ids = [
                entry.get("id")
                for entry in member_of_data.get("value", [])
                if entry.get("@odata.type") == "#microsoft.graph.group"
            ]

        except Exception as e:
            logger.warning(f"Could not get user groups: {e}")

        try:
            # Retrieve additional profile information from Microsoft Graph beta endpoint
            profile_output = await self._run_az_command([
                "az", "rest",
                "--method", "GET",
                "--uri", "https://graph.microsoft.com/beta/me/profile"
            ], env)
            profile_data = json.loads(profile_output)

        except Exception as e:
            logger.warning(f"Could not get user profile: {e}")

        return {
            "group_ids": group_ids,
            "profile": profile_data
        }

    def _resolve_roles(self, group_ids: list) -> dict:
        """Resolve group IDs to role names"""
        resolved_roles = set()
        matched_groups = []
        allowed_roles = set(self.settings.allowed_roles_list)
        role_group_mapping = {role: list(groups) for role, groups in self.settings.role_group_mapping.items()}

        # Merge environment constant mapping without duplicating entries
        for role_name, groups in AZURE_ROLE_GROUP_MAPPING.items():
            if role_name not in role_group_mapping:
                role_group_mapping[role_name] = list(groups)
            else:
                combined = set(role_group_mapping[role_name])
                combined.update(groups)
                role_group_mapping[role_name] = list(combined)
        legacy_role_mapping = self.settings.role_mapping_dict

        for group_id in group_ids:
            matched = False

            # Check explicit role group mapping (role -> [group ids])
            for role_name, group_list in role_group_mapping.items():
                if group_id in group_list:
                    matched = True
                    resolved_roles.add(role_name)
                    matched_groups.append({"role": role_name, "group_id": group_id})

            # Check legacy mapping (group id -> role)
            if group_id in legacy_role_mapping:
                matched = True
                legacy_role = legacy_role_mapping[group_id]
                resolved_roles.add(legacy_role)
                matched_groups.append({"role": legacy_role, "group_id": group_id})

            if not matched:
                matched_groups.append({"role": None, "group_id": group_id})

        if allowed_roles:
            filtered_roles = [
                role for role in resolved_roles if role in allowed_roles
            ]
        else:
            filtered_roles = list(resolved_roles)

        # Default role if no mapped roles found and allowed roles specify user
        if not filtered_roles and (not allowed_roles or "user" in allowed_roles):
            filtered_roles.append("user")

        return {
            "roles": filtered_roles,
            "matched_groups": matched_groups,
            "all_matched_roles": list(resolved_roles),
        }

    async def _run_az_command(self, cmd: list, env: dict) -> str:
        """Run an az CLI command and return output"""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise Exception(f"Command failed: {stderr.decode('utf-8')}")

        return stdout.decode('utf-8')

    async def check_auth_status(self, session_id: str) -> dict:
        """Check the status of an authentication process"""
        session_data = self.session_manager.get_session(session_id)

        if not session_data:
            return {
                "status": "error",
                "authorized": False,
                "message": "Session not found"
            }

        auth_info = self.active_auth_processes.get(session_id, {})

        # Check for timeout
        if auth_info and "started_at" in auth_info:
            elapsed = (datetime.utcnow() - auth_info["started_at"]).total_seconds()
            if elapsed > self.settings.AUTH_TIMEOUT_SECONDS:
                # Kill the process
                if "process" in auth_info and auth_info["process"].returncode is None:
                    auth_info["process"].terminate()

                return {
                    "status": "timeout",
                    "authorized": False,
                    "message": "Authentication timeout"
                }

        status = session_data.get("status", "pending")

        if status == "completed":
            user_info = session_data.get("user_info", {})
            roles = user_info.get("roles", [])

            # Check if user is authorized (has valid roles)
            authorized = bool(roles) or not self.settings.allowed_roles_list

            return {
                "status": "completed",
                "authorized": authorized,
                "user_info": {
                    "email": user_info.get("email"),
                    "user_name": user_info.get("user_name"),
                    "roles": roles
                }
            }

        elif status == "error":
            return {
                "status": "error",
                "authorized": False,
                "message": session_data.get("error", "Authentication failed")
            }

        else:
            return {
                "status": "pending",
                "authorized": False,
                "message": "Authentication in progress"
            }

    async def complete_auth(self, session_id: str) -> dict:
        """Complete the authentication and return session info"""
        session_data = self.session_manager.get_session(session_id)

        if not session_data:
            raise Exception("Session not found")

        if session_data.get("status") != "completed":
            raise Exception("Authentication not completed")

        user_info = session_data.get("user_info", {})
        access_token = user_info.get("access_token")
        if not access_token:
            raise Exception("Access token not available for session")

        # Store token information
        logger.info(f"Storing token for session {session_id}, token length: {len(access_token) if access_token else 0}")
        await self.token_manager.store_token(
            session_id=session_id,
            token=access_token,
            expires_at=user_info.get("token_expires_on"),
            refresh_token=None  # Azure CLI handles refresh internally
        )
        logger.info(f"Token stored successfully for session {session_id}")

        # Clean up auth process reference
        if session_id in self.active_auth_processes:
            del self.active_auth_processes[session_id]

        expires_at_raw = user_info.get("token_expires_on")
        token_expires_at = None
        if expires_at_raw:
            try:
                token_expires_at = datetime.fromisoformat(expires_at_raw.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"Could not parse token expiry '{expires_at_raw}' for session {session_id}")
                token_expires_at = datetime.utcnow()
        else:
            token_expires_at = datetime.utcnow()

        normalized_expiry = token_expires_at.isoformat()
        user_record = session_data.setdefault("user_info", {})
        user_record["token_expires_on"] = normalized_expiry
        user_record["access_token"] = access_token
        session_data["token_expires_at"] = normalized_expiry
        self.session_manager.save_session(session_id, session_data)

        return {
            "session_id": session_id,
            "user_info": {
                "email": user_info.get("email"),
                "user_name": user_info.get("user_name"),
                "tenant_id": user_info.get("tenant_id")
            },
            "roles": user_info.get("roles", []),
            "token_expires_at": token_expires_at,
            "access_token": access_token
        }

    async def refresh_session_token(self, session_id: str) -> dict:
        """Refresh the token for a session"""
        session_data = self.session_manager.get_session(session_id)

        if not session_data:
            raise Exception("Session not found")

        session_dir = Path(self.settings.SESSION_DIR) / session_id
        env = os.environ.copy()
        env["AZURE_CONFIG_DIR"] = str(session_dir)

        try:
            # Get a new access token
            token_cmd = ["az", "account", "get-access-token"]
            if self.settings.TENANT_ID:
                token_cmd.extend(["--tenant", self.settings.TENANT_ID])

            token_output = await self._run_az_command(token_cmd, env)
            token_data = json.loads(token_output)
            normalized_expires = self._normalize_timestamp(token_data.get("expiresOn"))
            if normalized_expires:
                token_data["expiresOn"] = normalized_expires

            new_access_token = token_data.get("accessToken")
            if not new_access_token:
                raise Exception("Failed to retrieve refreshed access token")

            # Update token information
            await self.token_manager.store_token(
                session_id=session_id,
                token=new_access_token,
                expires_at=token_data.get("expiresOn"),
                refresh_token=None
            )

            # Update session data
            user_record = session_data.setdefault("user_info", {})
            user_record["access_token"] = new_access_token
            user_record["token_expires_on"] = token_data.get("expiresOn")
            session_data["token_expires_at"] = token_data.get("expiresOn")
            self.session_manager.save_session(session_id, session_data)

            return {
                "token_refreshed": True,
                "expires_at": token_data.get("expiresOn"),
                "access_token": new_access_token
            }

        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            raise

    async def logout(self, session_id: str):
        """Logout and clean up session"""
        try:
            # Clean up auth process if still active
            if session_id in self.active_auth_processes:
                auth_info = self.active_auth_processes[session_id]
                if "process" in auth_info and auth_info["process"].returncode is None:
                    auth_info["process"].terminate()
                del self.active_auth_processes[session_id]

            # Remove token
            await self.token_manager.remove_token(session_id)

            # Delete session
            self.session_manager.delete_session(session_id)

        except Exception as e:
            logger.error(f"Error during logout: {e}")


"""
Azure OpenAI Service integration with user-specific credentials
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
import asyncio

from azure.identity import AzureCliCredential
from openai import AsyncAzureOpenAI
import openai

logger = logging.getLogger(__name__)

class AzureOpenAIService:
    def __init__(self, settings, token_manager):
        self.settings = settings
        self.token_manager = token_manager
        self._clients: Dict[str, AsyncAzureOpenAI] = {}

    async def get_client(self, session_id: str) -> AsyncAzureOpenAI:
        """Get or create an Azure OpenAI client for a session"""
        # Check if client exists and is valid
        if session_id in self._clients:
            return self._clients[session_id]

        # Create new client with user-specific credentials
        client = await self._create_client(session_id)
        self._clients[session_id] = client
        return client

    async def _create_client(self, session_id: str) -> AsyncAzureOpenAI:
        """Create a new Azure OpenAI client for a session"""
        try:
            # Set Azure config directory to session-specific folder
            session_dir = Path(self.settings.SESSION_DIR) / session_id

            session_file = session_dir / "session.json"
            if session_file.exists():
                try:
                    with open(session_file, "r") as f:
                        session_data = json.load(f)
                    logger.info(f"Loaded session data for {session_id}: {session_data}")
                except Exception as read_error:
                    logger.warning(f"Could not read session.json for {session_id}: {read_error}")

            # IMPORTANT: Set the environment variable for the current process
            # so that AzureCliCredential can find the session-specific Azure CLI config
            old_azure_config = os.environ.get("AZURE_CONFIG_DIR")
            os.environ["AZURE_CONFIG_DIR"] = str(session_dir)

            try:
                # Create credential using Azure CLI with tenant ID
                # It will now use the session-specific Azure config directory
                cred = AzureCliCredential(
                    tenant_id=self.settings.TENANT_ID,
                    process_timeout=30
                )
            finally:
                # Restore original environment variable
                if old_azure_config:
                    os.environ["AZURE_CONFIG_DIR"] = old_azure_config
                elif "AZURE_CONFIG_DIR" in os.environ:
                    del os.environ["AZURE_CONFIG_DIR"]

            # Token provider function that uses the session's credentials
            async def token_provider():
                try:
                    # Ensure Azure config directory is set for token retrieval
                    old_config = os.environ.get("AZURE_CONFIG_DIR")
                    os.environ["AZURE_CONFIG_DIR"] = str(session_dir)

                    try:
                        # Get token from Azure CLI credential
                        token = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: cred.get_token("https://cognitiveservices.azure.com/.default")
                        )
                        return token.token
                    finally:
                        # Restore original environment
                        if old_config:
                            os.environ["AZURE_CONFIG_DIR"] = old_config
                        elif "AZURE_CONFIG_DIR" in os.environ:
                            del os.environ["AZURE_CONFIG_DIR"]
                except Exception as e:
                    logger.error(f"Failed to get token for OpenAI: {e}")
                    # Fallback to stored token
                    stored_token = await self.token_manager.get_token(session_id)
                    if stored_token:
                        return stored_token
                    raise

            # Create the client with async support
            client = AsyncAzureOpenAI(
                azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT,
                azure_ad_token_provider=token_provider,
                api_version=self.settings.AZURE_OPENAI_API_VERSION,
                default_headers={
                    "User-Agent": f"AzureAuthChat/1.0 Session/{session_id}"
                }
            )

            logger.info(f"Created Azure OpenAI client for session {session_id}")
            return client

        except Exception as e:
            logger.error(f"Failed to create OpenAI client for session {session_id}: {e}")
            raise

    async def get_chat_response(
        self,
        session_id: str,
        message: str,
        context: Optional[List[dict]] = None
    ) -> dict:
        """Get a chat response from Azure OpenAI"""
        try:
            client = await self.get_client(session_id)

            # Prepare messages
            messages = []

            # Add system message
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant. Provide clear, concise, and accurate responses."
            })

            # Add context messages if provided
            if context:
                for ctx_msg in context:
                    messages.append({
                        "role": ctx_msg.get("role", "user"),
                        "content": ctx_msg.get("content", "")
                    })

            # Add user message
            messages.append({
                "role": "user",
                "content": message
            })

            # Make the API call
            response = await client.chat.completions.create(
                model=self.settings.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )

            # Extract response
            if response.choices and len(response.choices) > 0:
                assistant_message = response.choices[0].message.content
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None

                return {
                    "message": assistant_message,
                    "timestamp": datetime.utcnow(),
                    "usage": usage,
                    "model": self.settings.AZURE_OPENAI_DEPLOYMENT
                }
            else:
                raise Exception("No response from OpenAI")

        except openai.AuthenticationError as e:
            logger.error(f"Authentication error for session {session_id}: {e}")
            # Try to refresh token
            if session_id in self._clients:
                del self._clients[session_id]
            raise Exception("Authentication failed. Please re-authenticate.")

        except openai.RateLimitError as e:
            logger.error(f"Rate limit error for session {session_id}: {e}")
            raise Exception("Rate limit exceeded. Please try again later.")

        except Exception as e:
            logger.error(f"Failed to get chat response for session {session_id}: {e}")
            raise

    async def stream_chat_response(
        self,
        session_id: str,
        message: str,
        context: Optional[List[dict]] = None
    ):
        """Stream a chat response from Azure OpenAI"""
        try:
            client = await self.get_client(session_id)

            # Prepare messages (same as get_chat_response)
            messages = []
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant. Provide clear, concise, and accurate responses."
            })

            if context:
                for ctx_msg in context:
                    messages.append({
                        "role": ctx_msg.get("role", "user"),
                        "content": ctx_msg.get("content", "")
                    })

            messages.append({
                "role": "user",
                "content": message
            })

            # Stream the response
            stream = await client.chat.completions.create(
                model=self.settings.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content

        except Exception as e:
            logger.error(f"Failed to stream chat response for session {session_id}: {e}")
            raise

    async def get_embeddings(
        self,
        session_id: str,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> List[float]:
        """Get embeddings for text using Azure OpenAI"""
        try:
            client = await self.get_client(session_id)

            response = await client.embeddings.create(
                input=text,
                model=model
            )

            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                raise Exception("No embeddings returned")

        except Exception as e:
            logger.error(f"Failed to get embeddings for session {session_id}: {e}")
            raise

    async def cleanup_client(self, session_id: str):
        """Clean up client for a session"""
        if session_id in self._clients:
            del self._clients[session_id]
            logger.info(f"Cleaned up OpenAI client for session {session_id}")

    async def validate_configuration(self) -> bool:
        """Validate Azure OpenAI configuration"""
        try:
            if not self.settings.AZURE_OPENAI_ENDPOINT:
                logger.error("Azure OpenAI endpoint not configured")
                return False

            if not self.settings.AZURE_OPENAI_DEPLOYMENT:
                logger.error("Azure OpenAI deployment not configured")
                return False

            if not self.settings.TENANT_ID:
                logger.error("Azure tenant ID not configured")
                return False

            logger.info("Azure OpenAI configuration validated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to validate Azure OpenAI configuration: {e}")
            return False

    def get_client_status(self, session_id: str) -> dict:
        """Get status of OpenAI client for a session"""
        return {
            "has_client": session_id in self._clients,
            "endpoint": self.settings.AZURE_OPENAI_ENDPOINT,
            "deployment": self.settings.AZURE_OPENAI_DEPLOYMENT
        }

            raise
