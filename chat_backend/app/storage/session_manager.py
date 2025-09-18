import os
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from uuid import UUID, uuid4
from pathlib import Path
import asyncio
import aiofiles
from tinydb import TinyDB, Query

from ..models.schemas import ChatMessage, SessionInfo, MessageRole


class SessionManager:
    def __init__(self, base_path: str = "./data/users"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._session_cache: Dict[str, SessionInfo] = {}

    def _get_user_dir(self, user_id: str) -> Path:
        """Get user directory path"""
        user_dir = self.base_path / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _get_session_file(self, user_id: str, session_id: UUID) -> Path:
        """Get session file path"""
        user_dir = self._get_user_dir(user_id)
        return user_dir / f"{session_id}.json"

    def _get_user_metadata_file(self, user_id: str) -> Path:
        """Get user metadata file path"""
        user_dir = self._get_user_dir(user_id)
        return user_dir / "metadata.json"

    async def create_session(self, user_id: str, session_id: Optional[UUID] = None) -> UUID:
        """Create a new chat session"""
        if session_id is None:
            session_id = uuid4()

        session_file = self._get_session_file(user_id, session_id)

        # Initialize session data
        session_data = {
            "session_id": str(session_id),
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "messages": [],
            "metadata": {}
        }

        # Write session file
        async with aiofiles.open(session_file, 'w') as f:
            await f.write(json.dumps(session_data, indent=2))

        # Update session cache
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            message_count=0,
            memory_count=0
        )
        self._session_cache[f"{user_id}_{session_id}"] = session_info

        # Update user metadata
        await self._update_user_metadata(user_id, session_id)

        return session_id

    async def add_message(
        self,
        user_id: str,
        session_id: UUID,
        message: ChatMessage
    ) -> bool:
        """Add a message to a session"""
        session_file = self._get_session_file(user_id, session_id)

        if not session_file.exists():
            await self.create_session(user_id, session_id)

        try:
            # Read current session data
            async with aiofiles.open(session_file, 'r') as f:
                content = await f.read()
                session_data = json.loads(content)

            # Add new message
            message_dict = {
                "role": message.role.value,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata
            }

            session_data["messages"].append(message_dict)
            session_data["last_activity"] = datetime.now(timezone.utc).isoformat()

            # Write updated data
            async with aiofiles.open(session_file, 'w') as f:
                await f.write(json.dumps(session_data, indent=2))

            # Update cache
            cache_key = f"{user_id}_{session_id}"
            if cache_key in self._session_cache:
                self._session_cache[cache_key].last_activity = datetime.now(timezone.utc)
                self._session_cache[cache_key].message_count = len(session_data["messages"])

            return True

        except Exception as e:
            print(f"Error adding message: {e}")
            return False

    async def get_session_messages(
        self,
        user_id: str,
        session_id: UUID,
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get messages from a session"""
        session_file = self._get_session_file(user_id, session_id)

        if not session_file.exists():
            return []

        try:
            async with aiofiles.open(session_file, 'r') as f:
                content = await f.read()
                session_data = json.loads(content)

            messages = []
            message_list = session_data.get("messages", [])

            # Apply limit if specified
            if limit:
                message_list = message_list[-limit:]

            for msg_data in message_list:
                message = ChatMessage(
                    role=MessageRole(msg_data["role"]),
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    metadata=msg_data.get("metadata")
                )
                messages.append(message)

            return messages

        except Exception as e:
            print(f"Error reading session messages: {e}")
            return []

    async def get_session_info(self, user_id: str, session_id: UUID) -> Optional[SessionInfo]:
        """Get session information"""
        cache_key = f"{user_id}_{session_id}"

        # Check cache first
        if cache_key in self._session_cache:
            return self._session_cache[cache_key]

        # Load from file
        session_file = self._get_session_file(user_id, session_id)
        if not session_file.exists():
            return None

        try:
            async with aiofiles.open(session_file, 'r') as f:
                content = await f.read()
                session_data = json.loads(content)

            session_info = SessionInfo(
                session_id=UUID(session_data["session_id"]),
                user_id=session_data["user_id"],
                created_at=datetime.fromisoformat(session_data["created_at"]),
                last_activity=datetime.fromisoformat(session_data["last_activity"]),
                message_count=len(session_data.get("messages", [])),
                memory_count=0  # Will be updated by memory manager
            )

            # Cache the result
            self._session_cache[cache_key] = session_info
            return session_info

        except Exception as e:
            print(f"Error reading session info: {e}")
            return None

    async def list_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """List all sessions for a user"""
        user_dir = self._get_user_dir(user_id)
        sessions = []

        try:
            # Get all JSON files except metadata
            session_files = [f for f in user_dir.glob("*.json") if f.name != "metadata.json"]

            for session_file in session_files:
                try:
                    session_id = UUID(session_file.stem)
                    session_info = await self.get_session_info(user_id, session_id)
                    if session_info:
                        sessions.append(session_info)
                except ValueError:
                    # Skip invalid UUID filenames
                    continue

            # Sort by last activity
            sessions.sort(key=lambda x: x.last_activity, reverse=True)
            return sessions

        except Exception as e:
            print(f"Error listing user sessions: {e}")
            return []

    async def delete_session(self, user_id: str, session_id: UUID) -> bool:
        """Delete a session"""
        session_file = self._get_session_file(user_id, session_id)

        try:
            if session_file.exists():
                session_file.unlink()

            # Remove from cache
            cache_key = f"{user_id}_{session_id}"
            if cache_key in self._session_cache:
                del self._session_cache[cache_key]

            return True

        except Exception as e:
            print(f"Error deleting session: {e}")
            return False

    async def _update_user_metadata(self, user_id: str, session_id: UUID):
        """Update user metadata with session info"""
        metadata_file = self._get_user_metadata_file(user_id)

        try:
            # Read existing metadata or create new
            if metadata_file.exists():
                async with aiofiles.open(metadata_file, 'r') as f:
                    content = await f.read()
                    metadata = json.loads(content)
            else:
                metadata = {
                    "user_id": user_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "sessions": [],
                    "total_messages": 0
                }

            # Add session if not exists
            session_str = str(session_id)
            if session_str not in metadata["sessions"]:
                metadata["sessions"].append(session_str)

            metadata["last_activity"] = datetime.now(timezone.utc).isoformat()

            # Write updated metadata
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))

        except Exception as e:
            print(f"Error updating user metadata: {e}")

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        sessions = await self.list_user_sessions(user_id)

        total_messages = sum(session.message_count for session in sessions)
        total_sessions = len(sessions)

        recent_activity = None
        if sessions:
            recent_activity = max(session.last_activity for session in sessions)

        return {
            "user_id": user_id,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "recent_activity": recent_activity.isoformat() if recent_activity else None,
            "sessions": [
                {
                    "session_id": str(session.session_id),
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "message_count": session.message_count
                }
                for session in sessions
            ]
        }