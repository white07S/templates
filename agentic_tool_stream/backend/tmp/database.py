import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tinydb import TinyDB, Query

import config

logger = logging.getLogger(__name__)


class Database:
    """TinyDB-backed persistence layer for users and authentication sessions."""

    def __init__(self, db_file: str = config.DATABASE_FILE):
        self.db_path = Path(db_file)
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = TinyDB(self.db_path)
        self.users = self.db.table("users")
        self.sessions = self.db.table("sessions")

    # -------------------------------
    # User operations
    # -------------------------------
    def create_or_update_user(
        self,
        azure_object_id: str,
        email: str,
        username: str,
        roles: List[str],
    ) -> Dict:
        """Create a user if missing or update their roles/metadata."""
        now = datetime.utcnow().isoformat()
        User = Query()
        existing = self.users.get(User.azure_object_id == azure_object_id)

        if existing:
            updated_roles = sorted(set(roles))
            self.users.update(
                {
                    "email": email,
                    "username": username,
                    "roles": updated_roles,
                    "last_login": now,
                    "is_active": existing.get("is_active", True),
                },
                doc_ids=[existing.doc_id],
            )
            logger.info("Updated user %s roles to %s", email, updated_roles)
            return self.users.get(doc_id=existing.doc_id)

        user_doc = {
            "id": uuid.uuid4().hex,
            "azure_object_id": azure_object_id,
            "email": email,
            "username": username,
            "roles": sorted(set(roles)),
            "is_active": True,
            "created_at": now,
            "last_login": now,
        }
        doc_id = self.users.insert(user_doc)
        logger.info("Created new user %s (%s)", email, user_doc["id"])
        return self.users.get(doc_id=doc_id)

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        User = Query()
        return self.users.get(User.id == user_id)

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        User = Query()
        return self.users.get(User.email == email)

    # -------------------------------
    # Session operations
    # -------------------------------
    def create_session(
        self,
        user_id: str,
        username: str,
        device_id: str,
        fingerprint: str,
        token_hash: str,
        token_expires_at: str,
        azure_object_id: str,
        azure_tenant_id: Optional[str] = None,
        azure_token_expires_on: Optional[str] = None,
    ) -> Dict:
        """Persist a session tied to a hashed Graph access token."""
        now = datetime.utcnow().isoformat()
        session_doc = {
            "id": uuid.uuid4().hex,
            "user_id": user_id,
            "username": username,
            "azure_object_id": azure_object_id,
            "azure_tenant_id": azure_tenant_id,
            "device_id": device_id,
            "fingerprint": fingerprint,
            "token_hash": token_hash,
            "token_expires_at": token_expires_at,
            "azure_token_expires_on": azure_token_expires_on,
            "created_at": now,
            "last_used_at": now,
        }

        doc_id = self.sessions.insert(session_doc)
        logger.info("Created session for user_id=%s device_id=%s", user_id, device_id)
        return self.sessions.get(doc_id=doc_id)

    def get_session_by_token_hash(self, token_hash: str) -> Optional[Dict]:
        Session = Query()
        session = self.sessions.get(Session.token_hash == token_hash)
        if session:
            logger.debug("Resolved session %s for token hash", session["id"])
        return session

    def touch_session(self, session_id: str, token_expires_at: Optional[str] = None):
        Session = Query()
        updates = {"last_used_at": datetime.utcnow().isoformat()}
        if token_expires_at:
            updates["token_expires_at"] = token_expires_at
        self.sessions.update(updates, Session.id == session_id)

    def delete_session_by_device(self, device_id: str):
        Session = Query()
        self.sessions.remove(Session.device_id == device_id)
        logger.info("Deleted session for device_id=%s", device_id)

    def delete_session_by_token_hash(self, token_hash: str):
        Session = Query()
        self.sessions.remove(Session.token_hash == token_hash)
        logger.info("Deleted session for token hash")

    def get_user_devices(self, user_id: str) -> List[Dict]:
        Session = Query()
        sessions = self.sessions.search(Session.user_id == user_id)
        return [
            {
                "device_id": session["device_id"],
                "fingerprint": session["fingerprint"],
                "created_at": session["created_at"],
                "last_used_at": session["last_used_at"],
                "expires_at": session["token_expires_at"],
            }
            for session in sessions
        ]

    def list_sessions(self) -> List[Dict]:
        return list(self.sessions.all())


# Global database instance
db = Database()
