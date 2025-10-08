import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import contextmanager
import config

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_file: str = config.DATABASE_FILE):
        self.db_file = db_file
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_file, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT NOT NULL,
                    roles TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                )
            """)
            
            # Device tokens table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS device_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    device_id TEXT UNIQUE NOT NULL,
                    fingerprint TEXT NOT NULL,
                    access_token_hash TEXT NOT NULL,
                    refresh_token_hash TEXT NOT NULL,
                    access_expires_at TEXT NOT NULL,
                    refresh_expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_device_tokens_device_id ON device_tokens(device_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_device_tokens_user_id ON device_tokens(user_id)")
            
            logger.info("Database initialized successfully")
    
    # User operations
    def create_user(self, email: str, username: str, roles: List[str]) -> Optional[int]:
        """Create a new user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO users (email, username, roles, created_at)
                    VALUES (?, ?, ?, ?)
                """, (email, username, json.dumps(roles), datetime.utcnow().isoformat()))
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                logger.warning(f"User {email} already exists")
                return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "email": row["email"],
                    "username": row["username"],
                    "roles": json.loads(row["roles"]),
                    "is_active": bool(row["is_active"]),
                    "created_at": row["created_at"],
                    "last_login": row["last_login"]
                }
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "email": row["email"],
                    "username": row["username"],
                    "roles": json.loads(row["roles"]),
                    "is_active": bool(row["is_active"]),
                    "created_at": row["created_at"],
                    "last_login": row["last_login"]
                }
            return None
    
    def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET last_login = ? WHERE id = ?
            """, (datetime.utcnow().isoformat(), user_id))

    def update_user_roles(self, user_id: int, roles: List[str]):
        """Update user's roles"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET roles = ? WHERE id = ?
            """, (json.dumps(roles), user_id))
            logger.info(f"Updated roles for user {user_id} to {roles}")

    # Device token operations
    def create_device_token(self, user_id: int, username: str, device_id: str, fingerprint: str,
                           access_token_hash: str, refresh_token_hash: str,
                           access_expires_at: str, refresh_expires_at: str) -> Optional[int]:
        """Create a new device token"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO device_tokens 
                    (user_id, username, device_id, fingerprint, access_token_hash, refresh_token_hash,
                     access_expires_at, refresh_expires_at, created_at, last_used_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, username, device_id, fingerprint, access_token_hash, refresh_token_hash,
                      access_expires_at, refresh_expires_at, 
                      datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                logger.warning(f"Device token {device_id} already exists")
                return None
    
    def get_device_token(self, device_id: str) -> Optional[Dict]:
        """Get device token by device_id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM device_tokens WHERE device_id = ?", (device_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "username": row["username"],
                    "device_id": row["device_id"],
                    "fingerprint": row["fingerprint"],
                    "access_token_hash": row["access_token_hash"],
                    "refresh_token_hash": row["refresh_token_hash"],
                    "access_expires_at": row["access_expires_at"],
                    "refresh_expires_at": row["refresh_expires_at"],
                    "created_at": row["created_at"],
                    "last_used_at": row["last_used_at"]
                }
            return None


    
    def update_device_token(self, device_id: str, access_token_hash: str, 
                           access_expires_at: str):
        """Update device token's access token"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE device_tokens 
                SET access_token_hash = ?, access_expires_at = ?, last_used_at = ?
                WHERE device_id = ?
            """, (access_token_hash, access_expires_at, 
                  datetime.utcnow().isoformat(), device_id))
    
    def delete_device_token(self, device_id: str):
        """Delete a device token"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM device_tokens WHERE device_id = ?", (device_id,))
    
    def get_user_devices(self, user_id: int) -> List[Dict]:
        """Get all devices for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT device_id, fingerprint, created_at, last_used_at
                FROM device_tokens WHERE user_id = ?
            """, (user_id,))
            rows = cursor.fetchall()
            return [{
                "device_id": row["device_id"],
                "fingerprint": row["fingerprint"],
                "created_at": row["created_at"],
                "last_used_at": row["last_used_at"]
            } for row in rows]



# Global database instance
db = Database()