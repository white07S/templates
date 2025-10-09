#!/usr/bin/env python3
"""Script to view active Graph sessions with user information."""

from database import db

def main():
    print("\n=== Active Sessions with User Information ===\n")

    sessions = db.list_sessions()

    if not sessions:
        print("No active sessions found.")
        return

    for session in sessions:
        user = db.get_user_by_id(session["user_id"]) or {}
        print(f"Session ID: {session['id']}")
        print(f"  User ID: {session['user_id']}")
        print(f"  Username: {user.get('username')}")
        print(f"  Email: {user.get('email')}")
        print(f"  Roles: {', '.join(user.get('roles', []))}")
        print(f"  Device ID: {session['device_id']}")
        print(f"  Fingerprint: {session['fingerprint']}")
        print(f"  Expires At: {session.get('token_expires_at')}")
        print(f"  Created: {session['created_at']}")
        print(f"  Last Used: {session['last_used_at']}")
        print("-" * 60)

if __name__ == "__main__":
    main()
