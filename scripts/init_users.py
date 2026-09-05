#!/usr/bin/env python3
"""
Initialize default users for OWLBAN GROUP authentication system.
Seeds admin and demo users across all platforms.
Run this after starting the database but before starting auth services.
"""

import sys
import os
import json
import logging

# Suppress auth_lib INFO logging to stderr so scripts exit cleanly.
# This keeps PowerShell/Docker from treating normal output as an error.
logging.disable(logging.CRITICAL)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_lib import create_user, auth_manager


def init_users():
    """Create default users for all platforms."""
    users = [
        {
            "email": "admin@owlban.com",
            "username": "admin",
            "password": "Admin2024!",
            "role": "admin",
            "company": "OWLBAN_GROUP",
            "permissions": ["read", "write", "delete", "admin", "manage_users"]
        },
        {
            "email": "demo@owlban.com",
            "username": "demouser",
            "password": "Demo2024!",
            "role": "user",
            "company": "OWLBAN_GROUP",
            "permissions": ["read", "write"]
        },
        {
            "email": "oscar@owlban.com",
            "username": "oscaruser",
            "password": "Oscar2024!",
            "role": "executive",
            "company": "OSCAR_BROOME",
            "permissions": ["read", "write", "view_reports"]
        },
        {
            "email": "ai@owlban.com",
            "username": "aiuser",
            "password": "Ai2024!!",
            "role": "developer",
            "company": "BLACKBOX_AI",
            "permissions": ["read", "write", "api_access", "model_deploy"]
        },
    ]

    created = 0
    for user_data in users:
        success, message = create_user(
            email=user_data["email"],
            username=user_data["username"],
            password=user_data["password"],
            role=user_data["role"],
            company=user_data["company"],
            permissions=user_data["permissions"]
        )
        if success:
            print(f"Created user: {user_data['email']} ({user_data['company']})")
            created += 1
        else:
            print(f"Skipped user: {user_data['email']} - {message}")

    print(f"\nTotal users created: {created}")
    print(f"Total users in system: {len(auth_manager.users)}")
    return created


if __name__ == "__main__":
    print("=" * 50)
    print("OWLBAN GROUP - User Initialization")
    print("=" * 50)
    init_users()
    print("=" * 50)
    print("Done!")
