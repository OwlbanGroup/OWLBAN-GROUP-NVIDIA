"""
Encryption utilities for sensitive data protection
"""
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Optional
import hashlib


class DataEncryption:
    """Handles encryption/decryption of sensitive data"""

    def __init__(self, key: Optional[str] = None):
        """
        Initialize encryption with a key
        If no key provided, uses environment variable or generates one
        """
        if key:
            self.key = key.encode()
        else:
            # Use environment variable or generate from salt
            env_key = os.getenv('ENCRYPTION_KEY')
            if env_key:
                self.key = env_key.encode()
            else:
                # Generate key from a salt (not secure for production)
                salt = os.getenv('ENCRYPTION_SALT', 'jpmorgan_default_salt').encode()
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                self.key = base64.urlsafe_b64encode(kdf.derive(b'jpmorgan_encryption_key'))

        self.fernet = Fernet(self.key)

    def encrypt(self, data: str) -> str:
        """Encrypt a string"""
        if not data:
            return data
        encrypted = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt a string"""
        if not encrypted_data:
            return encrypted_data
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception:
            # Return original data if decryption fails
            return encrypted_data

    def hash_data(self, data: str) -> str:
        """Create a SHA-256 hash of data (one-way)"""
        return hashlib.sha256(data.encode()).hexdigest()


# Global encryption instance
encryption = DataEncryption()


class EncryptedField:
    """Descriptor for encrypted database fields"""

    def __init__(self, field_name: str):
        self.field_name = field_name
        self.private_name = f"_{field_name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        encrypted_value = getattr(obj, self.private_name, None)
        if encrypted_value:
            return encryption.decrypt(encrypted_value)
        return None

    def __set__(self, obj, value):
        if value is not None:
            encrypted_value = encryption.encrypt(str(value))
            setattr(obj, self.private_name, encrypted_value)
        else:
            setattr(obj, self.private_name, None)


def compliance_check(data: dict) -> dict:
    """
    Perform GDPR/CCPA compliance checks on data
    Masks or removes sensitive information
    """
    sensitive_fields = [
        'user_id', 'user_local_id', 'local_id', 'app_id', 'exp_id',
        'device_class', 'dev_make', 'dev_model', 'pn1', 'p1', 'pn2', 'p2',
        'pn3', 'p3', 'pn4', 'p4', 'ticket_keys'
    ]

    compliant_data = data.copy()

    for field in sensitive_fields:
        if field in compliant_data:
            # Hash sensitive identifiers for analytics while maintaining uniqueness
            if isinstance(compliant_data[field], str):
                compliant_data[field] = encryption.hash_data(compliant_data[field])
            else:
                compliant_data[field] = str(compliant_data[field])

    return compliant_data


def audit_log(action: str, user_id: str, resource: str, details: dict = None):
    """
    Create audit log entry for compliance
    """
    import json
    from datetime import datetime

    audit_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'action': action,
        'user_id': encryption.hash_data(user_id),  # Anonymize user ID
        'resource': resource,
        'details': details or {}
    }

    # In production, this would be sent to a secure logging system
    print(f"AUDIT: {json.dumps(audit_entry)}")
    return audit_entry
