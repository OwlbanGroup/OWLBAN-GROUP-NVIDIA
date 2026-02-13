"""
Multi-Factor Authentication (MFA) Service for JPMorgan Financial APIs
Provides TOTP and SMS-based MFA functionality.
"""

import secrets
import pyotp
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import hashlib

try:
    from src.logger import telemetry_logger
except ImportError:
    class FallbackLogger:
        def log_info(self, msg, context=None):
            print(f"INFO: {msg}")
        def log_error(self, msg, context=None):
            print(f"ERROR: {msg}")
    telemetry_logger = FallbackLogger()


# =============================================================================
# MFA STORE (Replace with database in production)
# =============================================================================

class MFAStore:
    """In-memory storage for MFA data"""
    
    def __init__(self):
        self.mfa_config = {}  # user_id -> MFA config
        self.mfa_codes = {}   # verification codes
        self.mfa_sessions = {}  # active MFA sessions
    
    def get_mfa_config(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get MFA config for a user"""
        return self.mfa_config.get(user_id)
    
    def set_mfa_config(self, user_id: str, config: Dict[str, Any]) -> None:
        """Set MFA config for a user"""
        self.mfa_config[user_id] = config
    
    def delete_mfa_config(self, user_id: str) -> bool:
        """Delete MFA config for a user"""
        if user_id in self.mfa_config:
            del self.mfa_config[user_id]
            return True
        return False
    
    def store_verification_code(self, code: str, data: Dict[str, Any], expires_in: int = 300) -> None:
        """Store verification code with expiration"""
        self.mfa_codes[code] = {
            **data,
            'expires_at': datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        }
    
    def get_verification_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Get and validate verification code"""
        if code not in self.mfa_codes:
            return None
        
        code_data = self.mfa_codes[code]
        if datetime.now(timezone.utc) > code_data['expires_at']:
            del self.mfa_codes[code]
            return None
        
        return code_data
    
    def delete_verification_code(self, code: str) -> None:
        """Delete a verification code"""
        if code in self.mfa_codes:
            del self.mfa_codes[code]
    
    def create_mfa_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Create an MFA session"""
        self.mfa_sessions[session_id] = {
            **data,
            'created_at': datetime.now(timezone.utc),
            'expires_at': datetime.now(timezone.utc) + timedelta(minutes=5)
        }
    
    def get_mfa_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get MFA session"""
        if session_id not in self.mfa_sessions:
            return None
        
        session = self.mfa_sessions[session_id]
        if datetime.now(timezone.utc) > session['expires_at']:
            del self.mfa_sessions[session_id]
            return None
        
        return session
    
    def delete_mfa_session(self, session_id: str) -> None:
        """Delete an MFA session"""
        if session_id in self.mfa_sessions:
            del self.mfa_sessions[session_id]


mfa_store = MFAStore()


# =============================================================================
# MFA SERVICE
# =============================================================================

class MFAService:
    """Service for multi-factor authentication"""
    
    def __init__(self):
        self.store = mfa_store
        self.logger = telemetry_logger
    
    def setup_mfa(self, user_id: str, method: str = 'totp') -> Dict[str, Any]:
        """
        Setup MFA for a user
        
        Args:
            user_id: The user's ID
            method: MFA method ('totp', 'sms', 'email')
            
        Returns:
            Dict with MFA setup data
        """
        if method not in ['totp', 'sms', 'email']:
            return {
                'status': 'error',
                'message': 'Invalid MFA method'
            }
        
        # Generate secret
        secret = pyotp.random_base32()
        
        # Generate backup codes
        backup_codes = [secrets.token_hex(8).upper() for _ in range(10)]
        
        config = {
            'user_id': user_id,
            'method': method,
            'secret': secret,
            'backup_codes': backup_codes,
            'enabled': False,  # Not enabled until verified
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        self.store.set_mfa_config(user_id, config)
        
        # Generate provisioning URI for TOTP
        if method == 'totp':
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=user_id,
                issuer_name='JPMorgan Financial'
            )
            config['provisioning_uri'] = provisioning_uri
        
        self.logger.log_info(f"MFA setup initiated for user: {user_id}", {'context': 'mfa'})
        
        return {
            'status': 'success',
            'mfa_config': config
        }
    
    def verify_and_enable_mfa(self, user_id: str, code: str) -> Dict[str, Any]:
        """
        Verify MFA code and enable MFA
        
        Args:
            user_id: The user's ID
            code: Verification code
            
        Returns:
            Dict with verification result
        """
        config = self.store.get_mfa_config(user_id)
        
        if not config:
            return {
                'status': 'error',
                'message': 'MFA not setup for this user'
            }
        
        # Verify the code
        if config['method'] == 'totp':
            totp = pyotp.TOTP(config['secret'])
            if not totp.verify(code):
                # Check backup codes
                if code.upper() in config.get('backup_codes', []):
                    # Remove used backup code
                    config['backup_codes'].remove(code.upper())
                    self.store.set_mfa_config(user_id, config)
                    self._enable_mfa(user_id)
                    return {
                        'status': 'success',
                        'message': 'MFA enabled using backup code'
                    }
                
                self.logger.log_info(f"Invalid MFA code for user: {user_id}", {'context': 'mfa'})
                return {
                    'status': 'error',
                    'message': 'Invalid verification code'
                }
        elif config['method'] in ['sms', 'email']:
            # Verify SMS/email code
            code_data = self.store.get_verification_code(code)
            if not code_data or code_data.get('user_id') != user_id:
                return {
                    'status': 'error',
                    'message': 'Invalid or expired verification code'
                }
            self.store.delete_verification_code(code)
        
        self._enable_mfa(user_id)
        
        return {
            'status': 'success',
            'message': 'MFA enabled successfully'
        }
    
    def _enable_mfa(self, user_id: str) -> None:
        """Enable MFA for a user"""
        config = self.store.get_mfa_config(user_id)
        if config:
            config['enabled'] = True
            config['enabled_at'] = datetime.now(timezone.utc).isoformat()
            self.store.set_mfa_config(user_id, config)
            self.logger.log_info(f"MFA enabled for user: {user_id}", {'context': 'mfa'})
    
    def verify_mfa(self, user_id: str, code: str) -> Dict[str, Any]:
        """
        Verify MFA code during login
        
        Args:
            user_id: The user's ID
            code: Verification code
            
        Returns:
            Dict with verification result
        """
        config = self.store.get_mfa_config(user_id)
        
        if not config or not config.get('enabled'):
            return {
                'status': 'error',
                'message': 'MFA not enabled for this user'
            }
        
        # Verify the code
        if config['method'] == 'totp':
            totp = pyotp.TOTP(config['secret'])
            if totp.verify(code):
                return {
                    'status': 'success',
                    'message': 'MFA verification successful'
                }
            
            # Check backup codes
            if code.upper() in config.get('backup_codes', []):
                return {
                    'status': 'success',
                    'message': 'MFA verification successful (backup code)'
                }
            
            return {
                'status': 'error',
                'message': 'Invalid verification code'
            }
        
        elif config['method'] in ['sms', 'email']:
            code_data = self.store.get_verification_code(code)
            if not code_data or code_data.get('user_id') != user_id:
                return {
                    'status': 'error',
                    'message': 'Invalid or expired verification code'
                }
            
            self.store.delete_verification_code(code)
            
            return {
                'status': 'success',
                'message': 'MFA verification successful'
            }
    
    def send_verification_code(self, user_id: str, destination: str = None) -> Dict[str, Any]:
        """
        Send verification code via SMS or email
        
        Args:
            user_id: The user's ID
            destination: Phone number or email (optional, uses config if not provided)
            
        Returns:
            Dict with result
        """
        config = self.store.get_mfa_config(user_id)
        
        if not config:
            return {
                'status': 'error',
                'message': 'MFA not setup for this user'
            }
        
        # Generate 6-digit code
        code = f"{secrets.randbelow(1000000):06d}"
        
        # Store code
        self.store.store_verification_code(code, {
            'user_id': user_id,
            'method': config['method']
        })
        
        # In production, send via SMS/email
        # For now, just log it
        self.logger.log_info(
            f"MFA code for user {user_id}: {code}",
            {'context': 'mfa', 'method': config['method']}
        )
        
        return {
            'status': 'success',
            'message': f'Verification code sent via {config["method"]}',
            'code': code  # Only for testing!
        }
    
    def disable_mfa(self, user_id: str) -> Dict[str, Any]:
        """
        Disable MFA for a user
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dict with result
        """
        config = self.store.get_mfa_config(user_id)
        
        if not config:
            return {
                'status': 'error',
                'message': 'MFA not setup for this user'
            }
        
        self.store.delete_mfa_config(user_id)
        
        self.logger.log_info(f"MFA disabled for user: {user_id}", {'context': 'mfa'})
        
        return {
            'status': 'success',
            'message': 'MFA disabled successfully'
        }
    
    def get_mfa_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get MFA status for a user
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dict with MFA status
        """
        config = self.store.get_mfa_config(user_id)
        
        if not config:
            return {
                'status': 'success',
                'enabled': False,
                'method': None
            }
        
        return {
            'status': 'success',
            'enabled': config.get('enabled', False),
            'method': config.get('method'),
            'backup_codes_remaining': len(config.get('backup_codes', []))
        }
    
    def create_trusted_device(self, user_id: str, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a trusted device for passwordless login
        
        Args:
            user_id: The user's ID
            device_info: Device information
            
        Returns:
            Dict with trusted device data
        """
        device_id = secrets.token_hex(16)
        
        trusted_device = {
            'device_id': device_id,
            'user_id': user_id,
            'device_name': device_info.get('device_name', 'Unknown Device'),
            'device_type': device_info.get('device_type', 'unknown'),
            'ip_address': device_info.get('ip_address', ''),
            'user_agent': device_info.get('user_agent', ''),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_used': datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.log_info(f"Trusted device created for user: {user_id}", {'context': 'mfa'})
        
        return {
            'status': 'success',
            'trusted_device': trusted_device
        }


# Global service instance
mfa_service = MFAService()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MFAService',
    'mfa_service',
    'MFAStore',
    'mfa_store'
]
