"""
Account Delegation Service for JPMorgan Financial APIs
Provides functionality for sharing account access with other users.
"""

import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

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
# DELEGATION STORE (Replace with database in production)
# =============================================================================

class DelegationStore:
    """In-memory storage for account delegations"""
    
    def __init__(self):
        self.delegations = {}  # delegation_id -> delegation
        self.delegation_requests = {}  # request_id -> request
    
    def create_delegation(self, delegation: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new delegation"""
        delegation_id = delegation['delegation_id']
        self.delegations[delegation_id] = delegation
        return delegation
    
    def get_delegation(self, delegation_id: str) -> Optional[Dict[str, Any]]:
        """Get a delegation by ID"""
        return self.delegations.get(delegation_id)
    
    def get_delegations_by_grantor(self, grantor_id: str) -> List[Dict[str, Any]]:
        """Get all delegations where user is the grantor"""
        return [d for d in self.delegations.values() if d.get('grantor_id') == grantor_id]
    
    def get_delegations_by_grantee(self, grantee_id: str) -> List[Dict[str, Any]]:
        """Get all delegations where user is the grantee"""
        return [d for d in self.delegations.values() if d.get('grantee_id') == grantee_id]
    
    def get_delegation_by_account(self, account_id: str, grantee_id: str) -> Optional[Dict[str, Any]]:
        """Get a delegation for a specific account and grantee"""
        for d in self.delegations.values():
            if d.get('account_id') == account_id and d.get('grantee_id') == grantee_id:
                return d
        return None
    
    def update_delegation(self, delegation_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a delegation"""
        if delegation_id in self.delegations:
            self.delegations[delegation_id].update(updates)
            return self.delegations[delegation_id]
        return None
    
    def delete_delegation(self, delegation_id: str) -> bool:
        """Delete a delegation"""
        if delegation_id in self.delegations:
            del self.delegations[delegation_id]
            return True
        return False
    
    def create_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create a delegation request"""
        request_id = request['request_id']
        self.delegation_requests[request_id] = request
        return request
    
    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get a request by ID"""
        return self.delegation_requests.get(request_id)
    
    def get_requests_for_grantor(self, grantor_id: str) -> List[Dict[str, Any]]:
        """Get all pending requests for a grantor"""
        return [r for r in self.delegation_requests.values() 
                if r.get('grantor_id') == grantor_id and r.get('status') == 'pending']
    
    def update_request(self, request_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a request"""
        if request_id in self.delegation_requests:
            self.delegation_requests[request_id].update(updates)
            return self.delegation_requests[request_id]
        return None
    
    def delete_request(self, request_id: str) -> bool:
        """Delete a request"""
        if request_id in self.delegation_requests:
            del self.delegation_requests[request_id]
            return True
        return False


delegation_store = DelegationStore()


# =============================================================================
# DELEGATION SERVICE
# =============================================================================

class DelegationService:
    """Service for managing account delegations"""
    
    def __init__(self):
        self.store = delegation_store
        self.logger = telemetry_logger
    
    # Permission levels
    PERMISSIONS = {
        'view': ['read_account', 'read_transactions'],
        'transactions': ['read_account', 'read_transactions', 'initiate_transfers'],
        'full': ['read_account', 'read_transactions', 'initiate_transfers', 'manage_account']
    }
    
    def request_delegation(self, grantor_id: str, grantee_id: str, account_id: str, 
                          permissions: str = 'view', message: str = '') -> Dict[str, Any]:
        """
        Request access to another user's account
        
        Args:
            grantor_id: The account owner (who is granting access)
            grantee_id: The user requesting access
            account_id: The account to access
            permissions: Permission level ('view', 'transactions', 'full')
            message: Optional message to the grantor
            
        Returns:
            Dict with request result
        """
        if permissions not in self.PERMISSIONS:
            return {
                'status': 'error',
                'message': f'Invalid permissions. Must be one of: {", ".join(self.PERMISSIONS.keys())}'
            }
        
        # Check if there's already an active delegation
        existing = self.store.get_delegation_by_account(account_id, grantee_id)
        if existing and existing.get('status') == 'active':
            return {
                'status': 'error',
                'message': 'Access already granted to this user for this account'
            }
        
        # Check for pending request
        for req in self.store.delegation_requests.values():
            if req.get('grantor_id') == grantor_id and req.get('grantee_id') == grantee_id:
                if req.get('account_id') == account_id and req.get('status') == 'pending':
                    return {
                        'status': 'error',
                        'message': 'A pending request already exists'
                    }
        
        request_id = f"DELREQ-{secrets.token_hex(4).upper()}"
        
        request = {
            'request_id': request_id,
            'grantor_id': grantor_id,
            'grantee_id': grantee_id,
            'account_id': account_id,
            'requested_permissions': permissions,
            'message': message,
            'status': 'pending',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'expires_at': (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        }
        
        self.store.create_request(request)
        
        self.logger.log_info(
            f"Delegation request created: {request_id} from {grantee_id} to {grantor_id}",
            {'context': 'delegation'}
        )
        
        return {
            'status': 'success',
            'request': request
        }
    
    def approve_request(self, request_id: str) -> Dict[str, Any]:
        """
        Approve a delegation request
        
        Args:
            request_id: The request ID
            
        Returns:
            Dict with result
        """
        request = self.store.get_request(request_id)
        
        if not request:
            return {
                'status': 'error',
                'message': 'Request not found'
            }
        
        if request['status'] != 'pending':
            return {
                'status': 'error',
                'message': f'Request already processed with status: {request["status"]}'
            }
        
        # Check expiration
        if datetime.fromisoformat(request['expires_at']) < datetime.now(timezone.utc):
            self.store.update_request(request_id, {'status': 'expired'})
            return {
                'status': 'error',
                'message': 'Request has expired'
            }
        
        # Create delegation
        delegation_id = f"DEL-{secrets.token_hex(4).upper()}"
        
        delegation = {
            'delegation_id': delegation_id,
            'grantor_id': request['grantor_id'],
            'grantee_id': request['grantee_id'],
            'account_id': request['account_id'],
            'permissions': request['requested_permissions'],
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'expires_at': (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        }
        
        self.store.create_delegation(delegation)
        self.store.update_request(request_id, {
            'status': 'approved',
            'delegation_id': delegation_id,
            'processed_at': datetime.now(timezone.utc).isoformat()
        })
        
        self.logger.log_info(
            f"Delegation approved: {delegation_id}",
            {'context': 'delegation'}
        )
        
        return {
            'status': 'success',
            'delegation': delegation
        }
    
    def reject_request(self, request_id: str, reason: str = '') -> Dict[str, Any]:
        """
        Reject a delegation request
        
        Args:
            request_id: The request ID
            reason: Optional reason for rejection
            
        Returns:
            Dict with result
        """
        request = self.store.get_request(request_id)
        
        if not request:
            return {
                'status': 'error',
                'message': 'Request not found'
            }
        
        if request['status'] != 'pending':
            return {
                'status': 'error',
                'message': f'Request already processed with status: {request["status"]}'
            }
        
        self.store.update_request(request_id, {
            'status': 'rejected',
            'rejection_reason': reason,
            'processed_at': datetime.now(timezone.utc).isoformat()
        })
        
        self.logger.log_info(
            f"Delegation request rejected: {request_id}",
            {'context': 'delegation'}
        )
        
        return {
            'status': 'success',
            'message': 'Request rejected'
        }
    
    def revoke_delegation(self, delegation_id: str) -> Dict[str, Any]:
        """
        Revoke an active delegation
        
        Args:
            delegation_id: The delegation ID
            
        Returns:
            Dict with result
        """
        delegation = self.store.get_delegation(delegation_id)
        
        if not delegation:
            return {
                'status': 'error',
                'message': 'Delegation not found'
            }
        
        self.store.update_delegation(delegation_id, {
            'status': 'revoked',
            'revoked_at': datetime.now(timezone.utc).isoformat()
        })
        
        self.logger.log_info(
            f"Delegation revoked: {delegation_id}",
            {'context': 'delegation'}
        )
        
        return {
            'status': 'success',
            'message': 'Delegation revoked'
        }
    
    def get_delegations(self, user_id: str) -> Dict[str, Any]:
        """
        Get all delegations for a user (both as grantor and grantee)
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dict with delegations
        """
        as_grantor = self.store.get_delegations_by_grantor(user_id)
        as_grantee = self.store.get_delegations_by_grantee(user_id)
        
        return {
            'status': 'success',
            'as_grantor': as_grantor,
            'as_grantee': as_grantee,
            'total': len(as_grantor) + len(as_grantee)
        }
    
    def get_pending_requests(self, user_id: str) -> Dict[str, Any]:
        """
        Get pending delegation requests for a user
        
        Args:
            user_id: The user's ID (as grantor)
            
        Returns:
            Dict with pending requests
        """
        requests = self.store.get_requests_for_grantor(user_id)
        
        return {
            'status': 'success',
            'requests': requests,
            'count': len(requests)
        }
    
    def check_access(self, user_id: str, account_id: str, required_permission: str) -> Dict[str, Any]:
        """
        Check if a user has access to an account through delegation
        
        Args:
            user_id: The user to check
            account_id: The account to access
            required_permission: The permission required
            
        Returns:
            Dict with access status
        """
        delegation = self.store.get_delegation_by_account(account_id, user_id)
        
        if not delegation:
            return {
                'status': 'error',
                'has_access': False,
                'message': 'No delegation found'
            }
        
        if delegation.get('status') != 'active':
            return {
                'status': 'error',
                'has_access': False,
                'message': f'Delegation status: {delegation.get("status")}'
            }
        
        # Check expiration
        if datetime.fromisoformat(delegation['expires_at']) < datetime.now(timezone.utc):
            return {
                'status': 'error',
                'has_access': False,
                'message': 'Delegation expired'
            }
        
        # Check permissions
        permission_level = delegation.get('permissions', 'view')
        allowed_permissions = self.PERMISSIONS.get(permission_level, [])
        
        if required_permission not in allowed_permissions:
            return {
                'status': 'error',
                'has_access': False,
                'message': f'Required permission: {required_permission}, granted: {permission_level}'
            }
        
        return {
            'status': 'success',
            'has_access': True,
            'permissions': allowed_permissions,
            'delegation_id': delegation['delegation_id']
        }
    
    def extend_delegation(self, delegation_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Extend a delegation's expiration
        
        Args:
            delegation_id: The delegation ID
            days: Number of days to extend
            
        Returns:
            Dict with result
        """
        delegation = self.store.get_delegation(delegation_id)
        
        if not delegation:
            return {
                'status': 'error',
                'message': 'Delegation not found'
            }
        
        current_expires = datetime.fromisoformat(delegation['expires_at'])
        new_expires = current_expires + timedelta(days=days)
        
        self.store.update_delegation(delegation_id, {
            'expires_at': new_expires.isoformat()
        })
        
        return {
            'status': 'success',
            'message': f'Delegation extended by {days} days',
            'new_expires_at': new_expires.isoformat()
        }


# Global service instance
delegation_service = DelegationService()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DelegationService',
    'delegation_service',
    'DelegationStore',
    'delegation_store'
]
