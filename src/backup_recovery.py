"""
Backup and Recovery Service for JPMorgan Financial APIs
Provides database backup and restore functionality.
"""

import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
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
# BACKUP STORE (Replace with database/file storage in production)
# =============================================================================

class BackupStore:
    """In-memory storage for backup metadata"""
    
    def __init__(self):
        self.backups = {}  # backup_id -> backup metadata
        self.restore_points = {}  # restore_id -> restore metadata
    
    def create_backup_record(self, backup: Dict[str, Any]) -> Dict[str, Any]:
        """Create a backup record"""
        backup_id = backup['backup_id']
        self.backups[backup_id] = backup
        return backup
    
    def get_backup(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Get a backup by ID"""
        return self.backups.get(backup_id)
    
    def get_backups_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all backups for a user"""
        return [b for b in self.backups.values() if b.get('user_id') == user_id]
    
    def get_all_backups(self) -> List[Dict[str, Any]]:
        """Get all backups"""
        return list(self.backups.values())
    
    def delete_backup_record(self, backup_id: str) -> bool:
        """Delete a backup record"""
        if backup_id in self.backups:
            del self.backups[backup_id]
            return True
        return False
    
    def create_restore_record(self, restore: Dict[str, Any]) -> Dict[str, Any]:
        """Create a restore record"""
        restore_id = restore['restore_id']
        self.restore_points[restore_id] = restore
        return restore
    
    def get_restore(self, restore_id: str) -> Optional[Dict[str, Any]]:
        """Get a restore by ID"""
        return self.restore_points.get(restore_id)
    
    def get_restores_by_backup(self, backup_id: str) -> List[Dict[str, Any]]:
        """Get all restores for a backup"""
        return [r for r in self.restore_points.values() if r.get('backup_id') == backup_id]


backup_store = BackupStore()


# =============================================================================
# BACKUP SERVICE
# =============================================================================

class BackupService:
    """Service for managing backups"""
    
    def __init__(self, backup_dir: str = None):
        self.store = backup_store
        self.logger = telemetry_logger
        self.backup_dir = backup_dir or os.path.join(os.getcwd(), 'backups')
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self, user_id: str, data: Dict[str, Any], backup_type: str = 'full') -> Dict[str, Any]:
        """
        Create a backup
        
        Args:
            user_id: The user's ID
            data: Data to backup
            backup_type: Type of backup ('full', 'incremental', 'config')
            
        Returns:
            Dict with backup result
        """
        import secrets
        
        timestamp = datetime.now(timezone.utc)
        backup_id = f"BACKUP-{timestamp.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4).upper()}"
        
        # Create backup file
        backup_filename = f"{backup_id}.json"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        # Write data to backup file
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Calculate checksum
        checksum = self._calculate_checksum(backup_path)
        
        # Get file size
        file_size = os.path.getsize(backup_path)
        
        backup = {
            'backup_id': backup_id,
            'user_id': user_id,
            'backup_type': backup_type,
            'filename': backup_filename,
            'file_path': backup_path,
            'file_size': file_size,
            'checksum': checksum,
            'status': 'completed',
            'created_at': timestamp.isoformat(),
            'expires_at': (timestamp + timedelta(days=30)).isoformat()
        }
        
        self.store.create_backup_record(backup)
        
        self.logger.log_info(f"Backup created: {backup_id}", {'context': 'backup'})
        
        return {
            'status': 'success',
            'backup': backup
        }
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """
        Verify a backup's integrity
        
        Args:
            backup_id: The backup ID
            
        Returns:
            Dict with verification result
        """
        backup = self.store.get_backup(backup_id)
        
        if not backup:
            return {
                'status': 'error',
                'message': 'Backup not found'
            }
        
        # Check if file exists
        if not os.path.exists(backup['file_path']):
            return {
                'status': 'error',
                'message': 'Backup file not found'
            }
        
        # Calculate checksum
        current_checksum = self._calculate_checksum(backup['file_path'])
        
        if current_checksum != backup['checksum']:
            return {
                'status': 'error',
                'message': 'Backup integrity check failed',
                'expected': backup['checksum'],
                'actual': current_checksum
            }
        
        return {
            'status': 'success',
            'message': 'Backup integrity verified',
            'checksum': current_checksum
        }
    
    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """
        Restore from a backup
        
        Args:
            backup_id: The backup ID
            
        Returns:
            Dict with restore result
        """
        import secrets
        
        backup = self.store.get_backup(backup_id)
        
        if not backup:
            return {
                'status': 'error',
                'message': 'Backup not found'
            }
        
        # Verify backup first
        verification = self.verify_backup(backup_id)
        if verification['status'] != 'success':
            return verification
        
        # Read backup data
        with open(backup['file_path'], 'r') as f:
            data = json.load(f)
        
        # Create restore record
        restore_id = f"RESTORE-{secrets.token_hex(4).UPPER()}"
        
        restore = {
            'restore_id': restore_id,
            'backup_id': backup_id,
            'user_id': backup['user_id'],
            'status': 'completed',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        self.store.create_restore_record(restore)
        
        self.logger.log_info(f"Backup restored: {backup_id}", {'context': 'backup'})
        
        return {
            'status': 'success',
            'restore': restore,
            'data': data
        }
    
    def list_backups(self, user_id: str = None) -> Dict[str, Any]:
        """
        List all backups
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            Dict with list of backups
        """
        if user_id:
            backups = self.store.get_backups_by_user(user_id)
        else:
            backups = self.store.get_all_backups()
        
        return {
            'status': 'success',
            'backups': backups,
            'count': len(backups)
        }
    
    def delete_backup(self, backup_id: str) -> Dict[str, Any]:
        """
        Delete a backup
        
        Args:
            backup_id: The backup ID
            
        Returns:
            Dict with result
        """
        backup = self.store.get_backup(backup_id)
        
        if not backup:
            return {
                'status': 'error',
                'message': 'Backup not found'
            }
        
        # Delete file
        if os.path.exists(backup['file_path']):
            os.remove(backup['file_path'])
        
        # Delete record
        self.store.delete_backup_record(backup_id)
        
        self.logger.log_info(f"Backup deleted: {backup_id}", {'context': 'backup'})
        
        return {
            'status': 'success',
            'message': 'Backup deleted successfully'
        }
    
    def create_auto_backup(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an automated backup with rotation
        
        Args:
            user_id: The user's ID
            data: Data to backup
            
        Returns:
            Dict with backup result
        """
        # Create the backup
        result = self.create_backup(user_id, data, 'auto')
        
        # Clean up old backups (keep last 7)
        backups = self.store.get_backups_by_user(user_id)
        backups.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Delete backups beyond the last 7
        for old_backup in backups[7:]:
            self.delete_backup(old_backup['backup_id'])
        
        return result
    
    def export_backup(self, backup_id: str, format: str = 'json') -> Dict[str, Any]:
        """
        Export a backup in a specific format
        
        Args:
            backup_id: The backup ID
            format: Export format ('json', 'tar.gz')
            
        Returns:
            Dict with export result
        """
        backup = self.store.get_backup(backup_id)
        
        if not backup:
            return {
                'status': 'error',
                'message': 'Backup not found'
            }
        
        if format == 'json':
            return {
                'status': 'success',
                'backup': backup,
                'file_path': backup['file_path']
            }
        
        elif format == 'tar.gz':
            # Create tar archive
            tar_filename = f"{backup_id}.tar.gz"
            tar_path = os.path.join(self.backup_dir, tar_filename)
            
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(backup['file_path'], arcname=f"{backup_id}.json")
            
            return {
                'status': 'success',
                'archive_path': tar_path,
                'file_size': os.path.getsize(tar_path)
            }
        
        return {
            'status': 'error',
            'message': 'Unsupported format'
        }
    
    def get_backup_info(self, backup_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a backup
        
        Args:
            backup_id: The backup ID
            
        Returns:
            Dict with backup information
        """
        backup = self.store.get_backup(backup_id)
        
        if not backup:
            return {
                'status': 'error',
                'message': 'Backup not found'
            }
        
        # Get restore history
        restores = self.store.get_restores_by_backup(backup_id)
        
        return {
            'status': 'success',
            'backup': backup,
            'restore_count': len(restores),
            'restores': restores
        }


# Global service instance
backup_service = BackupService()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BackupService',
    'backup_service',
    'BackupStore',
    'backup_store'
]
