"""
Audit Log Database Model
Provides tamper-proof audit logging with hash chain integrity
"""
from datetime import datetime, timezone
import hashlib
import json
from typing import Optional, Dict, Any, TYPE_CHECKING

from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from .base import Base


class AuditLogModel(Base):
    """
    SQLAlchemy model for audit logs with tamper-proof hash chain

    This model stores comprehensive audit information for all system operations
    including authentication, API calls, database operations, and security events.

    Hash Chain: Each log entry contains a hash of the previous entry, creating
    a tamper-proof chain that can be verified for integrity.
    """
    __tablename__ = 'audit_logs'
    __table_args__ = {'extend_existing': True}

    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )

    # User information
    user_id = Column(String(255), index=True)  # User ID if authenticated
    username = Column(String(255), index=True)  # Username if authenticated
    session_id = Column(String(255), index=True)  # Session identifier

    # Action details
    # Action type (e.g., 'login', 'api_call', 'db_update')
    action = Column(String(100), nullable=False, index=True)
    # Type of resource accessed (e.g., 'user', 'business', 'asset')
    resource_type = Column(String(100), index=True)
    resource_id = Column(String(255))  # ID of the resource accessed

    # Request information
    ip_address = Column(String(45), index=True)  # IPv4 or IPv6 address
    user_agent = Column(Text)  # Browser/client user agent
    request_method = Column(String(10))  # HTTP method
    endpoint = Column(String(500), index=True)  # API endpoint accessed

    # Response information
    status_code = Column(Integer, index=True)  # HTTP status code
    response_time_ms = Column(Integer)  # Response time in milliseconds

    # Data payload (stored as JSON)
    # Request payload (sanitized, no sensitive data)
    request_data = Column(Text)
    response_data = Column(Text)  # Response payload (sanitized)
    error_message = Column(Text)  # Error message if operation failed

    # Security and compliance
    # Severity: info, warning, error, critical
    severity = Column(String(20), default='info', index=True)
    # Category: authentication, authorization, data_access, etc.
    category = Column(String(50), index=True)
    # JSON array of compliance tags (e.g., ['PCI-DSS', 'GDPR'])
    compliance_tags = Column(Text)

    # Tamper-proof hash chain
    previous_hash = Column(String(64))  # SHA-256 hash of previous log
    # SHA-256 hash of current entry
    current_hash = Column(String(64), nullable=False, index=True)

    # Metadata
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Indexes for common queries
    __table_args__ = (
        Index('idx_audit_timestamp_action', 'timestamp', 'action'),
        Index('idx_audit_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_severity_timestamp', 'severity', 'timestamp'),
        Index('idx_audit_category_timestamp', 'category', 'timestamp'),
    )

    def __repr__(self):
        return (
            f"<AuditLog(id={self.id}, action={self.action}, "
            f"user={self.username}, timestamp={self.timestamp})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit log to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'user_id': self.user_id,
            'username': self.username,
            'session_id': self.session_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'request_method': self.request_method,
            'endpoint': self.endpoint,
            'status_code': self.status_code,
            'response_time_ms': self.response_time_ms,
            'request_data': (
                json.loads(self.request_data) if self.request_data else None
            ),
            'response_data': (
                json.loads(self.response_data) if self.response_data else None
            ),
            'error_message': self.error_message,
            'severity': self.severity,
            'category': self.category,
            'compliance_tags': (
                json.loads(self.compliance_tags)
                if self.compliance_tags else []
            ),
            'previous_hash': self.previous_hash,
            'current_hash': self.current_hash,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    @staticmethod
    def calculate_hash(
        log_data: Dict[str, Any],
        previous_hash: Optional[str] = None
    ) -> str:
        """
        Calculate SHA-256 hash for audit log entry

        Args:
            log_data: Dictionary containing log data
            previous_hash: Hash of previous log entry (for chain integrity)

        Returns:
            SHA-256 hash string
        """
        # Create a deterministic string representation of the log data
        # Don't modify the original log_data dictionary
        timestamp = log_data.get('timestamp', '')
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = str(timestamp)

        hash_input = {
            'timestamp': timestamp_str,
            'user_id': log_data.get('user_id', ''),
            'action': log_data.get('action', ''),
            'resource_type': log_data.get('resource_type', ''),
            'resource_id': log_data.get('resource_id', ''),
            'endpoint': log_data.get('endpoint', ''),
            'status_code': log_data.get('status_code', ''),
            'previous_hash': previous_hash or ''
        }

        # Convert to JSON string (sorted keys for consistency)
        hash_string = json.dumps(hash_input, sort_keys=True)

        # Calculate SHA-256 hash
        return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()

    @staticmethod
    def verify_chain_integrity(logs: list):
        """
        Verify the integrity of the audit log chain

        Args:
            logs: List of AuditLogModel instances (ordered by id)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not logs:
            return True, None

        previous_hash = None
        for i, log in enumerate(logs):
            # Verify current hash
            expected_hash = AuditLogModel.calculate_hash(
                log.to_dict(),
                previous_hash
            )
            if log.current_hash != expected_hash:
                return False, (
                    f"Hash mismatch at log ID {log.id}: "
                    f"expected {expected_hash}, got {log.current_hash}"
                )

            # Verify chain link
            if i > 0 and log.previous_hash != logs[i-1].current_hash:
                return False, (
                    f"Chain broken at log ID {log.id}: "
                    f"previous_hash doesn't match"
                )

            previous_hash = log.current_hash

        return True, None


class AuditLogSummary:
    """Summary statistics for audit logs"""

    def __init__(
        self,
        total_logs: int,
        by_action: Dict[str, int],
        by_severity: Dict[str, int],
        by_user: Dict[str, int],
        failed_attempts: int,
        time_range: tuple
    ):
        self.total_logs = total_logs
        self.by_action = by_action
        self.by_severity = by_severity
        self.by_user = by_user
        self.failed_attempts = failed_attempts
        self.time_range = time_range

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary"""
        return {
            'total_logs': self.total_logs,
            'by_action': self.by_action,
            'by_severity': self.by_severity,
            'by_user': self.by_user,
            'failed_attempts': self.failed_attempts,
            'time_range': {
                'start': (
                    self.time_range[0].isoformat()
                    if self.time_range[0] else None
                ),
                'end': (
                    self.time_range[1].isoformat()
                    if self.time_range[1] else None
                )
            }
        }
