"""
Comprehensive Audit Logging System for JPMorgan Financial APIs
Provides tamper-proof logging of all critical operations and financial transactions
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum
import structlog
from sqlalchemy import Column, Integer, String, DateTime, Text, Index
from sqlalchemy.ext.declarative import declarative_base

from .logger import telemetry_logger

Base = declarative_base()

class AuditEventType(str, Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_GENERATED = "token_generated"
    TOKEN_REVOKED = "token_revoked"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET = "password_reset"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    
    # Data access events
    DATA_READ = "data_read"
    DATA_CREATED = "data_created"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    
    # Financial transaction events
    TRANSACTION_INITIATED = "transaction_initiated"
    TRANSACTION_COMPLETED = "transaction_completed"
    TRANSACTION_FAILED = "transaction_failed"
    TRANSACTION_REVERSED = "transaction_reversed"
    
    # Business operations
    BUSINESS_CREATED = "business_created"
    BUSINESS_UPDATED = "business_updated"
    BUSINESS_DELETED = "business_deleted"
    ASSET_CREATED = "asset_created"
    ASSET_UPDATED = "asset_updated"
    ASSET_DELETED = "asset_deleted"
    
    # Security events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"
    
    # System events
    CONFIG_CHANGED = "config_changed"
    SYSTEM_ERROR = "system_error"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    
    # Compliance events
    DATA_RETENTION_APPLIED = "data_retention_applied"
    DATA_PURGED = "data_purged"
    GDPR_REQUEST = "gdpr_request"
    COMPLIANCE_REPORT_GENERATED = "compliance_report_generated"


class AuditLog(Base):
    """Audit log database model"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    event_type = Column(String(50), nullable=False, index=True)
    user_id = Column(String(100), index=True)
    username = Column(String(100), index=True)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    endpoint = Column(String(200), index=True)
    method = Column(String(10))
    status_code = Column(Integer)
    resource_type = Column(String(50), index=True)
    resource_id = Column(String(100), index=True)
    action = Column(String(50))
    details = Column(Text)  # JSON string
    previous_value = Column(Text)  # JSON string for updates
    new_value = Column(Text)  # JSON string for updates
    success = Column(String(10))  # 'true' or 'false'
    error_message = Column(Text)
    session_id = Column(String(100), index=True)
    request_id = Column(String(100))
    duration_ms = Column(Integer)  # Request duration in milliseconds
    hash_chain = Column(String(64))  # SHA-256 hash for tamper detection
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_user', 'user_id', 'username'),
        Index('idx_audit_event', 'event_type', 'timestamp'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, event_type={self.event_type}, user={self.username}, timestamp={self.timestamp})>"


class AuditLogger:
    """
    Comprehensive audit logging system with tamper-proof hash chain
    """
    
    def __init__(self, db_session=None):
        """Initialize audit logger"""
        self.db_session = db_session
        self.logger = structlog.get_logger()
        self.last_hash = "0" * 64  # Initial hash for chain
    
    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        previous_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        duration_ms: Optional[int] = None
    ) -> Optional[AuditLog]:
        """
        Log an audit event with all relevant information
        
        Args:
            event_type: Type of event (from AuditEventType enum)
            user_id: User identifier
            username: Username
            ip_address: Client IP address
            user_agent: User agent string
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            resource_type: Type of resource (business, asset, etc.)
            resource_id: Resource identifier
            action: Action performed
            details: Additional details as dictionary
            previous_value: Previous value (for updates)
            new_value: New value (for updates)
            success: Whether operation was successful
            error_message: Error message if failed
            session_id: Session identifier
            request_id: Request identifier
            duration_ms: Request duration in milliseconds
            
        Returns:
            AuditLog object if successful, None otherwise
        """
        try:
            # Create audit log entry
            audit_log = AuditLog(
                timestamp=datetime.now(timezone.utc),
                event_type=event_type.value if isinstance(event_type, AuditEventType) else event_type,
                user_id=user_id,
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                details=json.dumps(details) if details else None,
                previous_value=json.dumps(previous_value) if previous_value else None,
                new_value=json.dumps(new_value) if new_value else None,
                success='true' if success else 'false',
                error_message=error_message,
                session_id=session_id,
                request_id=request_id,
                duration_ms=duration_ms
            )
            
            # Calculate hash chain for tamper detection
            audit_log.hash_chain = self._calculate_hash(audit_log)
            
            # Save to database if session available
            if self.db_session:
                self.db_session.add(audit_log)
                self.db_session.commit()
                self.last_hash = audit_log.hash_chain
            
            # Also log to structured logger
            self.logger.info(
                "audit_event",
                event_type=event_type.value if isinstance(event_type, AuditEventType) else event_type,
                user_id=user_id,
                username=username,
                resource_type=resource_type,
                resource_id=resource_id,
                success=success
            )
            
            return audit_log
            
        except Exception as e:
            self.logger.error("Failed to log audit event", error=str(e), event_type=event_type)
            telemetry_logger.log_error(e, {'context': 'audit_logging'})
            return None
    
    def _calculate_hash(self, audit_log: AuditLog) -> str:
        """
        Calculate SHA-256 hash for tamper detection
        Creates a hash chain by including the previous hash
        """
        # Combine all relevant fields
        data = f"{self.last_hash}|{audit_log.timestamp}|{audit_log.event_type}|{audit_log.user_id}|{audit_log.endpoint}|{audit_log.details}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def verify_integrity(self, start_id: Optional[int] = None, end_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Verify the integrity of audit logs by checking hash chain
        
        Args:
            start_id: Starting audit log ID (optional)
            end_id: Ending audit log ID (optional)
            
        Returns:
            Dictionary with verification results
        """
        if not self.db_session:
            return {'error': 'No database session available'}
        
        try:
            query = self.db_session.query(AuditLog).order_by(AuditLog.id)
            
            if start_id:
                query = query.filter(AuditLog.id >= start_id)
            if end_id:
                query = query.filter(AuditLog.id <= end_id)
            
            logs = query.all()
            
            if not logs:
                return {'verified': True, 'message': 'No logs to verify'}
            
            # Verify hash chain
            previous_hash = "0" * 64
            tampered_logs = []
            
            for log in logs:
                # Recalculate hash
                data = f"{previous_hash}|{log.timestamp}|{log.event_type}|{log.user_id}|{log.endpoint}|{log.details}"
                expected_hash = hashlib.sha256(data.encode('utf-8')).hexdigest()
                
                if log.hash_chain != expected_hash:
                    tampered_logs.append({
                        'id': log.id,
                        'timestamp': log.timestamp.isoformat(),
                        'expected_hash': expected_hash,
                        'actual_hash': log.hash_chain
                    })
                
                previous_hash = log.hash_chain
            
            if tampered_logs:
                return {
                    'verified': False,
                    'message': f'Found {len(tampered_logs)} tampered logs',
                    'tampered_logs': tampered_logs
                }
            
            return {
                'verified': True,
                'message': f'All {len(logs)} logs verified successfully',
                'logs_checked': len(logs)
            }
            
        except Exception as e:
            self.logger.error("Failed to verify audit log integrity", error=str(e))
            return {'error': str(e)}
    
    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit trail with filters
        
        Args:
            user_id: Filter by user ID
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            event_type: Filter by event type
            start_date: Start date for filtering
            end_date: End date for filtering
            limit: Maximum number of records to return
            
        Returns:
            List of audit log dictionaries
        """
        if not self.db_session:
            return []
        
        try:
            query = self.db_session.query(AuditLog).order_by(AuditLog.timestamp.desc())
            
            if user_id:
                query = query.filter(AuditLog.user_id == user_id)
            if resource_type:
                query = query.filter(AuditLog.resource_type == resource_type)
            if resource_id:
                query = query.filter(AuditLog.resource_id == resource_id)
            if event_type:
                query = query.filter(AuditLog.event_type == event_type.value)
            if start_date:
                query = query.filter(AuditLog.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditLog.timestamp <= end_date)
            
            logs = query.limit(limit).all()
            
            return [self._log_to_dict(log) for log in logs]
            
        except Exception as e:
            self.logger.error("Failed to retrieve audit trail", error=str(e))
            return []
    
    def _log_to_dict(self, log: AuditLog) -> Dict[str, Any]:
        """Convert audit log to dictionary"""
        return {
            'id': log.id,
            'timestamp': log.timestamp.isoformat(),
            'event_type': log.event_type,
            'user_id': log.user_id,
            'username': log.username,
            'ip_address': log.ip_address,
            'endpoint': log.endpoint,
            'method': log.method,
            'status_code': log.status_code,
            'resource_type': log.resource_type,
            'resource_id': log.resource_id,
            'action': log.action,
            'details': json.loads(log.details) if log.details else None,
            'previous_value': json.loads(log.previous_value) if log.previous_value else None,
            'new_value': json.loads(log.new_value) if log.new_value else None,
            'success': log.success == 'true',
            'error_message': log.error_message,
            'session_id': log.session_id,
            'request_id': log.request_id,
            'duration_ms': log.duration_ms
        }
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = 'summary'
    ) -> Dict[str, Any]:
        """
        Generate compliance report for audit logs
        
        Args:
            start_date: Report start date
            end_date: Report end date
            report_type: Type of report ('summary', 'detailed', 'security')
            
        Returns:
            Dictionary containing report data
        """
        if not self.db_session:
            return {'error': 'No database session available'}
        
        try:
            query = self.db_session.query(AuditLog).filter(
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date
            )
            
            logs = query.all()
            
            # Calculate statistics
            total_events = len(logs)
            successful_events = sum(1 for log in logs if log.success == 'true')
            failed_events = total_events - successful_events
            
            # Count by event type
            event_counts = {}
            for log in logs:
                event_counts[log.event_type] = event_counts.get(log.event_type, 0) + 1
            
            # Count by user
            user_counts = {}
            for log in logs:
                if log.username:
                    user_counts[log.username] = user_counts.get(log.username, 0) + 1
            
            # Security events
            security_events = [log for log in logs if log.event_type in [
                AuditEventType.LOGIN_FAILURE.value,
                AuditEventType.ACCESS_DENIED.value,
                AuditEventType.RATE_LIMIT_EXCEEDED.value,
                AuditEventType.SUSPICIOUS_ACTIVITY.value,
                AuditEventType.SECURITY_VIOLATION.value
            ]]
            
            report = {
                'report_type': report_type,
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'summary': {
                    'total_events': total_events,
                    'successful_events': successful_events,
                    'failed_events': failed_events,
                    'success_rate': (successful_events / total_events * 100) if total_events > 0 else 0
                },
                'event_breakdown': event_counts,
                'top_users': dict(sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'security_events': {
                    'count': len(security_events),
                    'events': [self._log_to_dict(log) for log in security_events[:20]]
                },
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
            if report_type == 'detailed':
                report['all_events'] = [self._log_to_dict(log) for log in logs]
            
            return report
            
        except Exception as e:
            self.logger.error("Failed to generate compliance report", error=str(e))
            return {'error': str(e)}


# Global audit logger instance
audit_logger = AuditLogger()


def init_audit_logger(db_session):
    """Initialize audit logger with database session"""
    global audit_logger
    audit_logger = AuditLogger(db_session)
    return audit_logger


## Implementing security enhancements  
- Starting Phase 1 by focusing on enhanced security headers and CORS policies.  
- Enhancing Flask-Talisman configuration to improve application security.  
- Updating CORS policies in the main application to ensure proper access controls.  
- Editing the app file to integrate these security improvements.

## Configuring security headers  
- Updating Talisman configuration with a detailed content security policy (CSP).  
- CSP settings include restrictions for default, script, style, image, font, and connection sources, all defaulting to "'self'".

Assistant: Now let me enhance the CORS configuration to be more restrictive and secure:

<edit_file>
<path>../../app_final.py</path>
<content>
<<<<<<< SEARCH
CORS(app)
=======
# Enhanced CORS configuration for security
cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000').split(',')

# In production, restrict origins to known domains only
if os.environ.get('FLASK_ENV') == 'production':
    # Add your production domains here
    cors_origins = [
        'https://yourdomain.com',
        'https://api.yourdomain.com',
        'https://app.yourdomain.com'
    ]

CORS(app,
    origins=[origin.strip() for origin in cors_origins],
    methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'],
    allow_headers=['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin'],
    expose_headers=['Content-Length', 'X-Kuma-Revision'],
    supports_credentials=True,
    max_age=86400  # 24 hours
)
