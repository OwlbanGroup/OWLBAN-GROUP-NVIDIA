"""
Core Audit Logger Module
Provides comprehensive audit logging with tamper-proof hash chain
"""
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from flask import request, g
import hashlib

try:
    from src.models.audit_log import AuditLogModel, AuditLogSummary
    from src.database_fixed import DatabaseManager
    from src.logger import telemetry_logger
except ImportError:
    # Fallback for testing
    pass


class AuditLogger:
    """
    Comprehensive audit logging system with tamper-proof hash chain
    
    Features:
    - Tamper-proof hash chain for log integrity
    - Automatic user context extraction
    - Sensitive data sanitization
    - Real-time security alerting
    - Compliance-ready audit trails
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize audit logger
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.logger = telemetry_logger.get_logger()
        
    def _get_last_hash(self) -> Optional[str]:
        """Get the hash of the last audit log entry"""
        try:
            with self.db_manager.get_session() as session:
                last_log = session.query(AuditLogModel).order_by(
                    AuditLogModel.id.desc()
                ).first()
                return last_log.current_hash if last_log else None
        except Exception as e:
            self.logger.error(f"Failed to get last hash: {e}")
            return None
    
    def _sanitize_data(self, data: Any, max_length: int = 5000) -> str:
        """
        Sanitize data for logging (remove sensitive information)
        
        Args:
            data: Data to sanitize
            max_length: Maximum length of sanitized data
            
        Returns:
            Sanitized JSON string
        """
        if data is None:
            return None
        
        try:
            # Convert to dict if needed
            if not isinstance(data, dict):
                data = {'value': str(data)}
            
            # Create a copy to avoid modifying original
            sanitized = data.copy()
            
            # Remove sensitive fields
            sensitive_fields = [
                'password', 'token', 'secret', 'api_key', 'private_key',
                'credit_card', 'ssn', 'social_security', 'authorization'
            ]
            
            for field in sensitive_fields:
                if field in sanitized:
                    sanitized[field] = '***REDACTED***'
            
            # Convert to JSON and truncate if needed
            json_str = json.dumps(sanitized, default=str)
            if len(json_str) > max_length:
                json_str = json_str[:max_length] + '...[TRUNCATED]'
            
            return json_str
        except Exception as e:
            self.logger.error(f"Failed to sanitize data: {e}")
            return json.dumps({'error': 'Failed to sanitize data'})
    
    def _extract_user_context(self) -> Dict[str, Any]:
        """Extract user context from Flask request"""
        context = {
            'user_id': None,
            'username': None,
            'session_id': None,
            'ip_address': None,
            'user_agent': None
        }
        
        try:
            # Get IP address
            if request:
                context['ip_address'] = request.remote_addr
                context['user_agent'] = request.headers.get('User-Agent', '')
                
                # Try to get user info from Flask g object
                if hasattr(g, 'user_id'):
                    context['user_id'] = g.user_id
                if hasattr(g, 'username'):
                    context['username'] = g.username
                if hasattr(g, 'session_id'):
                    context['session_id'] = g.session_id
        except Exception as e:
            self.logger.warning(f"Failed to extract user context: {e}")
        
        return context
    
    def log_event(
        self,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        status_code: Optional[int] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        severity: str = 'info',
        category: str = 'general',
        compliance_tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        response_time_ms: Optional[int] = None
    ) -> Optional[AuditLogModel]:
        """
        Log an audit event
        
        Args:
            action: Action being performed (e.g., 'login', 'api_call', 'db_update')
            resource_type: Type of resource (e.g., 'user', 'business', 'asset')
            resource_id: ID of the resource
            status_code: HTTP status code
            request_data: Request payload (will be sanitized)
            response_data: Response payload (will be sanitized)
            error_message: Error message if operation failed
            severity: Severity level (info, warning, error, critical)
            category: Category (authentication, authorization, data_access, etc.)
            compliance_tags: List of compliance tags (e.g., ['PCI-DSS', 'GDPR'])
            user_id: User ID (if not in request context)
            username: Username (if not in request context)
            response_time_ms: Response time in milliseconds
            
        Returns:
            Created AuditLogModel instance or None if failed
        """
        try:
            # Extract user context
            context = self._extract_user_context()
            
            # Override with provided values
            if user_id:
                context['user_id'] = user_id
            if username:
                context['username'] = username
            
            # Get request details
            endpoint = None
            request_method = None
            if request:
                endpoint = request.path
                request_method = request.method
            
            # Get previous hash for chain
            previous_hash = self._get_last_hash()
            
            # Prepare log data
            log_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'user_id': context['user_id'],
                'username': context['username'],
                'session_id': context['session_id'],
                'action': action,
                'resource_type': resource_type,
                'resource_id': str(resource_id) if resource_id else None,
                'ip_address': context['ip_address'],
                'user_agent': context['user_agent'],
                'request_method': request_method,
                'endpoint': endpoint,
                'status_code': status_code,
                'response_time_ms': response_time_ms,
                'request_data': self._sanitize_data(request_data),
                'response_data': self._sanitize_data(response_data),
                'error_message': error_message,
                'severity': severity,
                'category': category,
                'compliance_tags': json.dumps(compliance_tags or []),
                'previous_hash': previous_hash
            }
            
            # Calculate current hash
            current_hash = AuditLogModel.calculate_hash(log_data, previous_hash)
            log_data['current_hash'] = current_hash
            
            # Create audit log entry
            with self.db_manager.get_session() as session:
                audit_log = AuditLogModel(**log_data)
                session.add(audit_log)
                session.commit()
                session.refresh(audit_log)
                
                self.logger.info(f"Audit log created: {action} by {context['username'] or 'anonymous'}")
                return audit_log
                
        except Exception as e:
            self.logger.error(f"Failed to create audit log: {e}")
            return None
    
    def log_authentication_attempt(
        self,
        username: str,
        success: bool,
        reason: Optional[str] = None,
        auth_method: str = 'password'
    ) -> Optional[AuditLogModel]:
        """
        Log authentication attempt
        
        Args:
            username: Username attempting authentication
            success: Whether authentication was successful
            reason: Reason for failure (if applicable)
            auth_method: Authentication method used
            
        Returns:
            Created AuditLogModel instance
        """
        return self.log_event(
            action='authentication_attempt',
            resource_type='user',
            resource_id=username,
            status_code=200 if success else 401,
            request_data={'auth_method': auth_method, 'username': username},
            response_data={'success': success},
            error_message=reason if not success else None,
            severity='info' if success else 'warning',
            category='authentication',
            compliance_tags=['PCI-DSS', 'SOX'],
            username=username
        )
    
    def log_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Optional[AuditLogModel]:
        """
        Log API call
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds
            request_data: Request payload
            response_data: Response payload
            error_message: Error message if failed
            
        Returns:
            Created AuditLogModel instance
        """
        severity = 'info'
        if status_code >= 500:
            severity = 'error'
        elif status_code >= 400:
            severity = 'warning'
        
        return self.log_event(
            action='api_call',
            resource_type='endpoint',
            resource_id=endpoint,
            status_code=status_code,
            request_data=request_data,
            response_data=response_data,
            error_message=error_message,
            severity=severity,
            category='api_access',
            compliance_tags=['GDPR'],
            response_time_ms=response_time_ms
        )
    
    def log_database_operation(
        self,
        operation: str,
        table: str,
        record_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Optional[AuditLogModel]:
        """
        Log database operation
        
        Args:
            operation: Operation type (create, read, update, delete)
            table: Database table name
            record_id: Record ID
            data: Data being modified
            success: Whether operation was successful
            error_message: Error message if failed
            
        Returns:
            Created AuditLogModel instance
        """
        return self.log_event(
            action=f'db_{operation}',
            resource_type='database',
            resource_id=f'{table}:{record_id}' if record_id else table,
            status_code=200 if success else 500,
            request_data={'operation': operation, 'table': table, 'data': data},
            error_message=error_message,
            severity='info' if success else 'error',
            category='data_access',
            compliance_tags=['GDPR', 'SOX']
        )
    
    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = 'warning',
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Optional[AuditLogModel]:
        """
        Log security event
        
        Args:
            event_type: Type of security event
            description: Description of the event
            severity: Severity level
            additional_data: Additional event data
            
        Returns:
            Created AuditLogModel instance
        """
        return self.log_event(
            action='security_event',
            resource_type='security',
            resource_id=event_type,
            request_data=additional_data,
            error_message=description,
            severity=severity,
            category='security',
            compliance_tags=['PCI-DSS', 'SOX', 'GDPR']
        )
    
    def log_failed_attempt(
        self,
        action: str,
        reason: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None
    ) -> Optional[AuditLogModel]:
        """
        Log failed attempt
        
        Args:
            action: Action that failed
            reason: Reason for failure
            resource_type: Type of resource
            resource_id: Resource ID
            
        Returns:
            Created AuditLogModel instance
        """
        return self.log_event(
            action=f'failed_{action}',
            resource_type=resource_type,
            resource_id=resource_id,
            status_code=403,
            error_message=reason,
            severity='warning',
            category='security',
            compliance_tags=['PCI-DSS']
        )
    
    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLogModel]:
        """
        Get audit trail with filters
        
        Args:
            user_id: Filter by user ID
            action: Filter by action
            resource_type: Filter by resource type
            severity: Filter by severity
            start_date: Start date for time range
            end_date: End date for time range
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            List of AuditLogModel instances
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(AuditLogModel)
                
                # Apply filters
                if user_id:
                    query = query.filter(AuditLogModel.user_id == user_id)
                if action:
                    query = query.filter(AuditLogModel.action == action)
                if resource_type:
                    query = query.filter(AuditLogModel.resource_type == resource_type)
                if severity:
                    query = query.filter(AuditLogModel.severity == severity)
                if start_date:
                    query = query.filter(AuditLogModel.timestamp >= start_date)
                if end_date:
                    query = query.filter(AuditLogModel.timestamp <= end_date)
                
                # Order by timestamp descending
                query = query.order_by(AuditLogModel.timestamp.desc())
                
                # Apply pagination
                query = query.limit(limit).offset(offset)
                
                return query.all()
        except Exception as e:
            self.logger.error(f"Failed to get audit trail: {e}")
            return []
    
    def get_audit_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> AuditLogSummary:
        """
        Get audit log summary statistics
        
        Args:
            start_date: Start date for time range
            end_date: End date for time range
            
        Returns:
            AuditLogSummary instance
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(AuditLogModel)
                
                # Apply date filters
                if start_date:
                    query = query.filter(AuditLogModel.timestamp >= start_date)
                if end_date:
                    query = query.filter(AuditLogModel.timestamp <= end_date)
                
                logs = query.all()
                
                # Calculate statistics
                total_logs = len(logs)
                by_action = {}
                by_severity = {}
                by_user = {}
                failed_attempts = 0
                
                for log in logs:
                    # Count by action
                    by_action[log.action] = by_action.get(log.action, 0) + 1
                    
                    # Count by severity
                    by_severity[log.severity] = by_severity.get(log.severity, 0) + 1
                    
                    # Count by user
                    if log.username:
                        by_user[log.username] = by_user.get(log.username, 0) + 1
                    
                    # Count failed attempts
                    if log.status_code and log.status_code >= 400:
                        failed_attempts += 1
                
                # Get time range
                time_range = (
                    min(log.timestamp for log in logs) if logs else None,
                    max(log.timestamp for log in logs) if logs else None
                )
                
                return AuditLogSummary(
                    total_logs=total_logs,
                    by_action=by_action,
                    by_severity=by_severity,
                    by_user=by_user,
                    failed_attempts=failed_attempts,
                    time_range=time_range
                )
        except Exception as e:
            self.logger.error(f"Failed to get audit summary: {e}")
            return AuditLogSummary(0, {}, {}, {}, 0, (None, None))
    
    def verify_integrity(
        self,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None
    ):
        """
        Verify audit log chain integrity
        
        Args:
            start_id: Start log ID (optional)
            end_id: End log ID (optional)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(AuditLogModel).order_by(AuditLogModel.id)
                
                if start_id:
                    query = query.filter(AuditLogModel.id >= start_id)
                if end_id:
                    query = query.filter(AuditLogModel.id <= end_id)
                
                logs = query.all()
                return AuditLogModel.verify_chain_integrity(logs)
        except Exception as e:
            self.logger.error(f"Failed to verify integrity: {e}")
            return False, f"Verification failed: {str(e)}"
    
    def export_audit_logs(
        self,
        format_type: str = 'json',
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export audit logs in specified format
        
        Args:
            format_type: Export format (json, csv)
            filters: Filters to apply
            
        Returns:
            Exported data as string
        """
        filters = filters or {}
        logs = self.get_audit_trail(**filters)
        
        if format_type == 'json':
            return json.dumps([log.to_dict() for log in logs], indent=2, default=str)
        elif format_type == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            if logs:
                fieldnames = logs[0].to_dict().keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for log in logs:
                    writer.writerow(log.to_dict())
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format_type}")


# Decorator for automatic audit logging
def audit_log(action: str, resource_type: Optional[str] = None, category: str = 'api_access'):
    """
    Decorator to automatically log API endpoint calls
    
    Usage:
        @app.route('/api/endpoint')
        @audit_log(action='api_call', resource_type='endpoint')
        def my_endpoint():
            ...
    """
    def decorator(f):
        from functools import wraps
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = datetime.now(timezone.utc)
            error_message = None
            status_code = 200
            response_data = None
            
            try:
                # Execute the function
                result = f(*args, **kwargs)
                
                # Extract status code and response data
                if isinstance(result, tuple):
                    response_data = result[0] if len(result) > 0 else None
                    status_code = result[1] if len(result) > 1 else 200
                else:
                    response_data = result
                
                return result
            except Exception as e:
                error_message = str(e)
                status_code = 500
                raise
            finally:
                # Calculate response time
                response_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                
                # Log the event (assuming audit_logger is available in app context)
                try:
                    from flask import current_app
                    if hasattr(current_app, 'audit_logger'):
                        current_app.audit_logger.log_event(
                            action=action,
                            resource_type=resource_type,
                            status_code=status_code,
                            response_time_ms=response_time_ms,
                            error_message=error_message,
                            category=category
                        )
                except Exception as log_error:
                    # Don't fail the request if logging fails
                    print(f"Audit logging failed: {log_error}")
        
        return wrapper
    return decorator
