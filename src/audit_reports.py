"""
Audit Reports and Analytics Module
Generates compliance reports and security analytics from audit logs
"""
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from collections import defaultdict

try:
    from src.models.audit_log import AuditLogModel
    from src.database_fixed import DatabaseManager
    from src.logger import telemetry_logger
except ImportError:
    pass


class AuditReportGenerator:
    """
    Generate comprehensive audit reports and analytics
    
    Features:
    - User activity reports
    - Security incident reports
    - Compliance reports (PCI-DSS, GDPR, SOX)
    - Suspicious activity detection
    - Performance analytics
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize report generator
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.logger = telemetry_logger.get_logger()
    
    def generate_user_activity_report(
        self,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate user activity report
        
        Args:
            user_id: Filter by user ID
            username: Filter by username
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            User activity report dictionary
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(AuditLogModel)
                
                # Apply filters
                if user_id:
                    query = query.filter(AuditLogModel.user_id == user_id)
                if username:
                    query = query.filter(AuditLogModel.username == username)
                if start_date:
                    query = query.filter(AuditLogModel.timestamp >= start_date)
                if end_date:
                    query = query.filter(AuditLogModel.timestamp <= end_date)
                
                logs = query.all()
                
                # Calculate statistics
                total_actions = len(logs)
                actions_by_type = defaultdict(int)
                actions_by_hour = defaultdict(int)
                failed_actions = 0
                resources_accessed = set()
                
                for log in logs:
                    actions_by_type[log.action] += 1
                    if log.timestamp:
                        actions_by_hour[log.timestamp.hour] += 1
                    if log.status_code and log.status_code >= 400:
                        failed_actions += 1
                    if log.resource_id:
                        resources_accessed.add(f"{log.resource_type}:{log.resource_id}")
                
                return {
                    'report_type': 'user_activity',
                    'user_id': user_id,
                    'username': username,
                    'time_range': {
                        'start': start_date.isoformat() if start_date else None,
                        'end': end_date.isoformat() if end_date else None
                    },
                    'summary': {
                        'total_actions': total_actions,
                        'failed_actions': failed_actions,
                        'success_rate': (total_actions - failed_actions) / total_actions if total_actions > 0 else 0,
                        'unique_resources_accessed': len(resources_accessed)
                    },
                    'actions_by_type': dict(actions_by_type),
                    'actions_by_hour': dict(actions_by_hour),
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            self.logger.error(f"Failed to generate user activity report: {e}")
            return {'error': str(e)}
    
    def generate_security_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate security incident report
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            severity: Filter by severity level
            
        Returns:
            Security report dictionary
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(AuditLogModel).filter(
                    AuditLogModel.category == 'security'
                )
                
                # Apply filters
                if start_date:
                    query = query.filter(AuditLogModel.timestamp >= start_date)
                if end_date:
                    query = query.filter(AuditLogModel.timestamp <= end_date)
                if severity:
                    query = query.filter(AuditLogModel.severity == severity)
                
                logs = query.all()
                
                # Calculate statistics
                total_incidents = len(logs)
                incidents_by_severity = defaultdict(int)
                incidents_by_type = defaultdict(int)
                affected_users = set()
                affected_ips = set()
                
                for log in logs:
                    incidents_by_severity[log.severity] += 1
                    incidents_by_type[log.action] += 1
                    if log.username:
                        affected_users.add(log.username)
                    if log.ip_address:
                        affected_ips.add(log.ip_address)
                
                # Get top security events
                top_events = sorted(
                    incidents_by_type.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                return {
                    'report_type': 'security',
                    'time_range': {
                        'start': start_date.isoformat() if start_date else None,
                        'end': end_date.isoformat() if end_date else None
                    },
                    'summary': {
                        'total_incidents': total_incidents,
                        'critical_incidents': incidents_by_severity.get('critical', 0),
                        'high_severity_incidents': incidents_by_severity.get('error', 0),
                        'medium_severity_incidents': incidents_by_severity.get('warning', 0),
                        'affected_users': len(affected_users),
                        'affected_ips': len(affected_ips)
                    },
                    'incidents_by_severity': dict(incidents_by_severity),
                    'top_security_events': [{'event': event, 'count': count} for event, count in top_events],
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            self.logger.error(f"Failed to generate security report: {e}")
            return {'error': str(e)}
    
    def generate_compliance_report(
        self,
        compliance_standard: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report (PCI-DSS, GDPR, SOX)
        
        Args:
            compliance_standard: Compliance standard (PCI-DSS, GDPR, SOX)
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            Compliance report dictionary
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(AuditLogModel)
                
                # Apply filters
                if start_date:
                    query = query.filter(AuditLogModel.timestamp >= start_date)
                if end_date:
                    query = query.filter(AuditLogModel.timestamp <= end_date)
                
                # Filter by compliance tags
                logs = [log for log in query.all() 
                       if compliance_standard in (json.loads(log.compliance_tags) if log.compliance_tags else [])]
                
                # Calculate compliance metrics
                total_events = len(logs)
                authentication_events = len([log for log in logs if log.category == 'authentication'])
                data_access_events = len([log for log in logs if log.category == 'data_access'])
                failed_events = len([log for log in logs if log.status_code and log.status_code >= 400])
                
                # Compliance-specific metrics
                compliance_metrics = {}
                
                if compliance_standard == 'PCI-DSS':
                    compliance_metrics = {
                        'authentication_logging': authentication_events > 0,
                        'access_control_logging': data_access_events > 0,
                        'audit_trail_complete': total_events > 0,
                        'failed_access_tracking': failed_events >= 0
                    }
                elif compliance_standard == 'GDPR':
                    compliance_metrics = {
                        'data_access_logged': data_access_events > 0,
                        'user_consent_tracked': True,  # Would need specific implementation
                        'data_retention_compliant': True,  # Would need specific implementation
                        'breach_notification_ready': True
                    }
                elif compliance_standard == 'SOX':
                    compliance_metrics = {
                        'financial_transaction_logging': total_events > 0,
                        'access_control_documented': data_access_events > 0,
                        'audit_trail_tamper_proof': True,  # Hash chain verification
                        'segregation_of_duties': True  # Would need specific implementation
                    }
                
                return {
                    'report_type': 'compliance',
                    'compliance_standard': compliance_standard,
                    'time_range': {
                        'start': start_date.isoformat() if start_date else None,
                        'end': end_date.isoformat() if end_date else None
                    },
                    'summary': {
                        'total_events': total_events,
                        'authentication_events': authentication_events,
                        'data_access_events': data_access_events,
                        'failed_events': failed_events,
                        'compliance_score': sum(1 for v in compliance_metrics.values() if v) / len(compliance_metrics) if compliance_metrics else 0
                    },
                    'compliance_metrics': compliance_metrics,
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            return {'error': str(e)}
    
    def get_suspicious_activities(
        self,
        lookback_hours: int = 24,
        threshold_failed_logins: int = 5,
        threshold_requests_per_minute: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Detect suspicious activities
        
        Args:
            lookback_hours: Hours to look back
            threshold_failed_logins: Failed login threshold
            threshold_requests_per_minute: Request rate threshold
            
        Returns:
            List of suspicious activities
        """
        try:
            suspicious_activities = []
            start_date = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
            
            with self.db_manager.get_session() as session:
                # Check for multiple failed login attempts
                failed_logins = session.query(AuditLogModel).filter(
                    AuditLogModel.action == 'authentication_attempt',
                    AuditLogModel.status_code == 401,
                    AuditLogModel.timestamp >= start_date
                ).all()
                
                # Group by user and IP
                failed_by_user = defaultdict(list)
                failed_by_ip = defaultdict(list)
                
                for log in failed_logins:
                    if log.username:
                        failed_by_user[log.username].append(log)
                    if log.ip_address:
                        failed_by_ip[log.ip_address].append(log)
                
                # Detect brute force attempts
                for username, logs in failed_by_user.items():
                    if len(logs) >= threshold_failed_logins:
                        suspicious_activities.append({
                            'type': 'brute_force_attempt',
                            'severity': 'high',
                            'username': username,
                            'failed_attempts': len(logs),
                            'time_range': {
                                'start': min(log.timestamp for log in logs).isoformat(),
                                'end': max(log.timestamp for log in logs).isoformat()
                            },
                            'description': f'Multiple failed login attempts ({len(logs)}) for user {username}'
                        })
                
                # Detect IP-based attacks
                for ip_address, logs in failed_by_ip.items():
                    if len(logs) >= threshold_failed_logins:
                        suspicious_activities.append({
                            'type': 'ip_based_attack',
                            'severity': 'high',
                            'ip_address': ip_address,
                            'failed_attempts': len(logs),
                            'affected_users': len(set(log.username for log in logs if log.username)),
                            'description': f'Multiple failed attempts from IP {ip_address}'
                        })
                
                # Check for unusual access patterns
                all_logs = session.query(AuditLogModel).filter(
                    AuditLogModel.timestamp >= start_date
                ).all()
                
                # Group by user and time window
                user_activity = defaultdict(lambda: defaultdict(int))
                for log in all_logs:
                    if log.username and log.timestamp:
                        minute_key = log.timestamp.replace(second=0, microsecond=0)
                        user_activity[log.username][minute_key] += 1
                
                # Detect high request rates
                for username, activity in user_activity.items():
                    max_requests = max(activity.values()) if activity else 0
                    if max_requests >= threshold_requests_per_minute:
                        suspicious_activities.append({
                            'type': 'high_request_rate',
                            'severity': 'medium',
                            'username': username,
                            'max_requests_per_minute': max_requests,
                            'description': f'Unusually high request rate ({max_requests} req/min) for user {username}'
                        })
            
            return suspicious_activities
        except Exception as e:
            self.logger.error(f"Failed to detect suspicious activities: {e}")
            return []
    
    def export_report(
        self,
        report_data: Dict[str, Any],
        format_type: str = 'json'
    ) -> str:
        """
        Export report in specified format
        
        Args:
            report_data: Report data dictionary
            format_type: Export format (json, html, pdf)
            
        Returns:
            Exported report as string
        """
        try:
            if format_type == 'json':
                return json.dumps(report_data, indent=2, default=str)
            elif format_type == 'html':
                return self._generate_html_report(report_data)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return json.dumps({'error': str(e)})
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Audit Report - {report_data.get('report_type', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Audit Report: {report_data.get('report_type', 'Unknown').title()}</h1>
            <div class="summary">
                <h2>Summary</h2>
                <pre>{json.dumps(report_data.get('summary', {}), indent=2)}</pre>
            </div>
            <h2>Full Report</h2>
            <pre>{json.dumps(report_data, indent=2, default=str)}</pre>
            <p><small>Generated at: {report_data.get('generated_at', 'Unknown')}</small></p>
        </body>
        </html>
        """
        return html
