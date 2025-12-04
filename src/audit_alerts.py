"""
Audit Alerts Module
Real-time security alerting and monitoring
"""
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Callable
from collections import defaultdict
from enum import Enum

try:
    from src.models.audit_log import AuditLogModel
    from src.database_fixed import DatabaseManager
    from src.logger import telemetry_logger
except ImportError:
    pass


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert types"""
    FAILED_LOGIN = "failed_login"
    BRUTE_FORCE = "brute_force"
    UNUSUAL_ACTIVITY = "unusual_activity"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_IP = "suspicious_ip"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class AuditAlert:
    """Audit alert data structure"""
    
    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        description: str,
        affected_user: Optional[str] = None,
        affected_resource: Optional[str] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.alert_id = f"ALERT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        self.alert_type = alert_type
        self.severity = severity
        self.title = title
        self.description = description
        self.affected_user = affected_user
        self.affected_resource = affected_resource
        self.ip_address = ip_address
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
        self.acknowledged = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'affected_user': self.affected_user,
            'affected_resource': self.affected_resource,
            'ip_address': self.ip_address,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }


class AlertRule:
    """Alert rule configuration"""
    
    def __init__(
        self,
        rule_id: str,
        name: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        condition: Callable,
        enabled: bool = True
    ):
        self.rule_id = rule_id
        self.name = name
        self.alert_type = alert_type
        self.severity = severity
        self.condition = condition
        self.enabled = enabled


class AuditAlertManager:
    """
    Manage security alerts and real-time monitoring
    
    Features:
    - Real-time alert generation
    - Configurable alert rules
    - Alert notification system
    - Alert acknowledgment and tracking
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize alert manager
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.logger = telemetry_logger.get_logger()
        self.alert_rules = []
        self.active_alerts = []
        self.alert_handlers = []
        
        # Initialize default alert rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        # Failed login attempts
        self.add_alert_rule(AlertRule(
            rule_id='failed_login_threshold',
            name='Multiple Failed Login Attempts',
            alert_type=AlertType.FAILED_LOGIN,
            severity=AlertSeverity.MEDIUM,
            condition=lambda logs: len([l for l in logs if l.action == 'authentication_attempt' and l.status_code == 401]) >= 5
        ))
        
        # Brute force detection
        self.add_alert_rule(AlertRule(
            rule_id='brute_force_detection',
            name='Brute Force Attack Detected',
            alert_type=AlertType.BRUTE_FORCE,
            severity=AlertSeverity.HIGH,
            condition=lambda logs: len([l for l in logs if l.action == 'authentication_attempt' and l.status_code == 401]) >= 10
        ))
        
        # Rate limit exceeded
        self.add_alert_rule(AlertRule(
            rule_id='rate_limit_exceeded',
            name='Rate Limit Exceeded',
            alert_type=AlertType.RATE_LIMIT_EXCEEDED,
            severity=AlertSeverity.MEDIUM,
            condition=lambda logs: len(logs) >= 100
        ))
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules.append(rule)
        self.logger.info(f"Alert rule added: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule"""
        self.alert_rules = [r for r in self.alert_rules if r.rule_id != rule_id]
        self.logger.info(f"Alert rule removed: {rule_id}")
    
    def add_alert_handler(self, handler: Callable[[AuditAlert], None]):
        """
        Add an alert handler function
        
        Args:
            handler: Function that takes an AuditAlert and handles it
        """
        self.alert_handlers.append(handler)
    
    def check_failed_login_attempts(
        self,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        lookback_minutes: int = 15,
        threshold: int = 5
    ) -> Optional[AuditAlert]:
        """
        Check for multiple failed login attempts
        
        Args:
            username: Filter by username
            ip_address: Filter by IP address
            lookback_minutes: Time window to check
            threshold: Number of failed attempts to trigger alert
            
        Returns:
            AuditAlert if threshold exceeded, None otherwise
        """
        try:
            start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
            
            with self.db_manager.get_session() as session:
                query = session.query(AuditLogModel).filter(
                    AuditLogModel.action == 'authentication_attempt',
                    AuditLogModel.status_code == 401,
                    AuditLogModel.timestamp >= start_time
                )
                
                if username:
                    query = query.filter(AuditLogModel.username == username)
                if ip_address:
                    query = query.filter(AuditLogModel.ip_address == ip_address)
                
                failed_attempts = query.count()
                
                if failed_attempts >= threshold:
                    alert = AuditAlert(
                        alert_type=AlertType.FAILED_LOGIN if failed_attempts < 10 else AlertType.BRUTE_FORCE,
                        severity=AlertSeverity.MEDIUM if failed_attempts < 10 else AlertSeverity.HIGH,
                        title=f"Multiple Failed Login Attempts Detected",
                        description=f"{failed_attempts} failed login attempts in the last {lookback_minutes} minutes",
                        affected_user=username,
                        ip_address=ip_address,
                        metadata={
                            'failed_attempts': failed_attempts,
                            'time_window_minutes': lookback_minutes,
                            'threshold': threshold
                        }
                    )
                    
                    self._trigger_alert(alert)
                    return alert
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to check failed login attempts: {e}")
            return None
    
    def check_unusual_activity(
        self,
        user_id: Optional[str] = None,
        lookback_hours: int = 24
    ) -> List[AuditAlert]:
        """
        Check for unusual user activity patterns
        
        Args:
            user_id: Filter by user ID
            lookback_hours: Time window to check
            
        Returns:
            List of alerts for unusual activities
        """
        alerts = []
        
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
            
            with self.db_manager.get_session() as session:
                query = session.query(AuditLogModel).filter(
                    AuditLogModel.timestamp >= start_time
                )
                
                if user_id:
                    query = query.filter(AuditLogModel.user_id == user_id)
                
                logs = query.all()
                
                # Check for unusual access times
                night_access = [log for log in logs if log.timestamp and log.timestamp.hour < 6]
                if len(night_access) > 10:
                    alerts.append(AuditAlert(
                        alert_type=AlertType.UNUSUAL_ACTIVITY,
                        severity=AlertSeverity.MEDIUM,
                        title="Unusual Access Time Detected",
                        description=f"User accessed system {len(night_access)} times during night hours (12 AM - 6 AM)",
                        affected_user=user_id,
                        metadata={'night_access_count': len(night_access)}
                    ))
                
                # Check for rapid resource access
                resource_access = defaultdict(int)
                for log in logs:
                    if log.resource_id:
                        resource_access[log.resource_id] += 1
                
                for resource_id, count in resource_access.items():
                    if count > 100:
                        alerts.append(AuditAlert(
                            alert_type=AlertType.UNUSUAL_ACTIVITY,
                            severity=AlertSeverity.MEDIUM,
                            title="Excessive Resource Access",
                            description=f"Resource {resource_id} accessed {count} times in {lookback_hours} hours",
                            affected_user=user_id,
                            affected_resource=resource_id,
                            metadata={'access_count': count}
                        ))
                
                # Trigger all alerts
                for alert in alerts:
                    self._trigger_alert(alert)
            
            return alerts
        except Exception as e:
            self.logger.error(f"Failed to check unusual activity: {e}")
            return []
    
    def check_suspicious_ip(
        self,
        ip_address: str,
        lookback_hours: int = 1
    ) -> Optional[AuditAlert]:
        """
        Check for suspicious IP activity
        
        Args:
            ip_address: IP address to check
            lookback_hours: Time window to check
            
        Returns:
            AuditAlert if suspicious activity detected
        """
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
            
            with self.db_manager.get_session() as session:
                logs = session.query(AuditLogModel).filter(
                    AuditLogModel.ip_address == ip_address,
                    AuditLogModel.timestamp >= start_time
                ).all()
                
                # Check for multiple users from same IP
                unique_users = set(log.username for log in logs if log.username)
                if len(unique_users) > 5:
                    alert = AuditAlert(
                        alert_type=AlertType.SUSPICIOUS_IP,
                        severity=AlertSeverity.HIGH,
                        title="Suspicious IP Activity",
                        description=f"IP {ip_address} accessed {len(unique_users)} different user accounts",
                        ip_address=ip_address,
                        metadata={
                            'unique_users': len(unique_users),
                            'total_requests': len(logs)
                        }
                    )
                    
                    self._trigger_alert(alert)
                    return alert
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to check suspicious IP: {e}")
            return None
    
    def check_unauthorized_access(
        self,
        resource_type: str,
        resource_id: str,
        lookback_minutes: int = 5
    ) -> Optional[AuditAlert]:
        """
        Check for unauthorized access attempts
        
        Args:
            resource_type: Type of resource
            resource_id: Resource ID
            lookback_minutes: Time window to check
            
        Returns:
            AuditAlert if unauthorized access detected
        """
        try:
            start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
            
            with self.db_manager.get_session() as session:
                failed_access = session.query(AuditLogModel).filter(
                    AuditLogModel.resource_type == resource_type,
                    AuditLogModel.resource_id == resource_id,
                    AuditLogModel.status_code == 403,
                    AuditLogModel.timestamp >= start_time
                ).count()
                
                if failed_access > 0:
                    alert = AuditAlert(
                        alert_type=AlertType.UNAUTHORIZED_ACCESS,
                        severity=AlertSeverity.HIGH,
                        title="Unauthorized Access Attempt",
                        description=f"{failed_access} unauthorized access attempts to {resource_type}:{resource_id}",
                        affected_resource=f"{resource_type}:{resource_id}",
                        metadata={'failed_attempts': failed_access}
                    )
                    
                    self._trigger_alert(alert)
                    return alert
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to check unauthorized access: {e}")
            return None
    
    def monitor_real_time(self, interval_seconds: int = 60):
        """
        Start real-time monitoring (runs continuously)
        
        Args:
            interval_seconds: Check interval in seconds
        """
        import time
        
        self.logger.info(f"Starting real-time audit monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                # Check for failed login attempts
                self.check_failed_login_attempts(lookback_minutes=15, threshold=5)
                
                # Check for unusual activity
                self.check_unusual_activity(lookback_hours=1)
                
                # Sleep until next check
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                self.logger.info("Real-time monitoring stopped")
                break
            except Exception as e:
                self.logger.error(f"Error in real-time monitoring: {e}")
                time.sleep(interval_seconds)
    
    def _trigger_alert(self, alert: AuditAlert):
        """
        Trigger an alert and notify handlers
        
        Args:
            alert: AuditAlert to trigger
        """
        # Add to active alerts
        self.active_alerts.append(alert)
        
        # Log the alert
        self.logger.warning(f"SECURITY ALERT: {alert.title} - {alert.description}")
        
        # Notify all handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None,
        acknowledged: Optional[bool] = None
    ) -> List[AuditAlert]:
        """
        Get active alerts with filters
        
        Args:
            severity: Filter by severity
            alert_type: Filter by alert type
            acknowledged: Filter by acknowledgment status
            
        Returns:
            List of matching alerts
        """
        alerts = self.active_alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert ID to acknowledge
            
        Returns:
            True if acknowledged, False if not found
        """
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Alert acknowledged: {alert_id}")
                return True
        
        return False
    
    def clear_acknowledged_alerts(self):
        """Clear all acknowledged alerts"""
        before_count = len(self.active_alerts)
        self.active_alerts = [a for a in self.active_alerts if not a.acknowledged]
        cleared_count = before_count - len(self.active_alerts)
        self.logger.info(f"Cleared {cleared_count} acknowledged alerts")
    
    def send_security_alert(
        self,
        alert: AuditAlert,
        notification_method: str = 'log'
    ):
        """
        Send security alert notification
        
        Args:
            alert: Alert to send
            notification_method: Notification method (log, email, slack, etc.)
        """
        if notification_method == 'log':
            self.logger.critical(f"SECURITY ALERT: {json.dumps(alert.to_dict(), indent=2)}")
        elif notification_method == 'email':
            # TODO: Implement email notification
            self.logger.info(f"Email notification would be sent for alert: {alert.alert_id}")
        elif notification_method == 'slack':
            # TODO: Implement Slack notification
            self.logger.info(f"Slack notification would be sent for alert: {alert.alert_id}")
        else:
            self.logger.warning(f"Unknown notification method: {notification_method}")
    
    def configure_alert_rules(self, rules_config: Dict[str, Any]):
        """
        Configure alert rules from configuration
        
        Args:
            rules_config: Alert rules configuration dictionary
        """
        # Clear existing rules
        self.alert_rules = []
        
        # Add rules from configuration
        for rule_id, rule_data in rules_config.items():
            # This would need to be implemented based on configuration format
            self.logger.info(f"Configured alert rule: {rule_id}")
