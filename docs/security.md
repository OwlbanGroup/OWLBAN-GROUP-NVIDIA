# Security Best Practices - JPMorgan Financial APIs

## Overview

This document outlines security best practices for the JPMorgan Financial APIs platform, covering authentication, authorization, data protection, infrastructure security, and compliance requirements.

## Authentication & Authorization

### OAuth2 Implementation

#### Secure Token Management

```python
import secrets
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet

class SecureTokenManager:
    def __init__(self, jwt_secret, encryption_key):
        self.jwt_secret = jwt_secret
        self.cipher = Fernet(encryption_key)

    def generate_access_token(self, client_id, scopes, expires_in=3600):
        """Generate secure JWT access token"""
        now = datetime.utcnow()

        payload = {
            'iss': 'jpmorgan-financial-apis',
            'sub': client_id,
            'iat': now,
            'exp': now + timedelta(seconds=expires_in),
            'scopes': scopes,
            'jti': secrets.token_urlsafe(32),  # Unique token ID
            'type': 'access'
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        return token

    def validate_access_token(self, token):
        """Validate and decode access token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])

            # Check token expiration
            if datetime.utcfromtimestamp(payload['exp']) < datetime.utcnow():
                raise jwt.ExpiredSignatureError("Token has expired")

            # Check token type
            if payload.get('type') != 'access':
                raise jwt.InvalidTokenError("Invalid token type")

            # Additional validation
            self._validate_token_claims(payload)

            return payload

        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

    def _validate_token_claims(self, payload):
        """Additional token validation"""
        required_claims = ['iss', 'sub', 'iat', 'exp', 'scopes', 'jti']

        for claim in required_claims:
            if claim not in payload:
                raise ValueError(f"Missing required claim: {claim}")

        # Validate issuer
        if payload['iss'] != 'jpmorgan-financial-apis':
            raise ValueError("Invalid token issuer")

        # Validate scopes
        if not isinstance(payload['scopes'], list):
            raise ValueError("Invalid scopes format")
```

#### Token Storage Security

```python
import redis
from cryptography.fernet import Fernet

class SecureTokenStorage:
    def __init__(self, redis_client, encryption_key):
        self.redis = redis_client
        self.cipher = Fernet(encryption_key)

    def store_refresh_token(self, client_id, refresh_token, ttl=604800):  # 7 days
        """Securely store encrypted refresh token"""
        encrypted_token = self.cipher.encrypt(refresh_token.encode())

        # Store with client-specific key
        key = f"refresh_token:{client_id}:{secrets.token_urlsafe(16)}"
        self.redis.setex(key, ttl, encrypted_token)

        return key

    def retrieve_refresh_token(self, token_key):
        """Retrieve and decrypt refresh token"""
        encrypted_token = self.redis.get(token_key)

        if not encrypted_token:
            return None

        try:
            decrypted_token = self.cipher.decrypt(encrypted_token).decode()
            return decrypted_token
        except Exception:
            return None

    def revoke_refresh_token(self, token_key):
        """Revoke refresh token"""
        self.redis.delete(token_key)

    def revoke_all_client_tokens(self, client_id):
        """Revoke all refresh tokens for a client"""
        pattern = f"refresh_token:{client_id}:*"
        keys = self.redis.keys(pattern)

        if keys:
            self.redis.delete(*keys)
```

### Multi-Factor Authentication (MFA)

```python
import pyotp
import qrcode
from io import BytesIO

class MFAManager:
    def __init__(self, redis_client):
        self.redis = redis_client

    def generate_mfa_secret(self, user_id):
        """Generate TOTP secret for user"""
        secret = pyotp.random_base32()
        self.redis.setex(f"mfa_secret:{user_id}", 300, secret)  # 5 min expiry
        return secret

    def generate_mfa_qr(self, user_id, username):
        """Generate QR code for MFA setup"""
        secret = self.redis.get(f"mfa_secret:{user_id}")
        if not secret:
            raise ValueError("MFA secret not found or expired")

        totp = pyotp.TOTP(secret.decode())
        provisioning_uri = totp.provisioning_uri(username, issuer_name="JPMorgan APIs")

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def verify_mfa_code(self, user_id, code):
        """Verify MFA code"""
        secret = self.redis.get(f"mfa_secret:{user_id}")
        if not secret:
            return False

        totp = pyotp.TOTP(secret.decode())
        return totp.verify(code)

    def enable_mfa(self, user_id):
        """Enable MFA for user after successful setup"""
        secret = self.redis.get(f"mfa_secret:{user_id}")
        if not secret:
            raise ValueError("MFA setup not completed")

        # Store permanent MFA secret
        self.redis.set(f"mfa_enabled:{user_id}", secret.decode())

        # Clean up temporary secret
        self.redis.delete(f"mfa_secret:{user_id}")

        return True
```

## Data Protection

### Encryption at Rest

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os

class DataEncryption:
    def __init__(self, master_key):
        self.master_key = master_key

    def derive_key(self, salt, info=b"data_encryption"):
        """Derive encryption key from master key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(self.master_key)

    def encrypt_data(self, data):
        """Encrypt sensitive data"""
        if isinstance(data, str):
            data = data.encode()

        salt = os.urandom(16)
        key = self.derive_key(salt)

        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()

        # PKCS7 padding
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length]) * padding_length
        padded_data = data + padding

        encrypted = encryptor.update(padded_data) + encryptor.finalize()

        # Combine salt + iv + encrypted data
        return salt + iv + encrypted

    def decrypt_data(self, encrypted_data):
        """Decrypt sensitive data"""
        salt = encrypted_data[:16]
        iv = encrypted_data[16:32]
        encrypted = encrypted_data[32:]

        key = self.derive_key(salt)

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()

        decrypted_padded = decryptor.update(encrypted) + decryptor.finalize()

        # Remove PKCS7 padding
        padding_length = decrypted_padded[-1]
        decrypted = decrypted_padded[:-padding_length]

        return decrypted.decode()
```

### Data Masking

```python
import re

class DataMasking:
    @staticmethod
    def mask_ssn(ssn):
        """Mask Social Security Number"""
        if not ssn:
            return ssn
        return re.sub(r'(\d{3})-?(\d{2})-?(\d{4})', r'***-**-\1', ssn)

    @staticmethod
    def mask_credit_card(card_number):
        """Mask credit card number"""
        if not card_number:
            return card_number
        # Keep last 4 digits
        return '*' * (len(card_number) - 4) + card_number[-4:]

    @staticmethod
    def mask_email(email):
        """Mask email address"""
        if not email or '@' not in email:
            return email

        username, domain = email.split('@', 1)
        if len(username) <= 2:
            masked_username = username[0] + '*' * (len(username) - 1)
        else:
            masked_username = username[0] + '*' * (len(username) - 2) + username[-1]

        return f"{masked_username}@{domain}"

    @staticmethod
    def mask_phone_number(phone):
        """Mask phone number"""
        if not phone:
            return phone

        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)

        if len(digits) == 10:  # US phone number
            return f"({digits[:3]}) {digits[3:6]}-{ '*' * 4}"
        elif len(digits) == 11:  # US phone with country code
            return f"+{digits[0]} ({digits[1:4]}) {digits[4:7]}-{ '*' * 4}"

        return phone  # Return original if format not recognized
```

## Infrastructure Security

### Network Security

#### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-server-network-policy
  namespace: jpmorgan-apis
spec:
  podSelector:
    matchLabels:
      app: api-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: monitoring
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - podSelector:
        matchLabels:
          app: api-server
    ports:
    - protocol: TCP
      port: 8000  # Allow health checks
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: jpmorgan-external-api
    ports:
    - protocol: TCP
      port: 443
  - to: []  # Deny all other egress
    ports:
    - protocol: TCP
      port: 53  # Allow DNS
```

#### Service Mesh Security

```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: jpmorgan-apis
spec:
  mtls:
    mode: STRICT  # Enforce mutual TLS

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: api-server-authz
  namespace: jpmorgan-apis
spec:
  selector:
    matchLabels:
      app: api-server
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/ingress-nginx/sa/ingress-nginx"]
    to:
    - operation:
        methods: ["GET", "POST", "PUT", "DELETE"]
        paths: ["/api/v1/*"]
  - from:
    - source:
        principals: ["cluster.local/ns/monitoring/sa/prometheus"]
    to:
    - operation:
        methods: ["GET"]
        paths: ["/metrics", "/health"]
```

### Container Security

#### Security Context

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-api-server
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
      - name: api-server
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: tmp-volume
          mountPath: /tmp
        - name: cache-volume
          mountPath: /app/cache
      volumes:
      - name: tmp-volume
        emptyDir: {}
      - name: cache-volume
        emptyDir: {}
```

#### Image Security

```dockerfile
# Use distroless base image for minimal attack surface
FROM gcr.io/distroless/python3-debian11

# Copy application code
COPY --chown=nonroot:nonroot . /app

# Switch to non-root user
USER nonroot

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "/app/app.py"]
```

### Secrets Management

```python
import boto3
from botocore.exceptions import ClientError
import hvac

class SecretsManager:
    def __init__(self, use_aws=True, vault_url=None):
        if use_aws:
            self.client = boto3.client('secretsmanager')
        else:
            self.vault_client = hvac.Client(url=vault_url)

    def get_secret(self, secret_name):
        """Retrieve secret from AWS Secrets Manager or HashiCorp Vault"""
        try:
            if hasattr(self, 'client'):
                # AWS Secrets Manager
                response = self.client.get_secret_value(SecretId=secret_name)
                return response['SecretString']
            else:
                # HashiCorp Vault
                secret = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=secret_name
                )
                return secret['data']['data']
        except ClientError as e:
            raise ValueError(f"Failed to retrieve secret {secret_name}: {e}")

    def rotate_secret(self, secret_name, new_value):
        """Rotate secret value"""
        if hasattr(self, 'client'):
            self.client.update_secret(
                SecretId=secret_name,
                SecretString=new_value
            )
        else:
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=secret_name,
                secret=dict(value=new_value)
            )
```

## Compliance

### GDPR Compliance

```python
from datetime import datetime, timedelta
import hashlib

class GDPRCompliance:
    def __init__(self, db_client):
        self.db = db_client

    def anonymize_user_data(self, user_id):
        """Anonymize user data for GDPR compliance"""
        # Generate anonymous identifier
        anon_id = hashlib.sha256(f"{user_id}{datetime.utcnow()}".encode()).hexdigest()[:16]

        # Update user record
        self.db.users.update_one(
            {'_id': user_id},
            {
                '$set': {
                    'anonymized_at': datetime.utcnow(),
                    'anonymous_id': anon_id,
                    'personal_data_removed': True
                },
                '$unset': {
                    'email': 1,
                    'phone': 1,
                    'address': 1,
                    'ssn': 1
                }
            }
        )

        # Log anonymization
        self._log_gdpr_action(user_id, 'anonymize', anon_id)

        return anon_id

    def handle_data_portability_request(self, user_id):
        """Export user data for portability"""
        user_data = self.db.users.find_one({'_id': user_id})
        transactions = list(self.db.transactions.find({'user_id': user_id}))

        export_data = {
            'user': user_data,
            'transactions': transactions,
            'export_date': datetime.utcnow(),
            'gdpr_compliant': True
        }

        # Log export
        self._log_gdpr_action(user_id, 'export', len(transactions))

        return export_data

    def schedule_data_deletion(self, user_id, retention_days=2555):  # 7 years
        """Schedule user data deletion"""
        deletion_date = datetime.utcnow() + timedelta(days=retention_days)

        self.db.deletion_schedule.insert_one({
            'user_id': user_id,
            'scheduled_deletion': deletion_date,
            'status': 'scheduled',
            'gdpr_request': True
        })

        self._log_gdpr_action(user_id, 'schedule_deletion', str(deletion_date))

    def _log_gdpr_action(self, user_id, action, details):
        """Log GDPR compliance actions"""
        self.db.gdpr_audit.insert_one({
            'user_id': user_id,
            'action': action,
            'details': details,
            'timestamp': datetime.utcnow(),
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent()
        })
```

### SOC 2 Compliance

```python
import logging
from datetime import datetime

class SOC2Compliance:
    def __init__(self, audit_logger):
        self.audit_logger = audit_logger

    def log_security_event(self, event_type, user_id, resource, action, success=True):
        """Log security events for SOC 2 compliance"""
        event = {
            'timestamp': datetime.utcnow(),
            'event_type': event_type,
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'success': success,
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent(),
            'session_id': self._get_session_id()
        }

        self.audit_logger.info("SOC2_SECURITY_EVENT", extra=event)

        # Store in database for compliance reporting
        self._store_audit_event(event)

    def validate_access_control(self, user_id, resource, action):
        """Validate access control for SOC 2"""
        # Check user permissions
        permissions = self._get_user_permissions(user_id)

        # Check resource access rules
        allowed = self._check_resource_access(permissions, resource, action)

        # Log access attempt
        self.log_security_event(
            'ACCESS_ATTEMPT',
            user_id,
            resource,
            action,
            success=allowed
        )

        return allowed

    def generate_compliance_report(self, start_date, end_date):
        """Generate SOC 2 compliance report"""
        events = self._get_audit_events(start_date, end_date)

        report = {
            'period': f"{start_date} to {end_date}",
            'total_events': len(events),
            'security_events': len([e for e in events if e['event_type'] == 'SECURITY']),
            'access_attempts': len([e for e in events if e['event_type'] == 'ACCESS_ATTEMPT']),
            'successful_access': len([e for e in events if e['success']]),
            'failed_access': len([e for e in events if not e['success']]),
            'unique_users': len(set(e['user_id'] for e in events)),
            'generated_at': datetime.utcnow()
        }

        return report

    def _get_user_permissions(self, user_id):
        """Get user permissions (mock implementation)"""
        # In real implementation, query from database
        return ['read', 'write']

    def _check_resource_access(self, permissions, resource, action):
        """Check if action is allowed on resource"""
        # Implement business logic for access control
        return action in permissions

    def _store_audit_event(self, event):
        """Store audit event in database"""
        # Implementation would store in audit database
        pass

    def _get_audit_events(self, start_date, end_date):
        """Retrieve audit events for reporting"""
        # Implementation would query audit database
        return []

    def _get_client_ip(self):
        """Get client IP address"""
        # Implementation would get from request context
        return "192.168.1.1"

    def _get_user_agent(self):
        """Get user agent string"""
        # Implementation would get from request headers
        return "Mozilla/5.0"

    def _get_session_id(self):
        """Get session ID"""
        # Implementation would get from session context
        return "session_123"
```

## Incident Response

### Security Incident Handling

```python
from datetime import datetime
import json

class IncidentResponse:
    def __init__(self, notification_service, audit_logger):
        self.notification = notification_service
        self.audit_logger = audit_logger

    def handle_security_incident(self, incident_type, details, severity='medium'):
        """Handle security incident according to response plan"""
        incident_id = self._generate_incident_id()

        incident = {
            'id': incident_id,
            'type': incident_type,
            'severity': severity,
            'details': details,
            'detected_at': datetime.utcnow(),
            'status': 'investigating',
            'assigned_to': None,
            'escalation_level': self._determine_escalation(severity)
        }

        # Log incident
        self.audit_logger.critical(f"SECURITY_INCIDENT: {incident_id}", extra=incident)

        # Notify response team
        self._notify_response_team(incident)

        # Take immediate actions based on incident type
        self._take_immediate_actions(incident_type, details)

        # Store incident record
        self._store_incident(incident)

        return incident_id

    def _determine_escalation(self, severity):
        """Determine escalation level based on severity"""
        escalation_map = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        return escalation_map.get(severity, 2)

    def _notify_response_team(self, incident):
        """Notify appropriate response team members"""
        subject = f"Security Incident - {incident['severity'].upper()}: {incident['type']}"

        message = f"""
        Security Incident Detected

        Incident ID: {incident['id']}
        Type: {incident['type']}
        Severity: {incident['severity']}
        Detected: {incident['detected_at']}

        Details: {json.dumps(incident['details'], indent=2)}

        Please investigate immediately.
        """

        # Send to different channels based on severity
        if incident['severity'] in ['high', 'critical']:
            self.notification.send_sms(subject, message)
            self.notification.send_email(subject, message, priority='high')
        else:
            self.notification.send_email(subject, message)

    def _take_immediate_actions(self, incident_type, details):
        """Take immediate security actions"""
        actions = {
            'unauthorized_access': self._handle_unauthorized_access,
            'data_breach': self._handle_data_breach,
            'malware_detected': self._handle_malware,
            'ddos_attack': self._handle_ddos
        }

        action_func = actions.get(incident_type, self._handle_generic_incident)
        action_func(details)

    def _handle_unauthorized_access(self, details):
        """Handle unauthorized access incident"""
        user_id = details.get('user_id')
        if user_id:
            # Revoke all tokens for user
            self._revoke_user_tokens(user_id)

            # Block user account temporarily
            self._block_user_account(user_id, duration_hours=24)

    def _handle_data_breach(self, details):
        """Handle data breach incident"""
        # Encrypt affected data
        self._encrypt_affected_data(details.get('affected_records', []))

        # Notify affected users
        self._notify_affected_users(details.get('affected_users', []))

        # Start forensic analysis
        self._initiate_forensic_analysis(details)

    def _handle_malware(self, details):
        """Handle malware detection"""
        # Isolate affected systems
        self._isolate_systems(details.get('affected_systems', []))

        # Scan for malware
        self._scan_for_malware()

        # Update security signatures
        self._update_security_signatures()

    def _handle_ddos(self, details):
        """Handle DDoS attack"""
        # Enable DDoS protection
        self._enable_ddos_protection()

        # Scale up infrastructure
        self._scale_infrastructure()

        # Update WAF rules
        self._update_waf_rules(details.get('attack_pattern'))

    def _handle_generic_incident(self, details):
        """Handle generic security incident"""
        # Log additional details
        self.audit_logger.warning("Generic security incident handled", extra=details)

    def _generate_incident_id(self):
        """Generate unique incident ID"""
        return f"INC-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"

    def _store_incident(self, incident):
        """Store incident record for tracking"""
        # Implementation would store in incident database
        pass

    # Helper methods (implementations would vary based on infrastructure)
    def _revoke_user_tokens(self, user_id): pass
    def _block_user_account(self, user_id, duration_hours): pass
    def _encrypt_affected_data(self, records): pass
    def _notify_affected_users(self, users): pass
    def _initiate_forensic_analysis(self, details): pass
    def _isolate_systems(self, systems): pass
    def _scan_for_malware(self): pass
    def _update_security_signatures(self): pass
    def _enable_ddos_protection(self): pass
    def _scale_infrastructure(self): pass
    def _update_waf_rules(self, pattern): pass
```

## Best Practices Summary

### Development Security
1. **Input Validation**: Validate all inputs to prevent injection attacks
2. **Output Encoding**: Encode outputs to prevent XSS attacks
3. **Authentication**: Use strong authentication mechanisms
4. **Authorization**: Implement proper access controls
5. **Session Management**: Secure session handling
6. **Error Handling**: Don't expose sensitive information in errors

### Infrastructure Security
1. **Network Segmentation**: Isolate different components
2. **Access Control**: Least privilege principle
3. **Monitoring**: Comprehensive security monitoring
4. **Patching**: Keep systems updated
5. **Backup**: Regular secure backups
6. **Disaster Recovery**: Tested recovery procedures

### Compliance
1. **Data Protection**: GDPR, CCPA compliance
2. **Audit Logging**: Comprehensive activity logging
3. **Access Reviews**: Regular permission reviews
4. **Training**: Security awareness training
5. **Incident Response**: Documented response procedures

### Continuous Security
1. **Security Testing**: Regular penetration testing
2. **Vulnerability Scanning**: Automated vulnerability detection
3. **Code Reviews**: Security-focused code reviews
4. **Threat Modeling**: Regular threat assessments
5. **Security Metrics**: Track security KPIs

---

**Last Updated**: November 2024
**Version**: 1.0.0
