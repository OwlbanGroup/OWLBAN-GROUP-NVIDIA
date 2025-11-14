#!/usr/bin/env python3
"""
HTTPS Configuration Setup Script for JPMorgan Financial APIs

This script helps configure HTTPS/SSL for the application including:
- Generating self-signed certificates for development
- Validating existing certificates
- Setting up Let's Encrypt certificates
- Configuring SSL/TLS settings
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


class HTTPSConfigurator:
    """HTTPS/SSL configuration manager"""

    def __init__(self, cert_dir="/etc/ssl/jpmorgan"):
        self.cert_dir = Path(cert_dir)
        self.cert_path = self.cert_dir / "server.crt"
        self.key_path = self.cert_dir / "server.key"
        self.csr_path = self.cert_dir / "server.csr"

    def create_cert_directory(self):
        """Create certificate directory if it doesn't exist"""
        try:
            self.cert_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Certificate directory created: {self.cert_dir}")
            return True
        except PermissionError:
            print(
                f"✗ Permission denied: Cannot create {self.cert_dir}. "
                f"Try running with sudo."
            )
            return False
        except OSError as e:
            print(f"✗ Error creating certificate directory: {e}")
            return False

    def generate_self_signed_cert(
        self,
        domain="localhost",
        days=365,
        country="US",
        state="NY",
        city="New York",
        org="JPMorgan",
        unit="IT"
    ):
        """Generate self-signed SSL certificate"""
        print("\n=== Generating Self-Signed Certificate ===")

        if not self.create_cert_directory():
            return False

        # Create OpenSSL configuration
        subject = (
            f"/C={country}/ST={state}/L={city}/O={org}/OU={unit}/"
            f"CN={domain}"
        )

        try:
            # Generate private key
            print("Generating private key...")
            subprocess.run(
                [
                    "openssl", "genrsa",
                    "-out", str(self.key_path),
                    "2048"
                ],
                check=True,
                capture_output=True
            )
            print(f"✓ Private key generated: {self.key_path}")

            # Generate certificate
            print("Generating certificate...")
            subprocess.run(
                [
                    "openssl", "req",
                    "-new",
                    "-x509",
                    "-key", str(self.key_path),
                    "-out", str(self.cert_path),
                    "-days", str(days),
                    "-subj", subject
                ],
                check=True,
                capture_output=True
            )
            print(f"✓ Certificate generated: {self.cert_path}")

            # Set proper permissions
            os.chmod(self.key_path, 0o600)
            os.chmod(self.cert_path, 0o644)

            print("\n✓ Self-signed certificate created successfully!")
            print(f"  Certificate: {self.cert_path}")
            print(f"  Private Key: {self.key_path}")
            print(f"  Valid for: {days} days")
            print(
                "\n⚠ Note: Self-signed certificates are for "
                "development only."
            )
            print(
                "  For production, use certificates from a trusted CA "
                "or Let's Encrypt."
            )

            return True

        except subprocess.CalledProcessError as e:
            print(f"✗ Error generating certificate: {e}")
            print(
                "  Make sure OpenSSL is installed: "
                "apt-get install openssl (Linux) or "
                "brew install openssl (macOS)"
            )
            return False
        except (OSError, ValueError) as e:
            print(f"✗ Unexpected error: {e}")
            return False

    def generate_csr(
        self,
        domain,
        country="US",
        state="NY",
        city="New York",
        org="JPMorgan",
        unit="IT",
        email="admin@jpmorgan.com"
    ):
        """Generate Certificate Signing Request for CA"""
        print("\n=== Generating Certificate Signing Request ===")

        if not self.create_cert_directory():
            return False

        subject = (
            f"/C={country}/ST={state}/L={city}/O={org}/OU={unit}/"
            f"CN={domain}/emailAddress={email}"
        )

        try:
            # Generate private key if it doesn't exist
            if not self.key_path.exists():
                print("Generating private key...")
                subprocess.run(
                    [
                        "openssl", "genrsa",
                        "-out", str(self.key_path),
                        "2048"
                    ],
                    check=True,
                    capture_output=True
                )
                os.chmod(self.key_path, 0o600)
                print(f"✓ Private key generated: {self.key_path}")

            # Generate CSR
            print("Generating CSR...")
            subprocess.run(
                [
                    "openssl", "req",
                    "-new",
                    "-key", str(self.key_path),
                    "-out", str(self.csr_path),
                    "-subj", subject
                ],
                check=True,
                capture_output=True
            )
            print(f"✓ CSR generated: {self.csr_path}")

            print("\n✓ Certificate Signing Request created successfully!")
            print(f"  CSR File: {self.csr_path}")
            print(f"  Private Key: {self.key_path}")
            print(
                "\n→ Submit the CSR to your Certificate Authority "
                "to obtain a signed certificate."
            )

            return True

        except subprocess.CalledProcessError as e:
            print(f"✗ Error generating CSR: {e}")
            return False
        except (OSError, ValueError) as e:
            print(f"✗ Unexpected error: {e}")
            return False

    def validate_certificate(self):
        """Validate existing SSL certificate"""
        print("\n=== Validating SSL Certificate ===")

        if not self.cert_path.exists():
            print(f"✗ Certificate not found: {self.cert_path}")
            return False

        if not self.key_path.exists():
            print(f"✗ Private key not found: {self.key_path}")
            return False

        try:
            # Check certificate validity
            result = subprocess.run(
                [
                    "openssl", "x509",
                    "-in", str(self.cert_path),
                    "-noout",
                    "-dates"
                ],
                check=True,
                capture_output=True,
                text=True
            )

            print("✓ Certificate is valid")
            print(result.stdout)

            # Check certificate details
            result = subprocess.run(
                [
                    "openssl", "x509",
                    "-in", str(self.cert_path),
                    "-noout",
                    "-subject",
                    "-issuer"
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)

            # Verify key matches certificate
            cert_modulus = subprocess.run(
                [
                    "openssl", "x509",
                    "-noout",
                    "-modulus",
                    "-in", str(self.cert_path)
                ],
                check=True,
                capture_output=True,
                text=True
            ).stdout

            key_modulus = subprocess.run(
                [
                    "openssl", "rsa",
                    "-noout",
                    "-modulus",
                    "-in", str(self.key_path)
                ],
                check=True,
                capture_output=True,
                text=True
            ).stdout

            if cert_modulus == key_modulus:
                print("✓ Private key matches certificate")
            else:
                print("✗ Private key does not match certificate")
                return False

            return True

        except subprocess.CalledProcessError as e:
            print(f"✗ Certificate validation failed: {e}")
            return False
        except (OSError, ValueError) as e:
            print(f"✗ Unexpected error: {e}")
            return False

    def setup_letsencrypt(self, domain, email):
        """Setup Let's Encrypt certificate using certbot"""
        print("\n=== Setting up Let's Encrypt Certificate ===")

        try:
            # Check if certbot is installed
            subprocess.run(
                ["certbot", "--version"],
                check=True,
                capture_output=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Certbot is not installed")
            print(
                "\nInstall certbot:"
            )
            print("  Ubuntu/Debian: sudo apt-get install certbot")
            print("  CentOS/RHEL: sudo yum install certbot")
            print("  macOS: brew install certbot")
            return False

        print(
            f"Setting up Let's Encrypt certificate for {domain}..."
        )
        print(
            "\n⚠ Note: This requires:"
        )
        print("  1. Domain must be publicly accessible")
        print("  2. Port 80 must be open")
        print("  3. DNS must point to this server")
        print(
            "\nRun the following command manually with appropriate "
            "permissions:"
        )
        print(
            f"\nsudo certbot certonly --standalone -d {domain} "
            f"--email {email} --agree-tos"
        )
        print(
            "\nAfter obtaining the certificate, update your "
            ".env.production:"
        )
        print(
            f"  SSL_CERT_PATH=/etc/letsencrypt/live/{domain}/fullchain.pem"
        )
        print(
            f"  SSL_KEY_PATH=/etc/letsencrypt/live/{domain}/privkey.pem"
        )

        return True

    def generate_nginx_config(self, domain, app_port=8000):
        """Generate NGINX configuration for HTTPS"""
        config = f"""
# NGINX Configuration for JPMorgan Financial APIs
# Save this to /etc/nginx/sites-available/jpmorgan-api

server {{
    listen 80;
    server_name {domain};
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name {domain};

    # SSL Configuration
    ssl_certificate {self.cert_path};
    ssl_certificate_key {self.key_path};
    
    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Proxy Settings
    location / {{
        proxy_pass http://localhost:{app_port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket Support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }}

    # Static Files (if any)
    location /static {{
        alias /app/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }}

    # Health Check Endpoint
    location /health {{
        proxy_pass http://localhost:{app_port}/health;
        access_log off;
    }}
}}
"""
        print("\n=== NGINX Configuration ===")
        print(config)
        print(
            "\nSave this configuration and enable it:"
        )
        print(
            "  sudo ln -s /etc/nginx/sites-available/jpmorgan-api "
            "/etc/nginx/sites-enabled/"
        )
        print("  sudo nginx -t")
        print("  sudo systemctl reload nginx")

        return config


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='HTTPS Configuration Setup for JPMorgan Financial APIs'
    )
    parser.add_argument(
        '--action',
        choices=[
            'generate-self-signed',
            'generate-csr',
            'validate',
            'letsencrypt',
            'nginx-config'
        ],
        required=True,
        help='Action to perform'
    )
    parser.add_argument(
        '--domain',
        default='localhost',
        help='Domain name for the certificate'
    )
    parser.add_argument(
        '--email',
        help='Email address for Let\'s Encrypt'
    )
    parser.add_argument(
        '--cert-dir',
        default='/etc/ssl/jpmorgan',
        help='Directory to store certificates'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Certificate validity in days (for self-signed)'
    )
    parser.add_argument(
        '--app-port',
        type=int,
        default=8000,
        help='Application port for NGINX config'
    )

    args = parser.parse_args()

    configurator = HTTPSConfigurator(args.cert_dir)

    if args.action == 'generate-self-signed':
        success = configurator.generate_self_signed_cert(
            domain=args.domain,
            days=args.days
        )
        sys.exit(0 if success else 1)

    elif args.action == 'generate-csr':
        success = configurator.generate_csr(domain=args.domain)
        sys.exit(0 if success else 1)

    elif args.action == 'validate':
        success = configurator.validate_certificate()
        sys.exit(0 if success else 1)

    elif args.action == 'letsencrypt':
        if not args.email:
            print("✗ Email is required for Let's Encrypt")
            sys.exit(1)
        success = configurator.setup_letsencrypt(
            domain=args.domain,
            email=args.email
        )
        sys.exit(0 if success else 1)

    elif args.action == 'nginx-config':
        configurator.generate_nginx_config(
            domain=args.domain,
            app_port=args.app_port
        )
        sys.exit(0)


if __name__ == '__main__':
    main()
