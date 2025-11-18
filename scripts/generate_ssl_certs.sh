#!/bin/bash

# Generate self-signed SSL certificates for NGINX
# For production, replace with proper certificates from Let's Encrypt or a CA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SSL_DIR="$PROJECT_DIR/nginx/ssl"

echo "==================================="
echo "Generating Self-Signed SSL Certificates"
echo "==================================="

# Create SSL directory if it doesn't exist
mkdir -p "$SSL_DIR"

# Generate private key
echo "Generating private key..."
openssl genrsa -out "$SSL_DIR/server.key" 2048

# Generate certificate signing request
echo "Generating certificate signing request..."
openssl req -new -key "$SSL_DIR/server.key" -out "$SSL_DIR/server.csr" \
    -subj "/C=US/ST=New York/L=New York/O=JPMorgan Chase/OU=IT/CN=localhost"

# Generate self-signed certificate (valid for 365 days)
echo "Generating self-signed certificate..."
openssl x509 -req -days 365 -in "$SSL_DIR/server.csr" \
    -signkey "$SSL_DIR/server.key" -out "$SSL_DIR/server.crt"

# Set proper permissions
chmod 600 "$SSL_DIR/server.key"
chmod 644 "$SSL_DIR/server.crt"

echo ""
echo "✓ SSL certificates generated successfully!"
echo "  - Private Key: $SSL_DIR/server.key"
echo "  - Certificate: $SSL_DIR/server.crt"
echo ""
echo "⚠️  WARNING: These are self-signed certificates for testing only!"
echo "   For production, use certificates from Let's Encrypt or a trusted CA."
echo ""
