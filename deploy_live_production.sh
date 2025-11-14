#!/bin/bash

################################################################################
# JPMorgan Financial APIs - Live Production Deployment Script
################################################################################
# This script deploys the application to live production environment
# with all necessary checks, backups, and monitoring setup
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="jpmorgan-financial-apis"
APP_DIR="/opt/jpmorgan-financial-apis"
BACKUP_DIR="/backups/jpmorgan"
LOG_DIR="/var/log/jpmorgan"
VENV_DIR="$APP_DIR/venv"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

create_backup() {
    log_info "Creating backup before deployment..."
    mkdir -p "$BACKUP_DIR"
    
    if [ -d "$APP_DIR" ]; then
        tar -czf "$BACKUP_DIR/backup_$TIMESTAMP.tar.gz" -C "$APP_DIR" . 2>/dev/null || true
        log_success "Backup created: $BACKUP_DIR/backup_$TIMESTAMP.tar.gz"
    else
        log_warning "No existing installation to backup"
    fi
}

setup_directories() {
    log_info "Setting up directories..."
    mkdir -p "$APP_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p /etc/ssl/jpmorgan
    log_success "Directories created"
}

install_dependencies() {
    log_info "Installing system dependencies..."
    
    # Update package list
    apt-get update -qq
    
    # Install required packages
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        postgresql \
        postgresql-contrib \
        redis-server \
        nginx \
        supervisor \
        openssl \
        curl \
        git \
        build-essential \
        libpq-dev \
        python3-dev
    
    log_success "System dependencies installed"
}

setup_database() {
    log_info "Setting up PostgreSQL database..."
    
    # Start PostgreSQL if not running
    systemctl start postgresql
    systemctl enable postgresql
    
    # Create database and user
    sudo -u postgres psql -c "CREATE DATABASE jpmorgan_financial_apis_prod;" 2>/dev/null || true
    sudo -u postgres psql -c "CREATE USER jpmorgan_prod WITH PASSWORD 'SecureP@ssw0rd2024';" 2>/dev/null || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE jpmorgan_financial_apis_prod TO jpmorgan_prod;" 2>/dev/null || true
    
    log_success "Database setup complete"
}

setup_redis() {
    log_info "Setting up Redis..."
    
    # Start Redis
    systemctl start redis-server
    systemctl enable redis-server
    
    # Configure Redis for production
    sed -i 's/^# maxmemory .*/maxmemory 256mb/' /etc/redis/redis.conf
    sed -i 's/^# maxmemory-policy .*/maxmemory-policy allkeys-lru/' /etc/redis/redis.conf
    
    systemctl restart redis-server
    
    log_success "Redis setup complete"
}

deploy_application() {
    log_info "Deploying application..."
    
    # Copy application files
    cp -r jpmorgan_financial_apis/* "$APP_DIR/"
    
    # Create virtual environment
    python3 -m venv "$VENV_DIR"
    
    # Activate virtual environment and install dependencies
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r "$APP_DIR/requirements.txt"
    
    # Copy production environment file
    cp jpmorgan_financial_apis/.env.production "$APP_DIR/.env"
    
    log_success "Application deployed"
}

setup_ssl() {
    log_info "Setting up SSL certificates..."
    
    # Generate self-signed certificate for initial setup
    python3 "$APP_DIR/scripts/setup_https.py" \
        --action generate-self-signed \
        --domain localhost \
        --cert-dir /etc/ssl/jpmorgan \
        --days 365
    
    log_success "SSL certificates generated"
    log_warning "Replace with proper CA-signed certificates for production!"
}

setup_nginx() {
    log_info "Configuring NGINX..."
    
    cat > /etc/nginx/sites-available/jpmorgan-api << 'EOF'
server {
    listen 80;
    server_name _;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name _;

    ssl_certificate /etc/ssl/jpmorgan/server.crt;
    ssl_certificate_key /etc/ssl/jpmorgan/server.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
EOF

    # Enable site
    ln -sf /etc/nginx/sites-available/jpmorgan-api /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Test and reload NGINX
    nginx -t
    systemctl restart nginx
    systemctl enable nginx
    
    log_success "NGINX configured"
}

setup_supervisor() {
    log_info "Configuring Supervisor..."
    
    cat > /etc/supervisor/conf.d/jpmorgan-api.conf << EOF
[program:jpmorgan-api]
command=$VENV_DIR/bin/gunicorn -w 4 -k gevent --bind 0.0.0.0:8000 app:app
directory=$APP_DIR
user=www-data
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=$LOG_DIR/gunicorn.err.log
stdout_logfile=$LOG_DIR/gunicorn.out.log
environment=PATH="$VENV_DIR/bin"
EOF

    # Reload supervisor
    supervisorctl reread
    supervisorctl update
    
    log_success "Supervisor configured"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Install Prometheus Node Exporter
    wget -q https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz
    tar xzf node_exporter-1.6.1.linux-amd64.tar.gz
    cp node_exporter-1.6.1.linux-amd64/node_exporter /usr/local/bin/
    rm -rf node_exporter-1.6.1.linux-amd64*
    
    # Create systemd service
    cat > /etc/systemd/system/node_exporter.service << 'EOF'
[Unit]
Description=Node Exporter
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/node_exporter
Restart=always

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl start node_exporter
    systemctl enable node_exporter
    
    log_success "Monitoring setup complete"
}

run_migrations() {
    log_info "Running database migrations..."
    
    source "$VENV_DIR/bin/activate"
    cd "$APP_DIR"
    
    # Run any migration scripts
    if [ -f "scripts/postgresql_migration.py" ]; then
        python scripts/postgresql_migration.py
    fi
    
    log_success "Migrations complete"
}

start_application() {
    log_info "Starting application..."
    
    supervisorctl start jpmorgan-api
    
    # Wait for application to start
    sleep 5
    
    log_success "Application started"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Check if application is responding
    if curl -f -k https://localhost/health > /dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
    
    # Run validation script
    source "$VENV_DIR/bin/activate"
    python "$APP_DIR/scripts/prod_validation.py" --url https://localhost
    
    log_success "All health checks passed"
}

setup_firewall() {
    log_info "Configuring firewall..."
    
    # Install UFW if not present
    apt-get install -y ufw
    
    # Configure firewall rules
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow ssh
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw allow 9090/tcp  # Prometheus
    ufw --force enable
    
    log_success "Firewall configured"
}

setup_logrotate() {
    log_info "Setting up log rotation..."
    
    cat > /etc/logrotate.d/jpmorgan-api << 'EOF'
/var/log/jpmorgan/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        supervisorctl restart jpmorgan-api > /dev/null 2>&1 || true
    endscript
}
EOF

    log_success "Log rotation configured"
}

print_summary() {
    echo ""
    echo "=========================================="
    echo "  DEPLOYMENT SUMMARY"
    echo "=========================================="
    echo ""
    log_success "Application deployed successfully!"
    echo ""
    echo "Application URL: https://localhost"
    echo "Health Check: https://localhost/health"
    echo "API Docs: https://localhost/docs"
    echo ""
    echo "Logs:"
    echo "  - Application: $LOG_DIR/gunicorn.out.log"
    echo "  - Errors: $LOG_DIR/gunicorn.err.log"
    echo "  - NGINX: /var/log/nginx/"
    echo ""
    echo "Management Commands:"
    echo "  - Start: supervisorctl start jpmorgan-api"
    echo "  - Stop: supervisorctl stop jpmorgan-api"
    echo "  - Restart: supervisorctl restart jpmorgan-api"
    echo "  - Status: supervisorctl status jpmorgan-api"
    echo ""
    log_warning "IMPORTANT: Update the following before going live:"
    echo "  1. Replace SSL certificates with CA-signed certificates"
    echo "  2. Update SECRET_KEY in .env"
    echo "  3. Update database credentials"
    echo "  4. Configure proper domain name in NGINX"
    echo "  5. Set up external monitoring and alerting"
    echo "  6. Configure backup automation"
    echo ""
}

################################################################################
# Main Deployment Flow
################################################################################

main() {
    log_info "Starting JPMorgan Financial APIs Production Deployment"
    echo ""
    
    check_root
    create_backup
    setup_directories
    install_dependencies
    setup_database
    setup_redis
    deploy_application
    setup_ssl
    setup_nginx
    setup_supervisor
    setup_monitoring
    run_migrations
    start_application
    setup_firewall
    setup_logrotate
    
    # Run health checks
    if run_health_checks; then
        print_summary
        exit 0
    else
        log_error "Deployment completed but health checks failed"
        log_error "Check logs at $LOG_DIR"
        exit 1
    fi
}

# Run main deployment
main
