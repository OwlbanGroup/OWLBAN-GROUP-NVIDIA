#!/bin/bash

# Production Deployment Script for JPMorgan Financial APIs
# This script handles the complete production deployment process

set -e  # Exit on any error

# Configuration
PROJECT_NAME="jpmorgan-financial-apis"
DEPLOY_DIR="/opt/${PROJECT_NAME}"
BACKUP_DIR="/opt/${PROJECT_NAME}/backups"
LOG_DIR="/var/log/${PROJECT_NAME}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/backup_${TIMESTAMP}.tar.gz"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if running as root or with sudo
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root - this is not recommended for production"
    fi
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if required ports are available
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        log_error "Port 8000 is already in use. Please stop the service or use a different port."
        exit 1
    fi

    log_success "Pre-deployment checks passed"
}

# Create backup
create_backup() {
    log_info "Creating backup..."

    if [ -d "$DEPLOY_DIR" ]; then
        mkdir -p "$BACKUP_DIR"
        tar -czf "$BACKUP_FILE" -C /opt "$PROJECT_NAME" 2>/dev/null || true
        log_success "Backup created: $BACKUP_FILE"
    else
        log_info "No existing deployment found, skipping backup"
    fi
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."

    sudo mkdir -p "$DEPLOY_DIR"
    sudo mkdir -p "$LOG_DIR"
    sudo mkdir -p "$BACKUP_DIR"

    # Set proper permissions
    sudo chown -R $USER:$USER "$DEPLOY_DIR"
    sudo chown -R $USER:$USER "$LOG_DIR"
    sudo chown -R $USER:$USER "$BACKUP_DIR"

    log_success "Directories created and permissions set"
}

# Copy application files
copy_application() {
    log_info "Copying application files..."

    # Copy all necessary files
    cp -r . "$DEPLOY_DIR/"

    # Remove development files
    cd "$DEPLOY_DIR"
    rm -rf .git .vscode __pycache__ *.pyc
    rm -f .env*  # Remove all .env files, will be configured separately

    log_success "Application files copied"
}

# Configure environment
configure_environment() {
    log_info "Configuring production environment..."

    cd "$DEPLOY_DIR"

    # Copy production environment file
    if [ ! -f ".env.production" ]; then
        log_error "Production environment file not found. Please create .env.production"
        exit 1
    fi

    cp .env.production .env

    # Generate secure secret key if not set
    if grep -q "SECRET_KEY=your-production-secret-key-change-this" .env; then
        SECRET_KEY=$(openssl rand -hex 32)
        sed -i "s|SECRET_KEY=your-production-secret-key-change-this|SECRET_KEY=$SECRET_KEY|" .env
        log_success "Generated secure secret key"
    fi

    # Set proper permissions on environment file
    chmod 600 .env

    log_success "Environment configured"
}

# Build and start services
start_services() {
    log_info "Building and starting services..."

    cd "$DEPLOY_DIR"

    # Pull latest images
    docker-compose -f docker-compose.prod.yml pull

    # Build custom images
    docker-compose -f docker-compose.prod.yml build --no-cache

    # Start services
    docker-compose -f docker-compose.prod.yml up -d

    log_success "Services started"
}

# Health checks
health_check() {
    log_info "Performing health checks..."

    # Wait for services to be ready
    sleep 30

    # Check if containers are running
    if ! docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        log_error "Some containers failed to start"
        docker-compose -f docker-compose.prod.yml logs
        exit 1
    fi

    # Check API health
    max_attempts=10
    attempt=1

    while [ $attempt -le $max_attempts ]; do
        log_info "Health check attempt $attempt/$max_attempts..."

        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "API health check passed"
            break
        else
            if [ $attempt -eq $max_attempts ]; then
                log_error "API health check failed after $max_attempts attempts"
                docker-compose -f docker-compose.prod.yml logs jpmorgan-apis
                exit 1
            fi
            sleep 10
            ((attempt++))
        fi
    done

    log_success "All health checks passed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."

    cd "$DEPLOY_DIR"

    # Check if monitoring services are running
    if docker-compose -f docker-compose.prod.yml ps | grep -q "prometheus"; then
        log_success "Prometheus is running"
    fi

    if docker-compose -f docker-compose.prod.yml ps | grep -q "grafana"; then
        log_success "Grafana is running"
    fi

    if docker-compose -f docker-compose.prod.yml ps | grep -q "alertmanager"; then
        log_success "AlertManager is running"
    fi
}

# Setup SSL (optional)
setup_ssl() {
    log_info "Setting up SSL certificates..."

    # This is a placeholder for SSL setup
    # In production, you would use certbot or similar
    log_info "SSL setup skipped - configure manually with certbot or your certificate provider"
}

# Post-deployment tasks
post_deployment() {
    log_info "Running post-deployment tasks..."

    # Setup log rotation
    setup_log_rotation

    # Setup backup cron job
    setup_backup_cron

    # Print deployment information
    print_deployment_info
}

# Setup log rotation
setup_log_rotation() {
    log_info "Setting up log rotation..."

    sudo tee /etc/logrotate.d/${PROJECT_NAME} > /dev/null <<EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml restart jpmorgan-apis
    endscript
}
EOF

    log_success "Log rotation configured"
}

# Setup backup cron job
setup_backup_cron() {
    log_info "Setting up backup cron job..."

    # Add backup cron job (daily at 2 AM)
    (crontab -l ; echo "0 2 * * * $DEPLOY_DIR/scripts/backup.sh") | crontab -

    log_success "Backup cron job configured"
}

# Print deployment information
print_deployment_info() {
    log_success "Deployment completed successfully!"
    echo
    echo "=================================================================="
    echo "JPMorgan Financial APIs Production Deployment"
    echo "=================================================================="
    echo
    echo "API Endpoint:     http://localhost:8000"
    echo "Health Check:     http://localhost:8000/health"
    echo "Swagger Docs:     http://localhost:8000/swagger/"
    echo "Grafana:          http://localhost:3000"
    echo "Prometheus:       http://localhost:9090"
    echo "AlertManager:     http://localhost:9093"
    echo
    echo "Logs:             $LOG_DIR"
    echo "Backups:          $BACKUP_DIR"
    echo "Application:      $DEPLOY_DIR"
    echo
    echo "To check status:  docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml ps"
    echo "To view logs:     docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml logs -f"
    echo "To restart:       docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml restart"
    echo "To stop:          docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml down"
    echo
    echo "=================================================================="
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."

    cd "$DEPLOY_DIR"

    # Stop services
    docker-compose -f docker-compose.prod.yml down

    # Restore backup if available
    if [ -f "$BACKUP_FILE" ]; then
        log_info "Restoring from backup..."
        rm -rf "$DEPLOY_DIR"
        mkdir -p "$DEPLOY_DIR"
        tar -xzf "$BACKUP_FILE" -C /opt
        log_success "Backup restored"
    fi

    # Restart services
    if [ -f "docker-compose.prod.yml" ]; then
        docker-compose -f docker-compose.prod.yml up -d
    fi

    log_success "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."

    # Remove old backups (keep last 7)
    cd "$BACKUP_DIR"
    ls -t *.tar.gz 2>/dev/null | tail -n +8 | xargs -r rm -f

    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting JPMorgan Financial APIs production deployment..."

    check_permissions
    pre_deployment_checks
    create_backup
    setup_directories
    copy_application
    configure_environment
    start_services
    health_check
    setup_monitoring
    setup_ssl
    post_deployment
    cleanup

    log_success "Production deployment completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    "rollback")
        rollback
        ;;
    "cleanup")
        cleanup
        ;;
    "status")
        cd "$DEPLOY_DIR" && docker-compose -f docker-compose.prod.yml ps
        ;;
    "logs")
        cd "$DEPLOY_DIR" && docker-compose -f docker-compose.prod.yml logs -f
        ;;
    "restart")
        cd "$DEPLOY_DIR" && docker-compose -f docker-compose.prod.yml restart
        ;;
    "stop")
        cd "$DEPLOY_DIR" && docker-compose -f docker-compose.prod.yml down
        ;;
    *)
        main
        ;;
esac
