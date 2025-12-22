#!/bin/bash

# JPMorgan Financial APIs - Production Deployment Script
# This script deploys the complete stack with monitoring and security

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="jpmorgan-financial-apis"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.production"

# Functions
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

check_dependencies() {
    log_info "Checking dependencies..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker service."
        exit 1
    fi

    log_success "All dependencies are available"
}

create_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating production environment file..."

        cat > "$ENV_FILE" << EOF
# JPMorgan Financial APIs - Production Environment Variables
# IMPORTANT: Change all default values before deploying to production!

# Flask Configuration
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=\${SECRET_KEY}

# Database Configuration
DATABASE_URL=postgresql://jpmorgan:\${DB_PASSWORD}@db:5432/jpmorgan_api
DB_PASSWORD=CHANGE_THIS_STRONG_PASSWORD

# Redis Configuration
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=

# API Configuration
API_BASE_URL=https://api.jpmorgan.com
API_VERSION=v1

# JPMorgan Integration (Use actual credentials)
JPMORGAN_OPENBANKING_CLIENT_ID=your_client_id
JPMORGAN_OPENBANKING_CLIENT_SECRET=your_client_secret
JPMORGAN_OPENBANKING_API_KEY=your_api_key

# Token Management
TOKEN_CLIENT_ID=your_token_client_id
TOKEN_CLIENT_SECRET=your_token_client_secret
TOKEN_URL=https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/app.log

# Telemetry
TELEMETRY_ENABLED=true
TELEMETRY_BATCH_SIZE=100

# Security
ALLOWED_ORIGINS=https://app.jpmorgan.com,https://dashboard.jpmorgan.com

# Audit Logging
AUDIT_LOG_ENABLED=true
AUDIT_LOG_RETENTION_DAYS=90
AUDIT_ALERT_ENABLED=true

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=CHANGE_THIS_ADMIN_PASSWORD

# Email Alerts (for AlertManager)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@jpmorgan.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL_RECIPIENT=ops@jpmorgan.com
EOF

        log_warning "Created $ENV_FILE with default values. Please update with production values!"
        log_warning "Especially change SECRET_KEY, passwords, and API credentials!"
        read -p "Press Enter after updating the environment file..."
    else
        log_info "Environment file $ENV_FILE already exists"
    fi
}

build_and_deploy() {
    log_info "Building and deploying $PROJECT_NAME..."

    # Build the application image
    log_info "Building Docker images..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache

    # Start the services
    log_info "Starting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30

    # Check service health
    check_services_health
}

check_services_health() {
    log_info "Checking service health..."

    services=("app" "db" "redis" "prometheus" "grafana" "alertmanager" "node-exporter")

    for service in "${services[@]}"; do
        if docker-compose -f "$DOCKER_COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            log_success "$service is running"
        else
            log_error "$service failed to start"
            show_logs "$service"
            exit 1
        fi
    done
}

show_logs() {
    service=$1
    log_info "Showing logs for $service:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs "$service"
}

run_tests() {
    log_info "Running tests..."

    # Run the test suite
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T app python -m pytest --tb=short --cov=jpmorgan_financial_apis --cov-report=term-missing; then
        log_success "All tests passed"
    else
        log_error "Some tests failed"
        exit 1
    fi
}

show_access_info() {
    log_info "Deployment completed successfully!"
    echo ""
    echo "========================================"
    echo "ACCESS INFORMATION"
    echo "========================================"
    echo ""
    echo "Application:"
    echo "  API:          http://localhost:5000"
    echo "  Health Check: http://localhost:5000/health"
    echo "  API Docs:     http://localhost:5000/api/docs/"
    echo ""
    echo "Monitoring:"
    echo "  Grafana:      http://localhost:3000 (admin/\${GRAFANA_ADMIN_PASSWORD})"
    echo "  Prometheus:   http://localhost:9090"
    echo "  AlertManager: http://localhost:9093"
    echo ""
    echo "Databases:"
    echo "  PostgreSQL:   localhost:5432 (jpmorgan/\${DB_PASSWORD})"
    echo "  Redis:        localhost:6379"
    echo ""
    echo "Node Exporter:  http://localhost:9100"
    echo ""
    echo "========================================"
}

cleanup() {
    log_info "Cleaning up..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v --remove-orphans
    log_success "Cleanup completed"
}

# Main script
case "${1:-deploy}" in
    "deploy")
        log_info "Starting production deployment of $PROJECT_NAME"
        check_dependencies
        create_env_file
        build_and_deploy
        run_tests
        show_access_info
        ;;
    "test")
        log_info "Running tests only"
        check_dependencies
        run_tests
        ;;
    "logs")
        service="${2:-app}"
        show_logs "$service"
        ;;
    "stop")
        log_info "Stopping all services"
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
        ;;
    "cleanup")
        cleanup
        ;;
    "restart")
        log_info "Restarting services"
        docker-compose -f "$DOCKER_COMPOSE_FILE" restart
        check_services_health
        ;;
    *)
        echo "Usage: $0 {deploy|test|logs|stop|cleanup|restart}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Full deployment with tests"
        echo "  test     - Run tests only"
        echo "  logs     - Show logs for a service (default: app)"
        echo "  stop     - Stop all services"
        echo "  cleanup  - Stop services and remove volumes"
        echo "  restart  - Restart all services"
        exit 1
        ;;
esac
