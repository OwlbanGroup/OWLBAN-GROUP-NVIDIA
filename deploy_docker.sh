#!/bin/bash

# JPMorgan Financial APIs - Docker Compose Production Deployment Script
# This script deploys the application using Docker Compose for production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
PROJECT_NAME="jpmorgan_prod"

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

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed. Please install docker-compose first."
        exit 1
    fi

    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_warning ".env file not found. Creating template..."
        create_env_template
        log_error "Please update the .env file with your production values and run the script again."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

create_env_template() {
    cat > .env << EOF
# JPMorgan Financial APIs - Production Environment Variables
# Update these values with your production configuration

# Database Configuration
DATABASE_PASSWORD=change_this_in_production

# Redis Configuration
REDIS_PASSWORD=change_this_in_production

# Application Secrets
SECRET_KEY=change_this_in_production_secret_key
TOKEN_CLIENT_ID=your_client_id
TOKEN_CLIENT_SECRET=your_client_secret
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token

# Optional: Docker Registry
# DOCKER_REGISTRY=your-registry.com
# DOCKER_TAG=latest
EOF
}

load_environment() {
    log_info "Loading environment variables..."

    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
        log_success "Environment variables loaded"
    else
        log_error ".env file not found"
        exit 1
    fi
}

validate_compose_file() {
    log_info "Validating Docker Compose configuration..."

    if ! docker-compose -f ${COMPOSE_FILE} config > /dev/null; then
        log_error "Docker Compose configuration is invalid"
        exit 1
    fi

    log_success "Docker Compose configuration is valid"
}

build_images() {
    log_info "Building Docker images..."

    docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} build

    log_success "Docker images built successfully"
}

start_services() {
    log_info "Starting services..."

    docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} up -d

    log_success "Services started successfully"
}

wait_for_services() {
    log_info "Waiting for services to be healthy..."

    # Wait for PostgreSQL
    log_info "Waiting for PostgreSQL..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} exec -T postgres pg_isready -U telemetry_user -d telemetry_db > /dev/null 2>&1; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done

    if [ $timeout -le 0 ]; then
        log_error "PostgreSQL failed to start"
        exit 1
    fi

    # Wait for Redis
    log_info "Waiting for Redis..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} exec -T redis redis-cli -a ${REDIS_PASSWORD} ping | grep -q PONG; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done

    if [ $timeout -le 0 ]; then
        log_error "Redis failed to start"
        exit 1
    fi

    # Wait for API
    log_info "Waiting for API service..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:5000/health > /dev/null 2>&1; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done

    if [ $timeout -le 0 ]; then
        log_error "API service failed to start"
        exit 1
    fi

    log_success "All services are healthy"
}

run_health_checks() {
    log_info "Running health checks..."

    # Test health endpoint
    if curl -f http://localhost:5000/health; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi

    # Test database connectivity (if API exposes a DB health endpoint)
    if curl -f http://localhost:5000/health/db 2>/dev/null; then
        log_success "Database health check passed"
    else
        log_warning "Database health check endpoint not available or failed"
    fi
}

show_deployment_info() {
    log_info "Deployment Information:"
    echo ""
    echo "Services:"
    docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} ps
    echo ""
    echo "API URL: http://localhost:5000"
    echo "Health Check: http://localhost:5000/health"
    echo ""
    echo "To view logs:"
    echo "  docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} logs -f"
    echo ""
    echo "To stop services:"
    echo "  docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} down"
    echo ""
    log_success "Deployment completed successfully!"
}

stop_services() {
    log_info "Stopping services..."

    docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} down

    log_success "Services stopped"
}

cleanup() {
    log_info "Cleaning up..."

    # Remove stopped containers
    docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} down --remove-orphans

    # Remove unused images
    docker image prune -f

    log_success "Cleanup completed"
}

backup_data() {
    log_info "Creating data backup..."

    BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p ${BACKUP_DIR}

    # Backup PostgreSQL data
    docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} exec -T postgres pg_dump -U telemetry_user telemetry_db > ${BACKUP_DIR}/postgres_backup.sql

    # Backup Redis data (if needed)
    # docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} exec -T redis redis-cli -a ${REDIS_PASSWORD} --rdb ${BACKUP_DIR}/redis_backup.rdb

    log_success "Backup created in ${BACKUP_DIR}"
}

# Main script
main() {
    echo "JPMorgan Financial APIs - Docker Compose Production Deployment"
    echo "=========================================================="

    # Parse command line arguments
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            load_environment
            validate_compose_file
            build_images
            start_services
            wait_for_services
            run_health_checks
            show_deployment_info
            ;;
        "stop")
            load_environment
            stop_services
            ;;
        "restart")
            load_environment
            stop_services
            start_services
            wait_for_services
            run_health_checks
            show_deployment_info
            ;;
        "status")
            load_environment
            show_deployment_info
            ;;
        "logs")
            load_environment
            docker-compose -f ${COMPOSE_FILE} -p ${PROJECT_NAME} logs -f
            ;;
        "backup")
            load_environment
            backup_data
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "Usage: $0 {deploy|stop|restart|status|logs|backup|cleanup}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy the application"
            echo "  stop     - Stop all services"
            echo "  restart  - Restart all services"
            echo "  status   - Show deployment status"
            echo "  logs     - Show service logs"
            echo "  backup   - Create data backup"
            echo "  cleanup  - Clean up containers and images"
            exit 1
            ;;
    esac
}

# Trap cleanup function
trap cleanup EXIT

# Run main function
main "$@"
